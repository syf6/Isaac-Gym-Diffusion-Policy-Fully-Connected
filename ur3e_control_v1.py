from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from isaacgym.torch_utils import to_torch, quat_mul, quat_conjugate
from ur3e_utils import quat_axis, orientation_error, cube_grasping_yaw, control_ik, control_osc, create_simulation, create_viewer, configure_ur3e_dof, create_assets, create_envs
from multiprocessing import Process, Queue
import threading
import torch
import os
import sys
import struct
import math
import numpy as np

import random
import time

# ジョイスティックの設定
joystick_path = "/dev/input/js0"
EVENT_FORMAT = 'IhBB'
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

def read_joystick(queue):
    try:
        with open(joystick_path, 'rb') as js_device:
            while True:
                event = js_device.read(EVENT_SIZE)
                if event:
                    time, value, type, code = struct.unpack(EVENT_FORMAT, event)
                    if type & JS_EVENT_INIT:
                        continue  # 初期化イベントを無視
                    queue.put((type, code, value))
    except FileNotFoundError:
        print(f"No such device: {joystick_path}")
    except PermissionError:
        print(f"Permission denied: {joystick_path}")

def main():
    # set random seed
    np.random.seed(42)
    torch.set_printoptions(precision=4, sci_mode=False)

    # parse arguments
    custom_parameters = [
        {"name": "--controller", "type": str, "default": "ik",
         "help": "Controller to use for ur3e. Options are {ik, osc}"},
        {"name": "--num_envs", "type": int, "default": 1, "help": "Number of environments to create"},
    ]
    args = gymutil.parse_arguments(
        description="ur3e Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
        custom_parameters=custom_parameters,
    )

    # Grab controller
    controller = args.controller
    assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

    # set torch device
    device = args.sim_device if args.use_gpu_pipeline else 'cpu'

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
    else:
        raise Exception("This example can only be used with PhysX")

    # Create simulation and viewer
    gym, sim = create_simulation(args, sim_params)
    viewer = create_viewer(gym, sim)

    # Asset parameters
    # asset_root = os.path.abspath("./assets")  # 絶対パスに変更
    asset_root = "../../assets"
    table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
    box_size = 0.03
    ur3e_asset_file = "urdf/ur_description/urdf/ur3e.urdf"

    # Create assets
    table_asset, box_asset,pillar_asset , ur3e_asset = create_assets(gym, sim, asset_root, table_dims, box_size, ur3e_asset_file)

    # Configure ur3e dofs
    ur3e_link_dict = gym.get_asset_rigid_body_dict(ur3e_asset)
    ur3e_hand_index = ur3e_link_dict["hand_e_link"]
    ur3e_dof_props = gym.get_asset_dof_properties(ur3e_asset)
    ur3e_dof_props = configure_ur3e_dof(controller, ur3e_dof_props)

    # Default dof states and position targets
    ur3e_num_dofs = gym.get_asset_dof_count(ur3e_asset)
    default_dof_pos = np.zeros(ur3e_num_dofs, dtype=np.float32)
    default_dof_pos[:6] = [3.1415, -1.5029, -1.4361, -1.1826, 1.6476, -0.0237]
    default_dof_pos[6:] = ur3e_dof_props["lower"][6:]

    default_dof_state = np.zeros(ur3e_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos

    default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

    # Create environments
    num_envs = args.num_envs
    num_per_row = int(math.sqrt(num_envs))
    spacing = 1.0
    
    envs, box_idxs, hand_idxs, init_pos_list, init_rot_list = create_envs(
        gym = gym, sim =sim , num_envs = num_envs, num_per_row = num_per_row, spacing = spacing, table_asset = table_asset, 
        box_asset = box_asset, pillar_asset = pillar_asset, ur3e_asset = ur3e_asset,
        ur3e_dof_props = ur3e_dof_props , default_dof_state = default_dof_state, 
        default_dof_pos = default_dof_pos, table_dims = table_dims, box_size = box_size)

    # Prepare tensors
    gym.prepare_sim(sim)

    init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
    init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

    down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))
    box_half_size = 0.5 * box_size
    corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
    corners = torch.stack(num_envs * [corner_coord]).to(device)
    down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

    _jacobian = gym.acquire_jacobian_tensor(sim, "ur3e")
    jacobian = gymtorch.wrap_tensor(_jacobian)
    j_eef = jacobian[:, ur3e_hand_index - 1, :, :6]

    _massmatrix = gym.acquire_mass_matrix_tensor(sim, "ur3e")
    mm = gymtorch.wrap_tensor(_massmatrix)
    mm = mm[:, :6, :6]

    _rb_states = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(_rb_states)

    _dof_states = gym.acquire_dof_state_tensor(sim)
    dof_states = gymtorch.wrap_tensor(_dof_states)
    dof_pos = dof_states[:, 0].view(num_envs, 8, 1)
    dof_vel = dof_states[:, 1].view(num_envs, 8, 1)

    hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

    pos_action = torch.zeros_like(dof_pos).squeeze(-1)
    effort_action = torch.zeros_like(pos_action)

    kp = 150.
    kd = 2.0 * np.sqrt(kp)
    kp_null = 10.
    kd_null = 2.0 * np.sqrt(kp_null)

    # ジョイスティック読み取りのスレッドを開始
    joystick_queue = Queue()
    joystick_thread = threading.Thread(target=read_joystick, args=(joystick_queue,))
    joystick_thread.daemon = True
    joystick_thread.start()

    # goal_posを初期化
    goal_pos = init_pos.clone()

    # Simulation loop
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        gym.refresh_mass_matrix_tensors(sim)

        box_pos = rb_states[box_idxs, :3]
        box_rot = rb_states[box_idxs, 3:7]
        hand_pos = rb_states[hand_idxs, :3]
        hand_rot = rb_states[hand_idxs, 3:7]
        hand_vel = rb_states[hand_idxs, 7:]

        to_box = box_pos - hand_pos
        box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
        box_dir = to_box / box_dist
        box_dot = box_dir @ down_dir.view(3, 1)

        grasp_offset = 0.12 if controller == "ik" else 0.125
        gripper_sep = 0.049 - (dof_pos[:, 6] + dof_pos[:, 7])
        gripped = (gripper_sep < box_size) & (box_dist < grasp_offset + 0.5 * box_size)

        import ipdb; ipdb.set_trace()
        yaw_q = cube_grasping_yaw(box_rot, corners)
        box_yaw_dir = quat_axis(yaw_q, 0)
        hand_yaw_dir = quat_axis(hand_rot, 0)
        yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

        to_init = init_pos - hand_pos
        init_dist = torch.norm(to_init, dim=-1)
        hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)
        return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

        above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 3)).squeeze(-1)
        grasp_pos = box_pos.clone()
        grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + grasp_offset, box_pos[:, 2] + grasp_offset * 2.5)

        # ジョイスティックの入力を取得
        try:
            type, code, value = joystick_queue.get_nowait()
            print(f"Joystick event: type={type}, code={code}, value={value}")
            if type == JS_EVENT_AXIS:
                # ジョイスティックの軸に基づいて目標位置を更新
                if code == 0:  # 左右方向の軸
                    print("Axis 0 moved to {value}")
                    goal_pos[:, 0] += value / 32767.0 * 0.01  # スケーリングして位置を更新
                elif code == 1:  # 前後方向の軸
                    print("Axis 1 moved to {value}")
                    goal_pos[:, 1] += value / 32767.0 * 0.01
                elif code == 3:  # 上下方向の軸
                    print("Axis 3 moved to {value}")
                    goal_pos[:, 2] += value / 32767.0 * 0.01
            elif type == JS_EVENT_BUTTON:
                # ジョイスティックのボタンに基づいてグリッパを制御
                if code == 0 and value == 1:  # ボタン0が押された
                    pos_action[:, 6:8] = torch.Tensor([[0.025, 0.025]] * num_envs).to(device)  # グリッパを閉じる
                elif code == 1 and value == 1:  # ボタン1が押された
                    pos_action[:, 6:8] = torch.Tensor([[0.0, 0.0]] * num_envs).to(device)  # グリッパを開く
                elif code == 3 and value == 1:
                    goal_pos = init_pos.clone()
        except:
            pass

        pos_err = goal_pos - hand_pos
        orn_err = orientation_error(init_rot, hand_rot)  # goal_rotをinit_rotに変更
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

        if controller == "ik":
            pos_action[:, :6] = dof_pos.squeeze(-1)[:, :6] + control_ik(dpose, j_eef, num_envs, device)
        else:
            effort_action[:, :6] = control_osc(dpose, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel, kp, kd, kp_null, kd_null, default_dof_pos_tensor, device)

        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
        gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)
        # print("Simulation step completed")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()