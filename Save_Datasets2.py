import os
import sys
import struct
import math
import numpy as np
import csv
import random
import time
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch, quat_mul, quat_conjugate
from ur3e_utils import quat_axis, orientation_error, cube_grasping_yaw, control_ik, control_osc, create_simulation, create_viewer, configure_ur3e_dof, create_assets, create_envs
from multiprocessing import Process, Queue
import threading
import torch
# print necessary datas and convert them into array
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
    # 配置相机属性
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1920  # 设置输出图像宽度为1920像素
    camera_props.height = 1080  # 设置输出图像高度为1080像素
    camera_props.horizontal_fov = 90.0  # 设置水平视场角为90度
    camera_props.enable_tensors = True  # 允许CUDA张量支持
    camera_props.near_plane = 0.1  # 设置近剪裁面距离为0.1
    camera_props.far_plane = 1000.0  # 设置远剪裁面距离为1000
    camera_props.supersampling_horizontal = 2  # 水平方向超采样
    camera_props.supersampling_vertical = 2  # 垂直方向超采样
    camera_props.use_collision_geometry = False  # 不渲染碰撞几何体

    viewer = create_viewer(gym, sim)
    

    # Asset parameters
    # asset_root = os.path.abspath("./assets")  # 絶対パスに変更
    asset_root = "../../assets"
    # table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
    table_dims = gymapi.Vec3(0.5, 0.5, 0.4)
    box_size = 0.03
    ur3e_asset_file = "urdf/ur_description/urdf/ur3e.urdf"

    # Create assets
    table_asset, box_asset,_, ur3e_asset, pillar1_asset ,pillar2_asset ,pillar3_asset ,pillar4_asset ,destination_asset= create_assets(gym, sim, asset_root, table_dims, box_size, ur3e_asset_file)

    # Configure ur3e dofs
    ur3e_link_dict = gym.get_asset_rigid_body_dict(ur3e_asset)
    ur3e_hand_index = ur3e_link_dict["hand_e_link"]
    ur3e_dof_props = gym.get_asset_dof_properties(ur3e_asset)
    ur3e_dof_props = configure_ur3e_dof(controller, ur3e_dof_props)

    # Default dof states and position targets
    ur3e_num_dofs = gym.get_asset_dof_count(ur3e_asset)
    default_dof_pos = np.zeros(ur3e_num_dofs, dtype=np.float32)
    default_dof_pos[:6] = [3.1415, -1.3, -1.4, -3.1415/2, 3.1415/2, -0.0237]
    # default_dof_pos[:6] = [3.1415, -3.1415/2, -3.1415/2, -3.1415/2, 3.1415/2, 0]
    default_dof_pos[6:] = ur3e_dof_props["lower"][6:]

    default_dof_state = np.zeros(ur3e_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos

    default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

    # Create environments
    num_envs = args.num_envs
    num_per_row = int(math.sqrt(num_envs))
    spacing = 1.0

    envs, box_idxs,  pillar1_idxs,pillar2_idxs,pillar3_idxs,pillar4_idxs, hand_idxs, init_pos_list, init_rot_list ,destination_idxs= create_envs(
        gym = gym, sim=sim, num_envs = num_envs, num_per_row = num_per_row, spacing = spacing, 
        table_asset = table_asset, box_asset = box_asset, destination_asset = destination_asset,
        pillar1_asset= pillar1_asset, pillar2_asset= pillar2_asset,pillar3_asset= pillar3_asset,pillar4_asset= pillar4_asset,ur3e_asset = ur3e_asset,
        ur3e_dof_props = ur3e_dof_props, default_dof_state = default_dof_state, default_dof_pos = default_dof_pos, 
        table_dims = table_dims, box_size = box_size,viewer= viewer) # add viewer here to adjust camera position
    
    # 定义 destination 上表面的区域
    destination_region_x = 0.5  # destination 的 x 坐标
    destination_region_y = 0.2  # destination 的 y 坐标
    destination_region_z = table_dims.z + 0.5 * box_size  # destination 的 z 坐标（上表面）
    destination_region_width = 0.1  # destination 上表面的区域宽度
    destination_region_depth = 0.1  # destination 上表面的区域深度
    destination_region_height = 0.01  # destination 上表面的区域高度（从上表面开始计算）

    # 定义区域的边界框
    destination_region_min = gymapi.Vec3(destination_region_x - destination_region_width / 2,
                                        destination_region_y - destination_region_depth / 2,
                                        destination_region_z)
    destination_region_max = gymapi.Vec3(destination_region_x + destination_region_width / 2,
                                        destination_region_y + destination_region_depth / 2,
                                        destination_region_z + destination_region_height)
 
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


    # For finger distance, rigid body handle
    env = envs[0]
    actor_handle = gym.find_actor_handle(env, "ur3e")

    # Get all joints names of UR3E
    joint_names = gym.get_asset_joint_names(ur3e_asset)
    print("Joint names:", joint_names)

    # get rigid body index
    left_finger_idx = gym.find_actor_rigid_body_index(env, actor_handle, "hande_left_finger", gymapi.DOMAIN_SIM)
    right_finger_idx = gym.find_actor_rigid_body_index(env, actor_handle, "hande_right_finger", gymapi.DOMAIN_SIM)
    
    '''Pillar falling function needs to be developed'''

    '''CSV Dataset Creation'''
    # 初始化 CSV 文件
    csv_file = open('dataset_2.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['episode', 'step', 'hand_pos_array', 'cube_pos_array',
                        'pillar1_pos_array', 'pillar2_pos_array', 'pillar3_pos_array', 'pillar4_pos_array',
                        'destination_pos_array', 'finger_distance_array', 'pos_action_array'])  
    
    episode = 0
    max_episodes = 2  # 设定最大 episode 数量


    # Simulation loop
    while episode < max_episodes:
        step = 0
        while not gym.query_viewer_has_closed(viewer):
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            gym.refresh_rigid_body_state_tensor(sim)
            gym.refresh_dof_state_tensor(sim)
            gym.refresh_jacobian_tensors(sim)
            gym.refresh_mass_matrix_tensors(sim)

            # 获取 cube 的位置
            cube_pos_array = rb_states[box_idxs, :3].cpu().numpy()
            
            '''
            State for dataset
            gripper position, table position, cube position, 
            pillar1 position,pillar2 position,pillar3 position,pillar4 position, 
            distination position, finger distance
            '''
            # ur3e End effector position
            hand_pos_array = rb_states[hand_idxs[0], :3].cpu().numpy()
            # print("hand position arry for dataset",hand_pos_array)

            # table position not successful
            # table_pos = rb_states[gym.get_actor_rigid_body_index(env, table_asset, 0, gymapi.DOMAIN_SIM), :3].cpu().numpy()

            # cube position 
            cube_pos_array = rb_states[box_idxs, :3].cpu().numpy().squeeze()
            # print("cube position array for ",cube_pos_array)

            # pillars positions 
            pillar1_pos_array = rb_states[pillar1_idxs, :3].cpu().numpy().squeeze()
            pillar2_pos_array = rb_states[pillar2_idxs, :3].cpu().numpy().squeeze()
            pillar3_pos_array = rb_states[pillar3_idxs, :3].cpu().numpy().squeeze()
            pillar4_pos_array = rb_states[pillar4_idxs, :3].cpu().numpy().squeeze()

            # pillar1_rot_array = rb_states[pillar1_idxs, 3:7].cpu().numpy().squeeze()
            # pillar2_rot_array = rb_states[pillar2_idxs, 3:7].cpu().numpy().squeeze()
            # pillar3_rot_array = rb_states[pillar3_idxs, 3:7].cpu().numpy().squeeze()
            # pillar4_rot_array = rb_states[pillar4_idxs, 3:7].cpu().numpy().squeeze()

            #print("pillar1_pos_array",pillar1_pos_array)
            #print("pillar2_pos_array",pillar2_pos_array)
            #print("pillar3_pos_array",pillar3_pos_array)
            #print("pillar4_pos_array",pillar4_pos_array)
            # print(pillar1_rot_array)
            # destination position 
            destination_pos_array = rb_states[destination_idxs, :3].cpu().numpy()
            #print("destination position array",destination_pos_array)

            left_finger_pos = rb_states[left_finger_idx, :3]
            right_finger_pos = rb_states[right_finger_idx, :3]
            #  calculate distance between 2 fingers of the gripper
            finger_distance = torch.norm(left_finger_pos - right_finger_pos, p=2).cpu().numpy()
            #print(finger_distance)

            box_pos = rb_states[box_idxs, :3]
            box_rot = rb_states[box_idxs, 3:7]
            hand_pos = rb_states[hand_idxs, :3]
            hand_rot = rb_states[hand_idxs, 3:7]
            hand_vel = rb_states[hand_idxs, 7:]

            # pillar_pos = _states[box_idxs, :3]
            # box_rot = rb_states[box_idxs, 3:7]

            # ジョイスティックの入力を取得
            try:
                # print(joystick_queue.get())
                # cannot use queue.get  here, destroy isaac gym
                type, code, value = joystick_queue.get_nowait()
                # print(joystick_queue.get_nowait())
                # ** pay attention to this **
                # print(f"Joystick event: type={type}, code={code}, value={value}")
                if type == JS_EVENT_AXIS:
                    # ジョイスティックの軸に基づいて目標位置を更新
                    if code == 1:  # 前後方向の軸
                        # print("Axis 0 moved to {value}")
                        goal_pos[:, 0] += value / 32767.0 * 0.005  # スケーリングして位置を更新
                    elif code == 0:  #  左右方向の軸
                        # print("Axis 1 moved to {value}")
                        goal_pos[:, 1] += value / 32767.0 * 0.005
                    elif code == 3:  # 上下方向の軸
                        # print("Axis 3 moved to {value}")
                        goal_pos[:, 2] += value / 32767.0 * 0.005
                elif type == JS_EVENT_BUTTON:
                    # ジョイスティックのボタンに基づいてグリッパを制御
                    if code == 0 and value == 1:  # ボタン0が押された
                        pos_action[:, 6:8] = torch.Tensor([[0.025, 0.025]] * num_envs).to(device)  # グリッパを閉じる
                        # print("sent position action1",pos_action)
                    elif code == 1 and value == 1:  # ボタン1が押された
                        pos_action[:, 6:8] = torch.Tensor([[0.0, 0.0]] * num_envs).to(device)  # グリッパを開く
                        # print("sent position action2",pos_action)
                    elif code == 3 and value == 1:
                        goal_pos = init_pos.clone()
            except:
                pass

            pos_err = goal_pos - hand_pos
            orn_err = orientation_error(init_rot, hand_rot)  # goal_rotをinit_rotに変更
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

            if controller == "ik":
                pos_action[:, :6] = dof_pos.squeeze(-1)[:, :6] + control_ik(dpose, j_eef, num_envs, device)
                pos_action_array = pos_action[:, :6].cpu().numpy().squeeze()
            else:
                effort_action[:, :6] = control_osc(dpose, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel, kp, kd, kp_null, kd_null, default_dof_pos_tensor, device)


            # 将数组转换为字符串
            hand_pos_str = str(hand_pos_array.tolist())
            cube_pos_str = str(cube_pos_array.tolist())
            pillar1_pos_str = str(pillar1_pos_array.tolist())
            pillar2_pos_str = str(pillar2_pos_array.tolist())
            pillar3_pos_str = str(pillar3_pos_array.tolist())
            pillar4_pos_str = str(pillar4_pos_array.tolist())
            destination_pos_str = str(destination_pos_array.tolist())
            finger_distance_str = str(finger_distance)
            pos_action_str = str(pos_action.tolist())

            # 写入 CSV 文件
            csv_writer.writerow([episode, step, hand_pos_str, cube_pos_str,
                                pillar1_pos_str, pillar2_pos_str, pillar3_pos_str, pillar4_pos_str,
                                destination_pos_str, finger_distance_str, pos_action_str])

            #print("sent position action",pos_action)
            '''
            sent position action tensor([[ 2.9687, -1.3991, -1.3259, -1.5515,  1.6443, -0.1803,  0.0000,  0.0000]],device='cuda:0')
            '''
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)
            # print("Simulation step completed")
            

            # 确保 cube_pos_array 是一个一维数组
            if cube_pos_array.ndim > 1:
                cube_pos_array = cube_pos_array.squeeze()

            # 检查 cube 是否在 destination 上表面的区域内
            if (destination_region_min.x <= cube_pos_array[0] <= destination_region_max.x and
                    destination_region_min.y <= cube_pos_array[1] <= destination_region_max.y and
                    destination_region_min.z <= cube_pos_array[2] <= destination_region_max.z):
                print(f"Episode {episode + 1}, Step {step + 1}: Arrived at destination surface area")
                episode += 1

            if episode >= max_episodes:
                break  # 退出外层循环

            step += 1
    # 关闭 CSV 文件
    csv_file.close()
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()