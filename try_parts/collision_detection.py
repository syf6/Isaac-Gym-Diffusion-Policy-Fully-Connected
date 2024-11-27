# Part 1
import os
import sys
import struct
import math
import numpy as np

import random
import time
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch, quat_mul, quat_conjugate
# from ur3e_utils import quat_axis, orientation_error, cube_grasping_yaw, control_ik, control_osc, create_simulation, create_viewer, configure_ur3e_dof, create_assets, create_envs
from multiprocessing import Process, Queue
import threading
import torch

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



# Part 2
# 设置环境边界
env_lower = gymapi.Vec3(-1, -1, -1)
env_upper = gymapi.Vec3(1, 1, 1)
env = gym.create_env(sim, env_lower, env_upper, 1)

# 定义桌子和Box的属性
asset_options_table = gymapi.AssetOptions()
asset_options_table.fix_base_link = True   # 固定桌子位置
asset_options_table.use_mesh_materials = True

asset_options_box = gymapi.AssetOptions()
asset_options_box.fix_base_link = False    # 允许Box自由移动
asset_options_box.use_mesh_materials = True

# 加载桌子和Box资产
table_asset = gym.create_box(sim, 0.5, 0.5, 0.1, asset_options_table)  # 桌子的尺寸
box_asset = gym.create_box(sim, 0.1, 0.1, 0.1, asset_options_box)       # Box的尺寸

# 设置桌子和Box的初始位置
table_pose = gymapi.Transform()
table_pose.p.z = 0.05  # 桌子高度稍高于地面

box_pose = gymapi.Transform()
box_pose.p.z = 0.3     # 将Box置于桌子上方，确保其在重力作用下可以下落

# 将资产添加到环境中
table_actor = gym.create_actor(env, table_asset, table_pose, "table", 0, 1)
box_actor = gym.create_actor(env, box_asset, box_pose, "box", 0, 1)


# Part 3
# 仿真步循环
for i in range(1000):  # 假设仿真1000步
    # 执行一步仿真
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 获取接触信息
    contacts = gym.get_env_rigid_contacts(env)  # 不需要传入 sim，只传入 env

    # 遍历每个接触事件
    for contact in contacts:
        actor1 = contact.actor_handle[0]
        actor2 = contact.actor_handle[1]
        contact_pos = contact.contact_pos  # 假设 `contact_pos` 是接触点的位置
        contact_force = contact.contact_force  # 假设 `contact_force` 是接触力大小

        # 判断是否是Box和桌子的碰撞
        if (actor1 == box_actor and actor2 == table_actor) or (actor1 == table_actor and actor2 == box_actor):
            print(f"Step {i}: Collision detected at position ({contact_pos.x}, {contact_pos.y}, {contact_pos.z}) with force magnitude {contact_force.norm()}")


# Part 4
# 仿真结束后，销毁资源
gym.destroy_sim(sim)
