from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time

# Import the required functions from ur3e_utils.py
from ur3e_utils import create_simulation, create_viewer, create_assets, create_envs, configure_ur3e_dof

# 手动设置参数而不是通过命令行解析
sim_params = gymapi.SimParams()

# 创建模拟环境
gym = gymapi.acquire_gym()  # 初始化 gym
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)  # 使用 PhysX 物理引擎

# 定义桌子和方块的尺寸
table_dims = gymapi.Vec3(1.0, 1.0, 0.1)  # 桌子尺寸 (长, 宽, 高)
box_size = 0.1  # 方块的边长

# 创建资产
asset_root = "../../assets"
ur3e_asset_file = "urdf/ur_description/urdf/ur3e.urdf"

table_asset, box_asset, box2_asset, ur3e_asset = create_assets(gym, sim, asset_root, table_dims, box_size, ur3e_asset_file)


# 创建环境
num_envs = 1  # 只创建一个环境
num_per_row = 1
spacing = 2.0

envs, box_idxs, hand_idxs, init_pos_list, init_rot_list = create_envs(
    gym, sim, num_envs, num_per_row, spacing, table_asset, box_asset, box2_asset, ur3e_asset, table_dims, box_size
)

# 在桌子上添加多个方块
num_cubes = 5  # 你想要的方块数量
for i in range(num_cubes):
    # 设置方块位置
    box_pose = gymapi.Transform()
    box_pose.p.x = 0.3 + i * 0.1  # 根据需要调整位置
    box_pose.p.y = 0.0  # 桌子上Y轴位置
    box_pose.p.z = table_dims.z + box_size / 2  # 在桌子上方
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)  # 不旋转

    # 创建方块
    box_handle = gym.create_actor(envs[0], box_asset, box_pose, f"box_{i}", 0, 0)

# 在模拟中添加一个地面
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# 运行模拟
while True:
    gym.step_sim(sim)


