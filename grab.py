
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch

# 初始化Isaac Gym
gym = gymapi.acquire_gym()
device = "cuda:0"  # 使用GPU设备

# 机器人设置
robot = None  # 这里设置你的机器人对象
current_pose = torch.zeros(7).to(device)  # 当前位姿 [x, y, z, roll, pitch, yaw, gripper]
desired_orientation = torch.zeros(4).to(device)  # 当前方向的四元数

# 夹具上下限
ur3e_lower_limits = np.array([-1.57, -1.57, -1.57, -1.57, -1.57, -1.57, 0.0])
ur3e_upper_limits = np.array([1.57, 1.57, 1.57, 1.57, 1.57, 1.57, 1.0])

# 手柄输入读取函数
def get_joystick_input():
    # 这里实现手柄输入读取
    joystick_input = {
        'x': 0.0,  # 控制X轴移动
        'y': 0.0,  # 控制Y轴移动
        'z': 0.0,  # 控制Z轴移动
        'roll': 0.0,  # 控制Roll旋转
        'pitch': 0.0,  # 控制Pitch旋转
        'yaw': 0.0,  # 控制Yaw旋转
        'gripper': 0.0  # 控制夹具
    }
    return joystick_input

# 控制机器人末端执行器
def control_ik(dpose):
    # 这里实现逆运动学控制
    return dpose  # 这个示例中直接返回输入值作为控制输出

# 控制循环
while not gym.query_viewer_has_closed(viewer):
    # 获取手柄输入
    joystick_input = get_joystick_input()

    # 解析手柄输入
    dx = joystick_input['x'] * 0.01  # X轴移动
    dy = joystick_input['y'] * 0.01  # Y轴移动
    dz = joystick_input['z'] * 0.01  # Z轴移动
    droll = joystick_input['roll'] * 0.1  # Roll旋转
    dpitch = joystick_input['pitch'] * 0.1  # Pitch旋转
    dyaw = joystick_input['yaw'] * 0.1  # Yaw旋转
    gripper_action = joystick_input['gripper']  # 夹具控制

    # 更新末端执行器的位置和方向
    hand_pos = current_pose[:3] + torch.tensor([dx, dy, dz]).to(device)
    
    # 计算新的方向
    # 这里需要用到四元数更新方向
    desired_orientation += torch.tensor([droll, dpitch, dyaw]).to(device)

    # 计算目标姿态
    desired_pose = torch.cat([hand_pos, desired_orientation], dim=-1)  # 期望姿态

    # 发送控制命令到机器人
    dpose = desired_pose - current_pose  # 计算增量
    control_output = control_ik(dpose)  # 进行逆运动学控制

    # 夹具控制
    target_gripper_pos = ur3e_lower_limits[6:] + gripper_action * (ur3e_upper_limits[6:] - ur3e_lower_limits[6:])
    control_output[6:] = target_gripper_pos  # 更新控制输出中的夹具目标位置

    # 发送控制命令
    pos_action[:] = control_output

    # 继续模拟
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # 更新当前位姿
    current_pose = control_output  # 更新当前位姿为控制输出

    # 其他逻辑保持不变
