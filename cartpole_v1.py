
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import torch
import numpy as np

class CartPole:
    def __init__(self, cfg):
        # 超参数
        self.max_push_force = 10.0  # 最大施加力
        self.dt = 0.02  # 时间步长
        self.num_envs = 16  # 环境数量
        self.dof_per_env = 2  # 每个环境的自由度数量

        # 设备设置
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 初始化张量
        self.obs_buf = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.rew_buf = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        # 创建模拟
        self.create_sim()

    def create_sim(self):
        # 创建模拟
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX)

        # 定义地面
        ground_plane = gymapi.PlaneParams()
        self.gym.add_ground(self.sim, ground_plane)

        # 加载CartPole资产
        asset_root = "./assets"  # 确保路径正确
        cart_asset = self.gym.load_asset(self.sim, asset_root, "cartpole.urdf", gymapi.AssetOptions())

        # 创建环境和演员
        self.envs = []
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 2)
            cart_handle = self.gym.create_actor(env, cart_asset, gymapi.Transform(), "cartpole", i, 1, 0)
            self.envs.append(env)

        # 初始化DOF状态张量
        self.dof_states_desc = gymtorch.wrap_tensor(self.gym.get_dof_state_tensor(self.sim))

    def pre_physics_step(self, actions):
        # 将控制力施加到小车上
        forces = torch.zeros((self.num_envs, self.dof_per_env), dtype=torch.float32, device=self.device)
        forces[:, 0] = actions * self.max_push_force  # 将动作施加到第一个DOF（小车）
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(forces))

    def post_physics_step(self):
        self.compute_observations()
        self.compute_reward()
        self.reset()  # 检查并重置环境（如果需要）

    def compute_observations(self):
        # 刷新状态张量
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # 将DOF状态复制到观测张量
        self.obs_buf[:, 0] = self.dof_states_desc[:, 0]  # 小车位置
        self.obs_buf[:, 1] = self.dof_states_desc[:, 1]  # 小车速度
        self.obs_buf[:, 2] = self.dof_states_desc[:, 2]  # 杆子角度
        self.obs_buf[:, 3] = self.dof_states_desc[:, 3]  # 杆子角速度

    def compute_reward(self):
        cart_pos = self.obs_buf[:, 0]
        cart_vel = self.obs_buf[:, 1]
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]

        self.rew_buf[:] = self.compute_cartpole_reward(cart_pos, cart_vel, pole_angle, pole_vel)

    @torch.jit.script
    def compute_cartpole_reward(cart_pos, cart_vel, pole_angle, pole_vel):
        angle_penalty = pole_angle * pole_angle
        vel_penalty = 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        return 1.0 - angle_penalty - vel_penalty

    def reset(self, env_ids=None):
        # 重置选定的环境
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        num_resets = len(env_ids)
        
        # 生成随机的DOF位置和速度
        p = 0.3 * (torch.rand((num_resets, self.dof_per_env), device=self.device) - 0.5)
        v = 0.5 * (torch.rand((num_resets, self.dof_per_env), device=self.device) - 0.5)
        
        # 为选定的环境应用新的DOF状态
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_states_desc), gymtorch.unwrap_tensor(env_ids), num_resets)

# 运行CartPole实验的示例
if __name__ == "__main__":
    gym_config = gymutil.parse_arguments()
    task = CartPole(gym_config)
    
    # 主模拟循环
    while not task.gym.should_step():
        actions = torch.zeros(task.num_envs, dtype=torch.float32, device=task.device)  # 替换为实际动作
        task.pre_physics_step(actions)
        task.post_physics_step()
        task.gym.step()

    task.gym.destroy_sim(task.sim)
