from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import math

# 初始化gym接口
gym = gymapi.acquire_gym()

# 设置模拟参数
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2

# 创建模拟器
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# 创建观众
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# 设置桌子的大小和立方体的大小、数量
table_dims = gymapi.Vec3(1.0, 1.0, 0.4)  # 桌子的长宽高
num_cubes = 5  # 用户可以在这里指定立方体数量
cube_size = 0.1  # 立方体的边长

# 创建桌子模型
table_asset_options = gymapi.AssetOptions()
table_asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)

# 创建立方体模型
cube_asset_options = gymapi.AssetOptions()
cube_asset = gym.create_box(sim, cube_size, cube_size, cube_size, cube_asset_options)

# 配置环境
num_envs = 1
spacing = 1.5  # 环境间距
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# 创建环境
env = gym.create_env(sim, env_lower, env_upper, int(math.sqrt(num_envs)))

# 添加桌子到环境中
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
table_handle = gym.create_actor(env, table_asset, table_pose, "table", 0, 0)

# 添加立方体到桌子上
for i in range(num_cubes):
    cube_pose = gymapi.Transform()
    # 随机分布立方体的位置，但确保它们位于桌子上方
    cube_pose.p.x = np.random.uniform(-0.4, 0.4)
    cube_pose.p.y = np.random.uniform(-0.4, 0.4)
    cube_pose.p.z = table_dims.z + 0.5 * cube_size
    cube_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    
    cube_handle = gym.create_actor(env, cube_asset, cube_pose, f"cube_{i}", 0, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

# 运行模拟
while not gym.query_viewer_has_closed(viewer):
    # Step simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

# 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)