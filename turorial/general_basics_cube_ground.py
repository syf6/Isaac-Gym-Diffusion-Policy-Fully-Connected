# SIMULATION SETUP
from isaacgym import gymapi # core api
from isaacgym import gymtorch # pytorch interop

import os


# set simulation parameters
sim_params = gymapi.SimParams()
sim_params.dt = 0.01 # time for each step
sim_params.physx.use_gpu = True

#...

# enable GPU pipeline
sim_params.use_gpu_pipeline = True
# acquire gym interface
gym = gymapi.acquire_gym()

# create new simulation
sim = gym.create_sim(0,0,gymapi.SIM_PHYSX, sim_params)

# add ground 

plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0)  # set ground at z=0
gym.add_ground(sim, plane_params)




# ENVIRONMENT CREATION

# prepare assets
asset_root = "../../assets"
num_envs = 10 

# load ur3e asset
'''
ur3e_asset_file = "urdf/ur_description/urdf/ur3e.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = True
asset_options.use_mesh_materials = True
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
asset_options.vhacd_enabled = True
asset_options.convex_decomposition_from_submeshes = True
ur3e_asset = gym.load_asset(sim, asset_root, ur3e_asset_file, asset_options)
'''

# franka_assets = gym.load_asset(sim, "../assets","urdf/franka_urdf",franka_options)
# create box asset
box_options = gymapi.AssetOptions()
box_options.density = 1.0  # density of the box

box_asset = gym.create_box(sim, 0.1, 0.1, 0.1, box_options)

# set box initial pose
box_pose = gymapi.Transform()
box_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)  # start the box 0.5 meters above the ground




# create environments and add actors
envs = []
spacing = 1.5  # spacing between environments
lower = gymapi.Vec3(-spacing, -spacing, 0.0)
upper = gymapi.Vec3(spacing, spacing, spacing)

for i in range(num_envs):
    # create new env ??
    env = gym.create_env(sim,lower, upper, num_envs)
    envs.append(env)

    # create actors from assets
    # franka = gym.create_actor(env, fraqnka_asset, franka_pose, ...)
    # create box actor
    box_actor = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
    gym.set_rigid_body_color(env, box_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.0, 0.5, 1.0))  # set color to blue

    # configure initgial poses, drives and other properties
    # ...





# require to use tensor API: prepare simulation buffers and tensor storage 
gym.prepare_sim(sim)
# details in slides, different tensors

# prepare viwer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

if viewer is None:
    print("Failed to create viewer")
    quit()

# Simulation loop
while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)  # 传递 viewer 对象

    # 处理 UI 事件
    gym.sync_frame_time(sim)

# 清理资源
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)