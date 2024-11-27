
import sys, os
import numpy as np

from isaacgym import gymapi, gymutil

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0

sim_params.use_gpu_pipeline = False


# parse arguments
args = gymutil.parse_arguments(description="Joint monkey: Animate degree-of-freedom ranges")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)


# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

#################################

asset_root = '../../assets'
asset_file = "urdf/ur_description/urdf/ur3e.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True
asset_options.use_mesh_materials = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


# set up the env grid
num_envs = 10
num_per_row = 2
# spacing = 2.5
spacing = 1.
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
# cam_pos = gymapi.Vec3(3, 2.0, 2)
cam_pos = gymapi.Vec3(6, 4.0, 8)
cam_target = gymapi.Vec3(-10, -5, 1)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


num_dofs = gym.get_asset_dof_count(asset)

# cache useful handles
envs = []
actor_handles = []
dof_states = []

print("Creating %d environments" % num_envs)
for i in range(num_envs):
    dof_state = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
    dof_states.append(dof_state)

    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, dof_state, gymapi.STATE_ALL)




for i in range(num_envs):
    dof_states[i]['pos'][1] = np.random.rand()*np.pi


while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    for i in range(num_envs):
        gym.set_actor_dof_states(envs[i], actor_handles[i], dof_states[i], gymapi.STATE_POS)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)



gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
