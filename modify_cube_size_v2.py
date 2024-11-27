from isaacgym import gymapi
from isaacgym import gymutil
import math
import numpy as np

# Acquire gym interface
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments()


# Set simulation parameters
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
    sim_params.physx.contact_offset = 0.001
else:
    raise Exception("This example requires PhysX physics engine.")

# Create simulation
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create simulation")

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Set asset options
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True

# Create table asset
table_dims = gymapi.Vec3(0.6, 1.0, 0.4)
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# Create box asset
box_size = 0.03
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

# Set environment parameters
num_envs = 2  # Modify this to change the number of environments
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Initialize actor transforms
table_pose = gymapi.Transform()
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

box_pose = gymapi.Transform()

envs = []

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# Create environments and add table and boxes
for i in range(num_envs):
    # Create environment
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # Add table
    gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # Add cubes on the table
    for _ in range(5):  # Number of cubes per table, adjust as needed
        box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.2)
        box_pose.p.y = table_pose.p.y + np.random.uniform(-0.2, 0.2)
        box_pose.p.z = table_dims.z + 0.5 * box_size
        box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))

        box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
        
        # Set random color for each cube
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

# Simulation loop
while not gym.query_viewer_has_closed(viewer):
    # Step the physics simulation
    gym.simulate(sim)
    # gym.fetch_results(sim, True)

    # Update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

# Clean up
# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)
