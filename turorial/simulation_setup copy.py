from isaacgym import gymapi # core api
from isaacgym import gymtorch # pytorch interop

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