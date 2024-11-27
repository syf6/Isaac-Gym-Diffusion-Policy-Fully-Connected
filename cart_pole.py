from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time


# SIMPLE RL TASK FRAMEWORK
class CartPole(BaseTask):
    # Common callbacks

    # create environment and actor
    def create_sim(self):

    # Apply controls
    def pre_physics_step(self,actions):
    
    # Compute observations, rewards and resets
    def post_physics_step(self):

    
    
    # GETTING OBESERVATION
    def compute_observations(self):

        # Refresh state tensor
        self.gym.refresh_dof_state_tensor(self.sim)

        # copy DOF states to observation tensor :  position, linear velocity, angle position, angular velocity
        self.obs_buf[:,0] = self.dof_pos[:,0]
        self.obs_buf[:,1] = self.dof_vel[:,0]
        self.obs_buf[:,2] = self.dof_pos[:,1]
        self.obs_buf[:,3] = self.dof_vel[:,1]

    

    # COMPUTING REWARDS WITH PYTORCH JIT
    def compute_reward(self):
        cart_pos = self.obs_buf[:,0]
        cart_vel = self.obs_buf[:,1]
        pole_angle = self.obs_buf[:,2]
        pole_vel = self.obs_buf[:,3]

        self.rew_buf[:] = compute_cartpole_reward(cart_pos,cart_vel,pole_angle,pole_vel)

        # Use the TOrchScript JIT compiler to run computations on GPU

        @torch.jit.script
        def compute_cartpole_reward(cart_pos,cart_vel,pole_angle,pole_vel):
            # type: (Tensor, Tensor, Tensor, Tensor) -> Tensor

            angle_penalty = pole_angle* pole_angle
            vel_penalty  = 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)

            return 1.0 - angle_penalty - vel_penalty
    
    # APPLYING ACTIONS 
    def pre_physics_step(step, actions):
        # prepare DOF force tensor
        forces = torch.zeros((num_envs, dof_per_env), dtype = torch.float32, device = self.device)

        # scale actions and write to cart DOF slice
        forces[:,0] = actions * self.max_push_force 

        # apply the forces to all actors
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(forces))

    # RESETTING SELECTED ENVS
    def reset(self, env_ids):

        # number of environments to reset
        num_resets = len(env_ids)

        # generate radnom DOF positions and velocities 
        p = 0.3 * (torch.rand((num_resets, dofs_per_env), device = self.device) - 0.5) 
        v = 0.5 * (torch.rand((num_resets, dof_per_env),device = self.device) - 0.5)

        # apply the new DOF states for the selected envs
        # using env_ids as the actor index tensor
        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_states_desc,gymtorch.unwrap_tensor(env_ids),num_resets)