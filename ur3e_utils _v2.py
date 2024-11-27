from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch
import random
import time



def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    if q_r.dim() == 1:
        q_r = q_r.unsqueeze(0)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose, j_eef, num_envs, device, damping=0.2):
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 6)
    return u


def control_osc(dpose, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel, kp, kd, kp_null, kd_null, default_dof_pos_tensor, device):
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (kp * dpose - kd * hand_vel.unsqueeze(-1))

    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)


def create_simulation(args, sim_params):
    # acquire gym interface
    gym = gymapi.acquire_gym()

    # create sim
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    
    if sim is None:
        raise Exception("Failed to create sim")
    
    return gym, sim


def create_viewer(gym, sim):
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")
    return viewer

def create_pillar_asset(gym,sim,pillar_width,pillar_depth,pillar_height):
    # create pillar asset
    asset_options = gymapi.AssetOptions()
    pillar_asset = gym.create_box(sim,pillar_width,pillar_depth,pillar_height,asset_options)
    return pillar_asset

def create_assets(gym, sim, asset_root, table_dims, box_size, ur3e_asset_file):
    # create table asset
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    # create box asset
    asset_options = gymapi.AssetOptions()
    box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)
    box2_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

    # 4 pillar asset
    pillar1_asset = create_pillar_asset(gym,sim,pillar_width = 0.01,pillar_depth = 0.01,pillar_height=0.15)
    pillar2_asset = create_pillar_asset(gym,sim,pillar_width = 0.01,pillar_depth = 0.01,pillar_height=0.15)
    pillar3_asset = create_pillar_asset(gym,sim,pillar_width = 0.01,pillar_depth = 0.01,pillar_height=0.15)
    pillar4_asset = create_pillar_asset(gym,sim,pillar_width = 0.01,pillar_depth = 0.01,pillar_height=0.15)

    # destination box
    asset_options = gymapi.AssetOptions()
    destination_width =0.1
    destination_depth =0.1
    destination_height =0.01
    destination_asset = gym.create_box(sim, destination_width , destination_depth , destination_height ,asset_options)

    # load ur3e asset
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
    
    return table_asset, box_asset, box2_asset , ur3e_asset ,pillar1_asset, pillar2_asset, pillar3_asset, pillar4_asset,destination_asset


def configure_ur3e_dof(controller, ur3e_dof_props):
    if controller == "ik":
        ur3e_dof_props["driveMode"][:6].fill(gymapi.DOF_MODE_POS)
        ur3e_dof_props["stiffness"][:6].fill(400.0)
        ur3e_dof_props["damping"][:6].fill(40.0)
    else:  # osc
        ur3e_dof_props["driveMode"][:6].fill(gymapi.DOF_MODE_EFFORT)
        ur3e_dof_props["stiffness"][:6].fill(0.0)
        ur3e_dof_props["damping"][:6].fill(0.0)
    ur3e_dof_props["driveMode"][6:].fill(gymapi.DOF_MODE_POS)
    ur3e_dof_props["stiffness"][6:].fill(400.0)
    ur3e_dof_props["damping"][6:].fill(40.0)
    return ur3e_dof_props

def add_pillar_in_create_envs(table_dims,box_size,pillar_pose, env, pillar_asset, i, gym , x,y):
    # add one pillar
    pillar_pose.p.x = x
    pillar_pose.p.y = y
    pillar_pose.p.z = table_dims.z + 0.5 * box_size

    pillar_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    pillar_handle = gym.create_actor(env, pillar_asset, pillar_pose, "pillar", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, pillar_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # get global index of box in rigid body state tensor
    pillar_idx = gym.get_actor_rigid_body_index(env, pillar_handle, 0, gymapi.DOMAIN_SIM)
    return pillar_idx

def create_envs(gym, sim, num_envs, num_per_row, spacing, table_asset, box_asset,destination_asset,pillar1_asset, pillar2_asset, pillar3_asset, pillar4_asset,ur3e_asset, ur3e_dof_props, default_dof_state, default_dof_pos, table_dims, box_size, viewer):
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    ur3e_pose = gymapi.Transform()
    
    ur3e_pose.p = gymapi.Vec3(0, 0, 0.3)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.5, 0, 0.5 * table_dims.z)

    box_pose = gymapi.Transform()

    pillar1_pose = gymapi.Transform()
    pillar2_pose = gymapi.Transform()
    pillar3_pose = gymapi.Transform()
    pillar4_pose = gymapi.Transform()

    destination_pose = gymapi.Transform()

    envs = []
    box_idxs = []
    pillar1_idxs = []
    pillar2_idxs = []
    pillar3_idxs = []
    pillar4_idxs = []
    destination_idxs = []
    hand_idxs = []
    init_pos_list = []
    init_rot_list = []

    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    for i in range(num_envs):
        # create env
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)

        # add camera positon closer
        # 设定相机的位置和朝向
        camera_position = gymapi.Vec3(1.0, 0.0, 1.5)  # 相机的位置 (x, y, z)
        camera_target = gymapi.Vec3(0.0, 0.0, 0.5)    # 相机的目标点，指向机械臂或桌子

        # 使用 viewer_camera_look_at 函数调整相机视角
        gym.viewer_camera_look_at(viewer, env, camera_position, camera_target)

        envs.append(env)

        # add table
        table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

        # add one pillar
        # pillar1_pose.p.x= 0.65
        # pillar1_pose.p.y= -0.2
        # pillar1_pose.p.z = table_dims.z + 0.5 * box_size

        # pillar1_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
        # pillar1_handle = gym.create_actor(env, pillar1_asset, pillar1_pose, "box", i, 0)
        # color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        # gym.set_rigid_body_color(env, pillar1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # # get global index of box in rigid body state tensor
        # pillar1_idx = gym.get_actor_rigid_body_index(env, pillar1_handle, 0, gymapi.DOMAIN_SIM)


        # rectangle shape obstables
        '''
        pillar1_idx = add_pillar_in_create_envs(table_dims,box_size,pillar1_pose, env, pillar1_asset, i, gym, x=0.65, y =-0.2)
        pillar1_idxs.append(pillar1_idx)

        pillar2_idx = add_pillar_in_create_envs(table_dims,box_size,pillar2_pose, env, pillar2_asset, i, gym, x = 0.4, y = -0.2)
        pillar2_idxs.append(pillar2_idx)

        pillar3_idx = add_pillar_in_create_envs(table_dims,box_size,pillar3_pose, env, pillar3_asset, i, gym, x = 0.65, y = 0.2)
        pillar3_idxs.append(pillar3_idx)

        pillar4_idx = add_pillar_in_create_envs(table_dims,box_size,pillar4_pose, env, pillar4_asset, i, gym, x = 0.4 , y = 0.2)
        pillar4_idxs.append(pillar4_idx)
        '''
        pillar1_idx = add_pillar_in_create_envs(table_dims,box_size,pillar1_pose, env, pillar1_asset, i, gym, x=0.328, y =0)
        pillar1_idxs.append(pillar1_idx)

        pillar2_idx = add_pillar_in_create_envs(table_dims,box_size,pillar2_pose, env, pillar2_asset, i, gym, x = 0.43, y = 0)
        pillar2_idxs.append(pillar2_idx)

        pillar3_idx = add_pillar_in_create_envs(table_dims,box_size,pillar3_pose, env, pillar3_asset, i, gym, x = 0.56, y = 0)
        pillar3_idxs.append(pillar3_idx)

        pillar4_idx = add_pillar_in_create_envs(table_dims,box_size,pillar4_pose, env, pillar4_asset, i, gym, x = 0.671, y = 0)
        pillar4_idxs.append(pillar4_idx)

        # add box
        # box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
        box_pose.p.x = 0.5 # 0.3
        box_pose.p.y = -0.2 #+ np.random.uniform(-0.2, 0.2) # -0.2
        box_pose.p.z = table_dims.z + 0.5 * box_size
        
        box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
        box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # get global index of box in rigid body state tensor
        box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
        box_idxs.append(box_idx)

        # add destination
        destination_pose.p.x = 0.5
        destination_pose.p.y = 0.2
        destination_pose.p.z = table_dims.z + 0.5 * box_size
        destination_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
        
        destination_handle = gym.create_actor(env, destination_asset, destination_pose, "box", i, 0)
        color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
        gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        # get global index of destination in rigid body state tensor
        destination_idx = gym.get_actor_rigid_body_index(env, destination_handle, 0, gymapi.DOMAIN_SIM)
        destination_idxs.append(destination_idx)


        # add ur3e
        ur3e_handle = gym.create_actor(env, ur3e_asset, ur3e_pose, "ur3e", i, 2)

        # set dof properties
        gym.set_actor_dof_properties(env, ur3e_handle, ur3e_dof_props)

        # set initial dof states
        gym.set_actor_dof_states(env, ur3e_handle, default_dof_state, gymapi.STATE_ALL)

        # set initial position targets
        gym.set_actor_dof_position_targets(env, ur3e_handle, default_dof_pos)

        # get initial hand pose
        hand_handle = gym.find_actor_rigid_body_handle(env, ur3e_handle, "hand_e_link")
        hand_pose = gym.get_rigid_transform(env, hand_handle)

        init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
        init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])
        

        # get global index of hand in rigid body state tensor
        hand_idx = gym.find_actor_rigid_body_index(env, ur3e_handle, "hand_e_link", gymapi.DOMAIN_SIM)
        hand_idxs.append(hand_idx)
    
    return envs, box_idxs,pillar1_idxs,pillar2_idxs,pillar3_idxs,pillar4_idxs, hand_idxs, init_pos_list, init_rot_list,destination_idxs