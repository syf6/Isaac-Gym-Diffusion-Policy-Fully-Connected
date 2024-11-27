# Inference part of conditional ddpm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import gymnasium as gym
import csv
import time
from gym.wrappers import RecordVideo

from tqdm import tqdm 

import pickle as pkl
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


if __name__=='__main__':
    with open('./data/fixed_policy_data_2.pkl', 'rb') as f:
        policy_ddpm = pkl.load(f)

    # 为环境添加视频录制功能
    env = gym.make('MountainCarContinuous-v0', render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, './videos', episode_trigger=lambda e: True)  # 将视频保存到 './videos' 文件夹中

    data_state = []
    data_action = []

    for e in range(5):
        states = []
        actions = []
    
        observation, info = env.reset()
        
        for i in range(1000):
            # 动作采样
            action = policy_ddpm.sampling(torch.tensor(observation).float(), n=1)
            action = action[0].numpy()

            # 应用动作并获取新的状态和奖励
            observation, reward, terminated, truncated, info = env.step(action)

            states.append(observation)
            actions.append(action)
        
            if terminated or truncated:
                break

        # 保存每个 episode 的状态和动作数据
        data_state.append(np.array(states))
        data_action.append(np.array(actions))
    
    env.close()

    # 保存数据
    with open('./data/recorded_states.pkl', 'wb') as f:
        pkl.dump(data_state, f)
    with open('./data/recorded_actions.pkl', 'wb') as f:
        pkl.dump(data_action, f)