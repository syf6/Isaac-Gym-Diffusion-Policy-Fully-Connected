import numpy as np 
import torch

from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["font.size"] = 18
import random
import os
import math

from torch import nn

from Train import Train
    
def make_dataloader(state, action, divisions, batch_size=64):
    data = np.hstack((state, action))
    np.random.shuffle(data)
    state = data[:,:1]
    action = data[:,1:]
    print(f"{state.shape =}")
    print(f"{action.shape =}")
    train_state = torch.tensor(state[:int(state.shape[0]*divisions)], dtype = torch.float32)
    #test_state = torch.tensor(state[int(state.shape[0]*divisions):], dtype = torch.float32)
    test_state = torch.tensor(np.random.uniform(0, 2*np.pi, 1000), dtype = torch.float32).reshape(-1,1)
    print(f"{train_state.shape = }")
    print(f"{test_state.shape = }")
    train_action = torch.tensor(action[:int(action.shape[0]*divisions)], dtype = torch.float32)
    #test_action = torch.tensor(action[int(action.shape[0]*divisions)::], dtype = torch.float32) 
    test_action = torch.tensor(np.array([math.sin(x)  for x in test_state]), dtype = torch.float32).reshape(-1,1)
    print(f"{train_action.shape = }")
    print(f"{test_action.shape = }")
    
    train_dataset = torch.utils.data.TensorDataset(train_state, train_action)
    val_dataset = torch.utils.data.TensorDataset(test_state, test_action)  
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}
    return dataloaders_dict


if __name__ =='__main__':

    divisions = 0.8
    dim = 1

    IDMtype = "test"

    state = np.random.uniform(0, 2*np.pi, 10000)
    #import ipdb;ipdb.set_trace()

    action = np.array([math.sin(x)  for x in state[:int(state.shape[0])]])
    # action_cos = np.array([math.cos(x)  for x in state[int(state.shape[0]/2):]])
    # action = np.concatenate([action_sin, action_cos], axis = 0)

    state = state.reshape(-1,1)
    action = action.reshape(-1,1)

    batch_size = 256
    lr = 1e-3
    dataloaders_dict = make_dataloader(state, action, divisions, batch_size)
    vit_idm_train = Train(IDMtype, f"test", "img/", dataloaders_dict, dim, 1000, lr, 42)
    #data_loader.save_study_data(f"IDM/{IDMtype}_euler_256")
    vit_idm_train.train_model()