import torch
from torch import nn
import numpy as np

class StateNet(torch.nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super().__init__()
        self.l1 = nn.Linear(state_space_size, 256)
        #self.l2 = nn.Linear(512, 1024)
        #self.l3 = nn.Linear(1024, 2046)
        #self.l4 = nn.Linear(2046, 1024)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, action_space_size)
        #self.dropout = nn.Dropout(p = 0.20)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        x4 = self.relu(x3)
        x_out = self.l3(x4)
        return x_out