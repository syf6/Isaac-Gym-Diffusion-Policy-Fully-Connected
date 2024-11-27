import numpy as np 
import torch
import torch.nn as nn
import torch.functional as F

from torch.utils.data import Dataset
from torchvision import datasets

from ModelStructure import StateNet
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams["font.size"] = 18
import random
import os
from tqdm import tqdm
from IPython import display
import math 


np.set_printoptions(suppress=True)

class WeightedL1Loss(nn.Module): 
    def __init__(self): 
        super(WeightedL1Loss, self).__init__() 

    def forward(self, input, target, weight): 
        loss = weight * torch.abs(input - target)
        return loss.sum() / weight.sum()

class Train:
    def __init__(self, model_type, model_name, learnig_curve_file, dataloaders_dict, dim, num_epochs, lr, seed=42):
        self.model_name = model_name
        self.learning_curve_file = learnig_curve_file
        self.dataloaders_dict = dataloaders_dict
        self.num_epochs = num_epochs
        self.lr = lr
        self.dim = dim
        # self.seed_everything(seed)
        
        if model_type == "fcmIDM":
            state_space_size = 3*1*self.dim + 3*1*self.dim + 3*1*self.dim + 3*1*self.dim
            action_space_size = 3 + 3       
            self.model = StateNet(state_space_size, action_space_size)         

        elif model_type == "fcmpolicy":
            state_space_size = 3*1*self.dim + 3*1*self.dim + 3*1*self.dim + 3*1*self.dim + 3*1*self.dim + 3*1*self.dim 
            action_space_size = 3 + 3       
            self.model = StateNet(state_space_size, action_space_size)      

        elif model_type == "test":
            state_space_size = 1
            action_space_size = 1      
            self.model = StateNet(state_space_size, action_space_size)    

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("使用デバイス:", self.device)
        self.model.to(self.device)


    def train_model(self):
        loss_list = []
        test_list = []
        best_loss = np.inf
        train_trans_loss = []
        train_rot_loss = []
        test_trans_loss = []
        test_rot_loss = []

        # loss function
        criterion = nn.L1Loss()
        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay=0.000025) 

        torch.backends.cudnn.benchmark = True

        for epoch in tqdm(range(self.num_epochs), leave = False):   
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()           
                epoch_loss = 0.0
                trans_loss = 0.0
                rot_loss = 0.0

                for inputs, labels in self.dataloaders_dict[phase]: 
                    inputs = inputs.to(self.device)     
                    labels = labels.to(self.device)  
                    optimizer.zero_grad()
                
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            #print(weights)

                        epoch_loss += loss.item() * inputs.size(0)
                        trans_loss += abs(outputs[:,:3] - labels[:,:3]).sum() / 3
                        rot_loss += abs(outputs[:,3:] - labels[:,3:]).sum() / 3

                epoch_loss = epoch_loss / len(self.dataloaders_dict[phase].dataset)
                trans_loss = trans_loss / len(self.dataloaders_dict[phase].dataset)
                rot_loss = rot_loss / len(self.dataloaders_dict[phase].dataset)
                if phase =='train':
                    loss_list.append(epoch_loss)
                    train_trans_loss.append(trans_loss.cpu().detach().numpy())
                    train_rot_loss.append(rot_loss.cpu().detach().numpy())
                elif phase =='val':
                    test_list.append(epoch_loss)
                    test_trans_loss.append(trans_loss.cpu().detach().numpy())
                    test_rot_loss.append(rot_loss.cpu().detach().numpy())
                    if epoch_loss < best_loss:
                        torch.save(self.model.state_dict(), f"./model/{self.model_name}_best.pt")
                        best_loss = epoch_loss            
                    
            display.clear_output(wait=True)    
            print(f'epoch: {epoch}  {phase} loss: {epoch_loss:.6f}')  

            if epoch % 100 == 0 and epoch != 0:
                plt.plot(test_list[:], label="Test loss")
                plt.plot(loss_list[:], label="Train loss")
                plt.plot(train_trans_loss[:], label="Train trans loss")
                plt.plot(test_trans_loss[:], label="Test trans loss")
                plt.plot(train_rot_loss[:], label="Train rot loss")
                plt.plot(test_rot_loss[:], label="Test rot loss")
                plt.xlabel("Number of epochs")
                plt.ylabel("Loss")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"./{self.learning_curve_file}/{epoch}_all.pdf")
                print(min(test_list))
                plt.clf()

                plt.plot(test_list[:], label="Test loss")
                plt.plot(loss_list[:], label="Train loss")
                plt.legend()
                plt.xlabel("Number of epochs")
                plt.ylabel("Loss")
                plt.tight_layout()
                plt.savefig(f"./{self.learning_curve_file}/{epoch}.pdf")
                plt.clf()  
        torch.save(self.model.state_dict(), f"./model/{self.model_name}_last.pt") 
        return loss_list, test_list
