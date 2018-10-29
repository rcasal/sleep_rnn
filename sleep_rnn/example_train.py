#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
[SimConfig]
Sim_filename = 'Exp_01'
Sim_variables = {'gru': [0, 1], 'h': [64, 128, 256]}
Sim_realizations = {'Exp01': 1}
Sim_name = 'E01'
Sim_path = './'
Sim_hostname = 'neptuno'
Sim_out_filename = 'out'
Sim_eout_filename = 'err'

Slurm_ntasks = 1
Slurm_tasks_per_node = 0
Slurm_cpus_per_task = 0
Slurm_nodes = 0
Slurm_email = 'jrestrepo@bioingenieria.edu.ar'
[end]
'''
import os


import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as T
from torch.utils.data import DataLoader

from model import model_gru, model_lstm

from SHHS_Dataset import SHHS_Dataset, collate_fn_RC, ToTensor
from train_model import train_model

# from matplotlib import pyplot as plt



######################################################################
# global parameters
# ----
#

cuda = torch.device('cuda:0')
torch.cuda.set_device(0)
print('Exp is running in {} No.{}'.format(
    torch.cuda.get_device_name(torch.cuda.current_device()), torch.cuda.current_device()))




######################################################################
# main
# ----
#
gru = 1
h = 1
Exp01 = 1

if gru==1:
    GRU = True
elif gru==0:
    GRU = False


# D_in is input dimension, H is hidden dimension; D_out is output dimension.
D_in, H, num_layers, batch_size, D_out = 1, h, 2, 2,  2
# GRU = True


# Database
datasets = {x: SHHS_Dataset(dbpath=os.path.abspath(os.path.join('../../db/All_shhsCell_SN/patient/', x)),
                                 transform=T.Compose([ToTensor()]), loadSaO2=False)
           for x in ['train', 'val']}


params = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': True,
          'drop_last': True, 'collate_fn': collate_fn_RC}

dataloaders = {x: DataLoader(datasets[x], **params)
               for x in ['train', 'val']}


# Load model and set criterion
if GRU:
    model_ft = model_gru(D_in, H, num_layers, batch_size, D_out).to(device=cuda)
else:
    model_ft = model_lstm(D_in, H, num_layers, batch_size, D_out).to(device=cuda)

learning_rate = 1e-4
optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate, betas=(0.9, 0.99))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# Prints
print('Parameters:')
print('\tGRU: {}'.format(GRU))
print('\tD_in: {} \n\tH: {} \n\tnum_layers: {} \n\tbatch_size_ {} \n\tD_out: {}'.format(
        D_in, H, num_layers, batch_size, D_out))
print('\n')

######################################################################
# Train and evaluate
# ------------------
#
model_ft = train_model(model_ft, optimizer_ft, dataloaders, exp_lr_scheduler, num_epochs=100, cuda=cuda)


torch.save(model_ft.state_dict(), './models/model_ft.pth')


