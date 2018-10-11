#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[SimConfig]
Sim_filename='Exp_01'
Sim_variables={'condition':[1,2],'pathology':[1,3]}
Sim_realizations={'Exp01':2}
Sim_name='E01'
Sim_path='./'
Sim_hostname='neptuno'
Sim_out_filename='out'
Sim_eout_filename='err'

Slurm_ntasks = 1
Slurm_tasks_per_node = 0
Slurm_cpus_per_task = 0
Slurm_nodes = 0
Slurm_email= 'jrestrepo@bioingenieria.edu.ar'
[end]

Created on Mon Jun 18 14:12:22 2018

@author: rcasal
"""

import os


from matplotlib import pyplot as plt

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as T
from torch.utils.data import DataLoader

from model import model_lstm
from model import model_gru, model_lstm

from SHHS_Dataset import SHHS_Dataset, collate_fn_RC, ToTensor
from train_model import train_model
from global_parameters import device, dtype_target, dtype_data


######################################################################
# main
# ----
#


# D_in is input dimension, H is hidden dimension; D_out is output dimension.
D_in, H, num_layers, batch_size, D_out = 2, 160, 2, 2,  2



# Database
datasets = {x: SHHS_Dataset(dbpath=os.path.abspath(os.path.join('../db/All_shhsCell_N/patient/', x)),
                                 transform=T.Compose([ToTensor()]))
           for x in ['train', 'val']}


params = {'batch_size': batch_size, 'num_workers': 1, 'pin_memory': False,
          'drop_last': True, 'collate_fn': collate_fn_RC}

dataloaders = {x: DataLoader(datasets[x], **params)
               for x in ['train', 'val']}


# Load model and set criterion
#model_ft = model_lstm(D_in, H, num_layers, batch_size, D_out).to(device)
model_ft = model_gru(D_in, H, num_layers, batch_size, D_out).to(device)

learning_rate = 1e-4
optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate, betas=(0.9, 0.99))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


######################################################################
# Train and evaluate
# ------------------
#
model_ft = train_model(model_ft, optimizer_ft, dataloaders, exp_lr_scheduler, num_epochs=50)


















# num_epochs = 20
# for epoch in range(num_epochs):
#
#     #since = time.time()
#
#     print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#     print('-' * 10)
#
#     # sample = training_set[1]
#     for i_batch, sample in enumerate(training_dataloader):
#
#
#         # sample to GPU
#         sample['feat'] = sample['feat'].to(device)
#         sample['target'] = sample['target'].to(device)
#         # sample['lengths'] = sample['lengths'].to(device)
#
#         # Forward pass
#         outputs = model(sample['feat'], sample['lengths'])
#
#         # Compute and print loss.
#         loss = model.ce_loss(outputs, sample['target'].to(device))
#
#         print(i_batch, loss.item())
#
#         # Before the backward pass, use the optimizer object to zero all of the
#         # gradients for the variables it will update (which are the learnable
#         # weights of the model). This is because by default, gradients are
#         # accumulated in buffers( i.e, not overwritten) whenever .backward()
#         # is called. Checkout docs of torch.autograd.backward for more details.
#         optimizer.zero_grad()
#
#         # Backward pass: compute gradient of the loss with respect to model
#         # parameters
#         loss.backward()
#
#         # Calling the step function on an Optimizer makes an update to its
#         # parameters
#         optimizer.step()
#
#
# #time_elapsed = time.time() - since
# print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))