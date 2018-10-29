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


# from matplotlib import pyplot as plt

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from model import model_gru, model_lstm

from SHHS_Dataset import SHHS_Dataset, collate_fn_RC, ToTensor
from eval_model import eval_model
from global_parameters import cuda, dtype_target, dtype_data


######################################################################
# main
# ----
#


# d_in is input dimension, H is hidden dimension; d_out is output dimension.
d_in, H, num_layers, batch_size, d_out = 1, 128, 2, 2,  2
GRU = True


# Database
datasets = {x: SHHS_Dataset(dbpath=os.path.abspath(os.path.join('../db/All_shhsCell_SN/patient/', x)),
                                 transform=T.Compose([ToTensor()]), loadSaO2=False)
           for x in ['train', 'val', 'test']}


params = {'batch_size': batch_size, 'num_workers': 0, 'pin_memory': True,
          'drop_last': True, 'collate_fn': collate_fn_RC}

dataloaders = {x: DataLoader(datasets[x], **params)
               for x in ['train', 'val', 'test']}


# Load model and set criterion
if GRU:
    model_ft = model_gru(d_in, H, num_layers, batch_size, d_out).to(device=cuda)
else:
    model_ft = model_lstm(d_in, H, num_layers, batch_size, d_out).to(device=cuda)

model_ft.load_state_dict(torch.load('../sleep_rnn/models/model_ft.pth'))

# Prints
print('Parameters:')
print('\tGRU: {}'.format(GRU))
print('\td_in: {} \n\tH: {} \n\tnum_layers: {} \n\tbatch_size_ {} \n\td_out: {}'.format(
        d_in, H, num_layers, batch_size, d_out))
print('\n')

######################################################################
# Train and evaluate
# ------------------
#
eval_model(model_ft, dataloaders)


torch.save(model_ft.state_dict(), './models/model_ft.pth')


