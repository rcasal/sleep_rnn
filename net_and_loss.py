
import os
import numpy as np
from scipy.io import savemat, loadmat
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader






#class TwoLayerNet(nn.Module):
#    def __init__(self, D_in, H, D_out):
#        """
#        In the constructor we instantiate two nn.Linear modules and assign them as
#        member variables.
#        """
#        super(TwoLayerNet, self).__init__()
#        self.linear1 = torch.nn.Linear(D_in, H)
#        self.linear2 = torch.nn.Linear(H, 2*H)
#        self.linear3 = torch.nn.Linear(2*H, D_out)
#
#    def forward(self, x):
#        """
#        In the forward function we accept a Tensor of input data and we must return
#        a Tensor of output data. We can use Modules defined in the constructor as
#        well as arbitrary operators on Tensors.
#        """
#        h_relu = self.linear1(x).clamp(min=0)
#        h_relu = self.linear2(h_relu)
#        y_pred = self.linear3(h_relu)
#        return F.softmax(y_pred, dim=1)



class CustomLayer(nn.Module):
    def __init__(self, D_in, hidden_size, num_layers, batch_size, D_out):

        super(CustomLayer, self).__init__()

        self.D_in = D_in
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.D_out = D_out

        self.lstm= torch.nn.LSTM(input_size=self.D_in,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 bidirectional=False,
                                 batch_first=True)

        self.linear = torch.nn.Linear(hidden_size, D_out)


    def init_hidden(self):
        """ the weights are of the form (num_layers, batch_size, nb_lstm_units)
        Set initial hidden and cell states
        ¿INICIO CON ZEROS O CON RANDN? VI LOS DOS CASOS."""
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

        h0 = torch.autograd.Variable(h0, requires_grad=True)
        c0 = torch.autograd.Variable(c0, requires_grad=True)

        return (h0, c0)


    def forward(self, x, x_lengths):
        """ reset the LSTM hidden state. Must be done before you run a new patient. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence """
        self.hidden = self.init_hidden()

        batch_size, seq_len, _ = x.size()

        # 1. Run through RNN
        # Dim transformation: (batch_size, seq_len, D_in) -> (batch_size, seq_len, hidden_size)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        x = nn.utils.rnn.pack_padded_sequence (x, x_lengths, batch_first=True)

        # now run through LSTM
        x, self.hidden = self.lstm(x, self.hidden)

        # undo the packing operation
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # 2. Project to target space
        # Dim transformation: (batch_size, seq_len, hidden_size) -> (batch_size * seq_len, hidden_size)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        x = x.contiguous()                  # This doesn't affect the tensor at all, just make sure that it is stored in a contiguous chunk of memory.
        x = x.view(-1, x.shape[2])

        # run through actual linear layer
        x = self.linear(x)


        # 3. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, hidden_size) -> (batch_size, seq_len, D_out)
        x = F.log_softmax(x, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, D_out)
        x = x.view(batch_size, seq_len, self.D_out)

        y_hat = x
        return y_hat

    def ce_loss(self, y_hat, y, x_lengths=[]):
        """ before we calculate the negative log likelihood, we need to mask out the activations
        this means we don't want to take into account padded items in the output vector
        simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        and calculate the loss on that. """

        # flatten all the labels
        y= y.view(-1)

        # flatten all predictions
        y_hat = y_hat.view(-1, self.D_out)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = -1
        mask = (y > tag_pad_token).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).item())

        # pick the values for the label and zero out the rest with the mask
        y_hat = y_hat[range(y_hat.shape[0]), y] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(y_hat) / nb_tokens

        return ce_loss