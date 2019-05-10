######################################################################
# Model CustomLayer
# -----------------
#
# Configure model class
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score

class ModelLstm(nn.Module):
    def __init__(self, d_in, hidden_size, num_layers, batch_size, d_out):

        super(ModelLstm, self).__init__()

        self.d_in = d_in
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.d_out = d_out

        self.lstm = torch.nn.LSTM(input_size=self.d_in,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=True,
                                  batch_first=True)

        # self.lstm = nn.DataParallel(self.lstm, device_ids=[0, 1])
        # torch.nn.init.xavier_normal_(self.lstm.all_weights)

        self.linear = torch.nn.Linear(hidden_size*2, d_out)  # .type(dst_type=self.dtype_data) # 2 for bidirection

    def init_hidden(self):
        """ the weights are of the form (num_layers, batch_size, nb_lstm_units)
        Set initial hidden and cell states
        ¿INICIO CON ZEROS O CON RANDN? VI LOS DOS CASOS."""

        h0 = torch.autograd.Variable(
            next(self.parameters()).data.new(self.num_layers * 2, self.batch_size, self.hidden_size),
            requires_grad=False)
        c0 = torch.autograd.Variable(
            next(self.parameters()).data.new(self.num_layers * 2, self.batch_size, self.hidden_size),
            requires_grad=False)

        return h0.zero_(), c0.zero_()

    def forward(self, x, x_lengths):
        """ reset the LSTM hidden state. Must be done before you run a new patient. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence """

        self.hidden = self.init_hidden()

        batch_size, seq_len, _ = x.size()

        # 1. Run through RNN
        # Dim transformation: (batch_size, seq_len, d_in) -> (batch_size, seq_len, hidden_size)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        x = nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)

        # now run through LSTM
        x, self.hidden = self.lstm(x, self.hidden)

        # undo the packing operation
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # 2. Project to target space
        # Dim transformation: (batch_size, seq_len, hidden_size) -> (batch_size * seq_len, hidden_size)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        # This doesn't affect the tensor at all, just make sure that it is stored in a contiguous chunk of memory.
        x = x.contiguous()
        x = x.view(-1, x.shape[2])

        # run through actual linear layer
        x = F.relu(self.linear(x))

        # 3. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, hidden_size) -> (batch_size, seq_len, d_out)
        x = F.log_softmax(x, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, d_out)
        x = x.view(batch_size, seq_len, self.d_out)

        y_hat = x  # .type(dtype=dtype_target)
        return y_hat


class ModelGru(nn.Module):
    def __init__(self, d_in, hidden_size, num_layers, batch_size, d_out):

        super(ModelGru, self).__init__()

        self.d_in = d_in
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.d_out = d_out

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(input_size=self.d_in,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          bidirectional=True,
                          batch_first=True)

        self.linear = torch.nn.Linear(hidden_size*2, d_out)  # .type(dst_type=self.dtype_data) # 2 for bidirection

    def init_hidden(self):
        """ the weights are of the form (num_layers, batch_size, nb_gru_units)
        Set initial hidden and cell states
        ¿INICIO CON ZEROS O CON RANDN? VI LOS DOS CASOS."""
        h0 = torch.autograd.Variable(next(self.parameters()).data.new(
            self.num_layers*2, self.batch_size, self.hidden_size), requires_grad=False)

        return h0.zero_()

    def forward(self, x, x_lengths=None):
        """ reset the GRU hidden state. Must be done before you run a new patient. Otherwise the GRU will treat
        # a new batch as a continuation of a sequence """

        self.hidden = self.init_hidden()

        batch_size, seq_len, _ = x.size()

        # 1. Run through RNN
        # Dim transformation: (batch_size, seq_len, d_in) -> (batch_size, seq_len, hidden_size)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the GRU
        if x_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, x_lengths, batch_first=True)

        # now run through GRU
        x, self.hidden = self.gru(x, self.hidden)

        # undo the packing operation
        if x_lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # 2. Project to target space
        # Dim transformation: (batch_size, seq_len, hidden_size) -> (batch_size * seq_len, hidden_size)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        x = x.contiguous()
        x = x.view(-1, x.shape[2])

        # run through actual linear layer
        x = F.relu(self.linear(x))

        # 3. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, hidden_size) -> (batch_size, seq_len, d_out)
        x = F.log_softmax(x, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, d_out)
        x = x.view(batch_size, seq_len, self.d_out)

        y_hat = x  # .type(dtype=dtype_target)
        return y_hat


def ce_loss(y_hat, y):
    """ before we calculate the negative log likelihood, we need to mask out the activations
    this means we don't want to take into account padded items in the output vector
    simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
    and calculate the loss on that. """

    # flatten all the labels
    y = y.view(-1)

    _, _, d_out = y_hat.shape
    # flatten all predictions
    y_hat = y_hat.view(-1, d_out)

    # create a mask by filtering out all tokens that ARE NOT the padding token
    tag_pad_token = -1
    mask = (y > tag_pad_token).float()  # .type(dtype=dtype_target)

    # count how many tokens we have
    nb_tokens = int(torch.sum(mask).item())

    # pick the values for the label and zero out the rest with the mask
    y_hat = y_hat[range(y_hat.shape[0]), y] * mask

    # compute cross entropy loss which ignores all <PAD> tokens
    ce_loss = -torch.sum(y_hat) / nb_tokens

    # del variables
    del y_hat, y, tag_pad_token, nb_tokens, mask

    return ce_loss


def statistics(y_hat, y, y_lengths):

    _, preds = torch.max(y_hat, 2)

    accuracy = 0.0
    sensibility = 0.0
    specificity = 0.0
    precision = 0.0
    npv = 0.0
    for i in range(y.size()[0]):
        # accuracy1 += torch.sum(preds[i, 0:y_lengths[i]] == y.data[i, 0:y_lengths[i]]).float() / y_lengths[i]

        # confusion matrix: 1 and 1 (TP); 1 and 0 (FP); 0 and 0 (TN); 0 and 1 (FN)
        aux = preds[i, 0:y_lengths[i]].float() / y.data[i, 0:y_lengths[i]].float()

        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)

        tp = torch.sum(aux == 1.0).item() + 1e-8
        fp = torch.sum(aux == float('inf')).item() + 1e-8
        tn = torch.sum(torch.isnan(aux)).item() + 1e-8
        fn = torch.sum(aux == 0).item() + 1e-8

        # accuracy += (tp + tn) / (tp + tn + fp + fn)
        sensibility += tp / (tp + fn)
        specificity += tn / (tn + fp)
        precision += tp / (tp + fp)
        npv += tn / (tn + fn)

        # Otra forma de calcular accuracy util para multiclass
        coinc = (preds[i, 0:y_lengths[i]].float() == y.data[i, 0:y_lengths[i]].float()).sum()
        accuracy += coinc.item() / preds[i, 0:y_lengths[i]].__len__()

        del tp, fp, tn, fn, aux

    accuracy = accuracy / len(y_lengths)
    sensibility = sensibility / len(y_lengths)
    specificity = specificity / len(y_lengths)
    precision = precision / len(y_lengths)
    npv = npv / len(y_lengths)

    del y_hat, y, y_lengths

    return accuracy, sensibility, specificity, precision, npv



def statistics_test(y_hat, y, y_lengths):

    _, preds = torch.max(y_hat, 2)

    stats = {}
    stats['acc'] = [0.0, 0.0]
    stats['se'] = [0.0, 0.0]
    stats['sp'] = [0.0, 0.0]
    stats['pre'] = [0.0, 0.0]
    stats['npv'] = [0.0, 0.0]
    stats['kappa'] = [0.0, 0.0]
    stats['err'] = [0.0, 0.0]
    stats['rel_err'] = [0.0, 0.0]
    stats['tst'] = [0.0, 0.0]
    stats['tst_est'] = [0.0, 0.0]
    stats['trt'] = [0.0, 0.0]

    for j in range(2):
        for i in range(y.size()[0]):

            y_i = y[i, 0:y_lengths[i]]
            preds_i = preds[i, 0:y_lengths[i]]
            if j is 1:
                y_i, _ = torch.median(y_i.view(-1, 30), dim=1)
                preds_i, _ = torch.median(preds_i.view(-1, 30), dim=1)
                # preds[preds == 0.5] = 1  # Si empatan les asigno dormido (podría asignarle el valor ant)

            # confusion matrix: 1 and 1 (TP); 1 and 0 (FP); 0 and 0 (TN); 0 and 1 (FN)
            aux = preds_i.float() / y_i.data.float()

            tp = torch.sum(aux == 1.0).item() + 1e-8
            fp = torch.sum(aux == float('inf')).item() + 1e-8
            tn = torch.sum(torch.isnan(aux)).item() + 1e-8
            fn = torch.sum(aux == 0).item() + 1e-8

            stats['acc'][j] += (tp + tn) / (tp + tn + fp + fn)
            stats['se'][j] += tp / (tp + fn)
            stats['sp'][j] += tn / (tn + fp)
            stats['pre'][j] += tp / (tp + fp)
            stats['npv'][j] += tn / (tn + fn)
            a_rnd = ((tp + fp)*(tp + fn) + (fp + tn)*(fn + tn))/(tp + tn + fp + fn)**2
            stats['kappa'][j] += (stats['acc'][j] - a_rnd)/(1 - a_rnd)

            if j is 0:
                tst = torch.sum(y_i == 1).item() / (60 * 60)
                tst_est = torch.sum(preds_i == 1).item() / (60 * 60)
                trt = len(y_i)  / (60 * 60)
            else:
                tst = torch.sum(y_i == 1).item() / (2 * 60)
                tst_est = torch.sum(preds_i == 1).item() / (2 * 60)
                trt = len(y_i)/ (2 * 60)

            stats['err'][j] += abs(tst-tst_est)
            stats['rel_err'][j] += abs(tst-tst_est)/tst
            stats['tst'][j] += tst
            stats['tst_est'][j] += tst_est
            stats['trt'][j] += trt

            # Plots
            # if j is 0:
            #     plt.figure()
            #     t = np.arange(0, y_lengths[i], 1) / (60 * 60)
            # else:
            #     t = np.arange(0, y_lengths[i], 30) / (60 * 60)
            #
            # plt.subplot(2, 1, j+1)
            # plt.plot(t, y_i.numpy(), 'k', t, preds_i.numpy(), 'r--')
            # plt.xlabel('Time [hs]')
            # plt.title(stats['err'][j]*60)

            # Plots para franceses:
            # f = plt.figure()
            # plt.plot(t, y_i.numpy(), 'k', t, preds_i.numpy(), 'r--')
            # plt.xlabel('Time [hs]')
            # plt.yticks(np.arange(2), ('W', 'S'))
            # plt.show()
            # # f.savefig("Subj1.pdf", bbox_inches='tight')


        stats['acc'][j] = stats['acc'][j] / len(y_lengths)
        stats['se'][j] = stats['se'][j] / len(y_lengths)
        stats['sp'][j] = stats['sp'][j] / len(y_lengths)
        stats['pre'][j] = stats['pre'][j] / len(y_lengths)
        stats['npv'][j] = stats['npv'][j] / len(y_lengths)
        stats['kappa'][j] = stats['kappa'][j] / len(y_lengths)
        stats['err'][j] = stats['err'][j] / len(y_lengths)
        stats['rel_err'][j] = stats['rel_err'][j] / len(y_lengths)
        stats['tst'][j] = stats['tst'][j]/ len(y_lengths)
        stats['tst_est'][j] = stats['tst_est'][j]/ len(y_lengths)
        stats['trt'][j] = stats['trt'][j]/ len(y_lengths)

    # plt.show()
    return stats



def statistics_test_multiclass(y_hat, y, y_lengths, n_class):
    "Sólo funciona para n_batch 1!!!"

    if len(y_lengths) is 1:

        _, preds = torch.max(y_hat, 2)

        stats = {}
        stats['acc'] = 0.0
        stats['kappa'] = 0.0
        stats['confusion_matrix'] = np.zeros([n_class,n_class])
        stats['confusion_matrix_percentage'] = np.zeros([n_class,n_class])

        y_i = y[0, 0:y_lengths[0]]
        preds_i = preds[0, 0:y_lengths[0]]

        y_i, _ = torch.median(y_i.view(-1, 30), dim=1)
        preds_i, _ = torch.median(preds_i.view(-1, 30), dim=1)

        coinc = (preds_i.float() == y_i.data.float()).sum()
        stats['acc'] = coinc.item()/preds_i.__len__()

        stats['confusion_matrix'] = confusion_matrix(preds_i, y_i, [0, 1, 2, 3, 4, 5])

        # stats['confusion_matrix_percentage'] = stats['confusion_matrix'] / stats['confusion_matrix'].astype(np.float).sum(axis=0)

        stats['kappa']  = cohen_kappa_score(preds_i, y_i, [0, 1, 2, 3, 4, 5])

        # print(stats['confusion_matrix'])
        # # plots
        # plt.subplot(2,1,1)
        # plt.plot(y_i.numpy(), 'k')
        # tit = 'Acc:' + str(stats['acc'])
        # plt.title(tit)
        # plt.subplot(2,1,2)
        # plt.plot(preds_i.numpy(), 'k')
        # plt.xlabel('Time [hs]')
        # plt.show()
    else:
        print('Error. Minibatch size must be 1!')
        exit()
    return stats
