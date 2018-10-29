######################################################################
# Model CustomLayer
# -----------------
#
# Configure model class
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class model_lstm(nn.Module):
    def __init__(self, D_in, hidden_size, num_layers, batch_size, D_out):

        super(model_lstm, self).__init__()

        self.D_in = D_in
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.D_out = D_out


        self.lstm = torch.nn.LSTM(input_size=self.D_in,
                                 hidden_size=self.hidden_size,
                                 num_layers=self.num_layers,
                                 bidirectional=True,
                                 batch_first=True)
        #self.lstm = nn.DataParallel(self.lstm, device_ids=[0, 1])
        #torch.nn.init.xavier_normal_(self.lstm.all_weights)



        self.linear = torch.nn.Linear(hidden_size*2, D_out)#.type(dst_type=self.dtype_data) # 2 for bidirection

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
        x = F.relu(self.linear(x))


        # 3. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, hidden_size) -> (batch_size, seq_len, D_out)
        x = F.log_softmax(x, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, D_out)
        x = x.view(batch_size, seq_len, self.D_out)

        y_hat = x #.type(dtype=dtype_target)
        return y_hat



class model_gru(nn.Module):
    def __init__(self, D_in, hidden_size, num_layers, batch_size, D_out):

        super(model_gru, self).__init__()

        self.D_in = D_in
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.D_out = D_out



        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(input_size=self.D_in,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=True,
                                  batch_first=True)


        self.linear = torch.nn.Linear(hidden_size*2, D_out) #.type(dst_type=self.dtype_data) # 2 for bidirection

    def init_hidden(self):
        """ the weights are of the form (num_layers, batch_size, nb_gru_units)
        Set initial hidden and cell states
        ¿INICIO CON ZEROS O CON RANDN? VI LOS DOS CASOS."""
        h0 = torch.autograd.Variable(next(self.parameters()).data.new(self.num_layers*2, self.batch_size,  self.hidden_size), requires_grad=False)

        return h0.zero_()


    def forward(self, x, x_lengths):
        """ reset the GRU hidden state. Must be done before you run a new patient. Otherwise the GRU will treat
        # a new batch as a continuation of a sequence """

        self.hidden = self.init_hidden()

        batch_size, seq_len, _ = x.size()

        # 1. Run through RNN
        # Dim transformation: (batch_size, seq_len, D_in) -> (batch_size, seq_len, hidden_size)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the GRU
        x = nn.utils.rnn.pack_padded_sequence (x, x_lengths, batch_first=True)

        # now run through GRU
        x, self.hidden = self.gru(x, self.hidden)

        # undo the packing operation
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # 2. Project to target space
        # Dim transformation: (batch_size, seq_len, hidden_size) -> (batch_size * seq_len, hidden_size)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        x = x.contiguous()                  # This doesn't affect the tensor at all, just make sure that it is stored in a contiguous chunk of memory.
        x = x.view(-1, x.shape[2])

        # run through actual linear layer
        x = F.relu(self.linear(x))


        # 3. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, hidden_size) -> (batch_size, seq_len, D_out)
        x = F.log_softmax(x, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, D_out)
        x = x.view(batch_size, seq_len, self.D_out)

        y_hat = x #.type(dtype=dtype_target)
        return y_hat





def ce_loss(y_hat, y):
    """ before we calculate the negative log likelihood, we need to mask out the activations
    this means we don't want to take into account padded items in the output vector
    simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
    and calculate the loss on that. """

    # flatten all the labels
    y= y.view(-1)

    _, _, D_out = y_hat.shape
    # flatten all predictions
    y_hat = y_hat.view(-1, D_out)

    # create a mask by filtering out all tokens that ARE NOT the padding token
    tag_pad_token = -1
    mask = (y > tag_pad_token).float()#.type(dtype=dtype_target)

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

    # accuracy1 = 0.0
    accuracy = 0.0
    sensibility = 0.0
    specificity = 0.0
    precision = 0.0
    npv = 0.0
    for i in range(y.size()[0]):
        #accuracy1 += torch.sum(preds[i, 0:y_lengths[i]] == y.data[i, 0:y_lengths[i]]).float() / y_lengths[i]

        #confusion matrix: 1 and 1 (TP); 1 and 0 (FP); 0 and 0 (TN); 0 and 1 (FN)
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

        accuracy += (tp + tn) / (tp + tn + fp + fn)
        sensibility += tp / (tp + fn)
        specificity += tn / (tn + fp)
        precision += tp / (tp + fp)
        npv += tn / (tn + fn)

    accuracy = accuracy / len(y_lengths)
    sensibility = sensibility / len(y_lengths)
    specificity = specificity / len(y_lengths)
    precision = precision / len(y_lengths)
    npv = npv / len(y_lengths)

    del tp, fp, tn, fn, aux, y_hat, y, y_lengths

    return accuracy, sensibility, specificity, precision, npv