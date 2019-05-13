"""Basic and Advanced Neural Network Modules."""
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from torch.nn import Module
import torch.nn as nn


class DetLinear(Module):
    """Deterministic Linear units.

    Almost a direct copy of torch.nn.Linear, except initialized to identity
    function.
    """

    def __init__(self, indim, bias=True):
        """Create a simple linear layer with dimension `indim`."""
        super(DetLinear, self).__init__()
        self.indim = indim
        self.weight = Parameter(torch.eye(indim))
        if bias:
            self.bias = Parameter(torch.zeros(indim))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        """One forward pass."""
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        """Vebose info."""
        return 'indim={}, bias={}'.format(
            self.indim, self.bias is not None
        )


class DetMLP(Module):
    """Detministic multi-Layer Perceptron."""

    def __init__(self, indim, nhidden, bias=True,
                 activate_hid=nn.ReLU(), activate_out=nn.ReLU()):
        """Initialize a deterministic MLP.

        Parameters
        ----------
        indim: int
            Input dimension to the MLP. All hidden layers and the output layer
            are constrained to have the same dimension.
        nhidden: int
            Number of hidden layers of the MLP. If less than 1, set to 1 hidden
            layer.
        bias: bool [True]
            Apply bias for this network?

        """
        super(DetMLP, self).__init__()
        self.indim = indim
        if nhidden < 1:
            print("At least 1 hidden layer should be created.")
            self.nhidden = 1  # at least 1 hidden layer
        else:
            self.nhidden = nhidden
        self.bias = bias
        self.activate_hid = activate_hid
        self.activate_out = activate_out
        self.linears = nn.ModuleList(
            [DetLinear(indim, bias=bias) for ii in range(1 + self.nhidden)]
        )

    def forward(self, x):
        """One forward pass."""
        for ii, layer in enumerate(self.linears):
            if ii == self.nhidden:
                break
            x = self.activate_hid(layer(x))
        return self.activate_out(self.linears[-1](x))

    def extra_repr(self):
        """Vebose info."""
        return 'indim={}, nhidden={}, bias={}'.format(
            self.indim, self.nhidden, self.bias is not None
        )


class MLP(Module):
    """Multi-Layer Perceptron."""

    def __init__(self, indim, outdim, hiddims=[], bias=True,
                 activate_hid=nn.ReLU(), activate_out=nn.ReLU(),
                 batchnorm=[]):
        """Initialize a MLP.

        Parameters
        ----------
        indim: int
            Input dimension to the MLP.
        outdim: int
            Output dimension to the MLP.
        hiddims: list of int
            A list of hidden dimensions. Default ([]) means no hidden layers.
        bias: bool [True]
            Apply bias for this network?
        activate_hid: callable, optional
            Activation function for hidden layers. Default to ReLU.
        activate_out: callable, optional
            Activation function for output layer. Default to ReLU.

        """
        super(MLP, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hiddims = hiddims
        self.nhidden = len(hiddims)
        self.bias = bias
        self.activate_hid = activate_hid
        self.activate_out = activate_out
        if self.nhidden == 0:
            print("No hidden layers.")
        indims = [indim] + hiddims
        outdims = hiddims + [outdim]
        self.layers = nn.ModuleList([])
        for ii in range(self.nhidden):
            self.layers.append(nn.Linear(indims[ii], outdims[ii], bias=bias))
            if len(batchnorm) > 0 and batchnorm[ii]:
                self.layers.append(nn.BatchNorm1d(outdims[ii], momentum=0.05))
            self.layers.append(activate_hid)
        self.layers.append(nn.Linear(indims[ii+1], outdims[ii+1], bias=bias))
        self.layers.append(activate_out)

    def forward(self, x):
        """One forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x


class AdvancedLSTMCell(nn.LSTMCell):
    """Extend LSTMCell to learn initial states."""

    def __init__(self, *args, **kwargs):
        """Initialize a LSTMCell.

        Parameters
        ----------
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If `False`, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

        Inputs
        ------
        input, (h_0, c_0)
        input of shape (batch, input_size): tensor containing input features
        h_0 of shape (batch, hidden_size): tensor containing the initial hidden
            state for each element in the batch.
        c_0 of shape (batch, hidden_size): tensor containing the initial cell
            state for each element in the batch.
        If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.

        Outputs
        -------
        h_1 of shape (batch, hidden_size): tensor containing the next hidden
            state for each element in the batch.
        c_1 of shape (batch, hidden_size): tensor containing the next cell
            state for each element in the batch.

        """
        super(AdvancedLSTMCell, self).__init__(*args, **kwargs)
        self.h0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())

    def initial_state(self, n):
        return (
            self.h0.expand(n, -1).contiguous(),
            self.c0.expand(n, -1).contiguous()
        )


class AdvancedLSTM(nn.LSTM):
    """Extend LSTM to learn initial states."""

    def __init__(self, *args, **kwargs):
        """Initialize a LSTM.

        Parameters
        ----------
        input_size – The number of expected features in the input x
        hidden_size – The number of features in the hidden state h
        num_layers – Number of recurrent layers. E.g., setting num_layers=2
            would mean stacking two LSTMs together to form a stacked LSTM,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias – If False, then the layer does not use bias weights b_ih and
            b_hh. Default: True
        batch_first – If True, then the input and output tensors are provided
            as (batch, seq, feature). Default: False
        dropout – If non-zero, introduces a Dropout layer on the outputs of
            each LSTM layer except the last layer, with dropout probability
            equal to dropout. Default: 0
        bidirectional – If True, becomes a bidirectional LSTM. Default: False

        Inputs
        ------
        input, (h_0, c_0)
        input of shape (seq_len, batch, input_size): tensor containing the
            features of the input sequence. The input can also be a packed
            variable length sequence.
        h_0 of shape (num_layers * num_directions, batch, hidden_size):
            tensor containing the initial hidden state for each element in the
            batch. If the RNN is bidirectional, num_directions should be 2,
            else it should be 1.
        c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor
            containing the initial cell state for each element in the batch.
        If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.

        Outputs
        -------
        output, (h_n, c_n)
        output of shape (seq_len, batch, num_directions * hidden_size): tensor
            containing the output features (h_t) from the last layer of the
            LSTM, for each t. If a torch.nn.utils.rnn.PackedSequence has been
            given as the input, the output will also be a packed sequence.
        For the unpacked case, the directions can be separated using output.
            view(seq_len, batch, num_directions, hidden_size), with forward
            and backward being direction 0 and 1 respectively. Similarly,
            the directions can be separated in the packed case.
        h_n of shape (num_layers * num_directions, batch, hidden_size):
            tensor containing the hidden state for t = seq_len.
        Like output, the layers can be separated using h_n.view(num_layers,
            num_directions, batch, hidden_size) and similarly for c_n.
        c_n (num_layers * num_directions, batch, hidden_size): tensor
            containing the cell state for t = seq_len

        """
        super(AdvancedLSTM, self).__init__(*args, **kwargs)
        bi = 2 if self.bidirectional else 1
        self.h0 = nn.Parameter(torch.FloatTensor(
            bi, 1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(
            bi, 1, self.hidden_size).zero_())

    def initial_state(self, n):
        return (
            self.h0.expand(-1, n, -1).contiguous(),
            self.c0.expand(-1, n, -1).contiguous()
        )

    def forward(self, input, hx=None):
        if hx is None:
            n = input.batch_sizes[0]
            hx = self.initial_state(n)
        return super(AdvancedLSTM, self).forward(input, hx=hx)


class SequenceShuffle(nn.Module):
    """
    Performs efficient pooling (using indexing) for pyramid LSTM without
    iteration.
    """

    @staticmethod
    def concate_sequence(sequence, length):
        """pyramid LSTM, merge consecutive time step together"""
        shape = sequence.size()

        # efficient using indexing, don't need iteration
        input_range = sequence.data.new(shape[0] // 2).zero_().long()
        torch.arange(1, int(shape[0]), 2, out=input_range)
        input_concate = torch.cat(
            (sequence[input_range - 1], sequence[input_range]), 2)

        length = np.array(length) // 2

        return input_concate, length

    def forward(self, seq):
        assert isinstance(seq, PackedSequence)
        h, seq_lengths = pad_packed_sequence(seq)
        h, seq_lengths = self.concate_sequence(h, seq_lengths)
        h = pack_padded_sequence(h, seq_lengths)
        return h


class PyramidalLSTM(AdvancedLSTM):
    """Pyramidal LSTM could reduce the sequence length by half."""

    def __init__(self, *args, **kwargs):
        """See AdvancedLSTM."""
        super(PyramidalLSTM, self).__init__(*args, **kwargs)
        self.shuffle = SequenceShuffle()

    def forward(self, input, hx=None):
        return super(PyramidalLSTM, self).forward(self.shuffle(input), hx=hx)


class WeightdropLSTM(nn.LSTM):
    """
    LSTM with dropout, apply dropout on the hidden-to-hidden weight matrix
    and the input.
    """

    @staticmethod
    def generate_mask(p, inp):
        """Generate mask for random drop of network weights.

        p is the drop rate, aka the probability of being 0
        modified from https://github.com/salesforce/awd-lstm-lm/blob/master/
        using .new() to initialize from the same device, much more efficient
        """
        return (inp.data.new(1, inp.size(1)).bernoulli_(1. - p) / (1. - p))\
            .expand_as(inp)

    def __init__(self, input_size, hidden_size, dropout_weight, dropout_inp):
        """Initialize a WeightdropLSTM.

        Parameters
        ----------
        input_size: int
            The number of expected features in the input x
        hidden_size: int
            The number of features in the hidden state h
        dropout_weight: float
            Introduces a Dropout layer on the hidden-to-hidden weight matrix
            of LSTM, with dropout probability equal to dropout.
        dropout_inp: float
            Introduces a Dropout layer on the input matrix of LSTM, with
            dropout probability equal to dropout.

        """
        super(WeightdropLSTM, self).__init__(input_size=input_size,
                                             hidden_size=hidden_size,
                                             bidirectional=True)
        self.old_weight_hh_l0 = self.weight_hh_l0
        self.weight_hh_l0 = None
        del self._parameters['weight_hh_l0']
        self.dropout_layer = nn.Dropout(dropout_weight)
        self.dropout_inp = dropout_inp

    def flatten_parameters(self):
        """
        Overwrite this method, which prevents PyTorch from putting all weight
        into a large chunk.
        """
        self._data_ptrs = []

    def forward(self, inp, hx=None):
        self.weight_hh_l0 = self.dropout_layer(self.old_weight_hh_l0)

        if self.training and self.dropout_inp != 0:
            input, batch_size = inp
            between_layer_mask = self.generate_mask(
                p=self.dropout_inp, inp=input)
            dropedinput = input * between_layer_mask
            inp = PackedSequence(dropedinput, batch_size)

        return super(WeightdropLSTM, self).forward(inp, hx=hx)
