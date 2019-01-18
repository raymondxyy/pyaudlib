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

    def __init__(self, indim, nhidden, bias=True):
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
        self.relu = nn.ReLU()
        self.linears = nn.ModuleList(
            [DetLinear(indim, bias=bias) for ii in range(1+self.nhidden)]
        )

    def forward(self, x):
        """One forward pass."""
        for layer in self.linears:
            x = self.relu(layer(x))
        return x

    def extra_repr(self):
        """Vebose info."""
        return 'indim={}, nhidden={}, bias={}'.format(
            self.indim, self.nhidden, self.bias is not None
        )


class MLP(Module):
    """Multi-Layer Perceptron."""

    def __init__(self, indim, outdim, hiddims=[], bias=True,
                 activate_hid=nn.ReLU(), activate_out=nn.ReLU()):
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
        self.linears = nn.ModuleList(
            [nn.Linear(indims[ii], outdims[ii], bias=bias)
             for ii in range(1+self.nhidden)]
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
        return 'indim={}, outdim={}, hiddims={}, nhidden={}, bias={}'.format(
            self.indim, self.outdim, self.hiddims, self.nhidden,
            self.bias is not None
        )


class AdvancedLSTMCell(nn.LSTMCell):
    # TODO: Add docstring
    # Extend LSTMCell to learn initial state
    def __init__(self, *args, **kwargs):
        super(AdvancedLSTMCell, self).__init__(*args, **kwargs)
        self.h0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())

    def initial_state(self, n):
        return (
            self.h0.expand(n, -1).contiguous(),
            self.c0.expand(n, -1).contiguous()
        )


class AdvancedLSTM(nn.LSTM):
    # Class for learning initial hidden states when using LSTMs
    def __init__(self, *args, **kwargs):
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
    # Performs pooling for pBLSTM

    @staticmethod
    def concate_sequence(sequence, length):
        '''pyramid BiLSTM, merge consecutive time step together'''
        shape = sequence.size()  # (S, B, D)

        # efficient using indexing, don't need iteration
        input_range = sequence.data.new(shape[0]//2).zero_().long()
        torch.arange(1, int(shape[0]), 2, out=input_range)
        input_concate = torch.cat(
            (sequence[input_range-1], sequence[input_range]), 2)

        length = np.array(length) // 2

        return input_concate, length

    def forward(self, seq):
        assert isinstance(seq, PackedSequence)
        h, seq_lengths = pad_packed_sequence(seq)

        # pyramid BiLSTM, merge consecutive time step together
        # input size should be (S, B, D)
        h, seq_lengths = self.concate_sequence(h, seq_lengths)
        h = pack_padded_sequence(h, seq_lengths)
        return h


class PyramidalLSTM(AdvancedLSTM):
    # Pyramidal LSTM

    def __init__(self, *args, **kwargs):
        super(PyramidalLSTM, self).__init__(*args, **kwargs)
        self.shuffle = SequenceShuffle()

    def forward(self, input, hx=None):
        return super(PyramidalLSTM, self).forward(self.shuffle(input), hx=hx)


class WeightdropLSTM(nn.LSTM):
    # TODO: add docstring.

    @staticmethod
    def generate_mask(p, inp=None):
        """Generate mask for random drop of network weights.

        p is the drop rate, aka the probability of being 0
        modified from https://github.com/salesforce/awd-lstm-lm/blob/master/
        using .new() to initialize from the same device, much more efficient
        """
        # TODO: how is inp=None valid?
        # TODO: Remove variable.
        return (Variable(inp.data.new(1, inp.size(1)).bernoulli_(1. - p)) / (1. - p)).expand_as(inp)

    def __init__(self, input_size, hidden_size, dropout_weight, dropout_bet):
        # TODO: Add docstring.
        super(WeightdropLSTM, self).__init__(input_size=input_size,
                                             hidden_size=hidden_size,
                                             bidirectional=True)
        self.old_weight_hh_l0 = self.weight_hh_l0
        self.weight_hh_l0 = None
        del self._parameters['weight_hh_l0']
        self.dropout_layer = nn.Dropout(dropout_weight)
        self.dropout_bet = dropout_bet

    def flatten_parameters(self):
        # TODO: Add docstring.
        # overwrite, prevent pytorch from putting all weight into a large chunk
        self._data_ptrs = []

    def forward(self, inp, hx=None):
        self.weight_hh_l0 = self.dropout_layer(self.old_weight_hh_l0)
        raw_output = None

        if self.training and self.dropout_bet != 0:
            input, batch_size = inp
            between_layer_mask = self.generate_mask(
                p=self.dropout_bet, inp=input)
            dropedinput = input * between_layer_mask
            inp = PackedSequence(dropedinput, batch_size)

        return super(WeightdropLSTM, self).forward(inp, hx=hx)


