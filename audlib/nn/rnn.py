"""A collection of Recurrent Neural Networks for processing time series."""
import torch
from torch.nn import Module
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from .nn import MLP
from .util import detach, UnpackedSequence


class ExtendedLSTMCell(nn.LSTMCell):
    """Extended LSTMCell with learnable initial states."""

    def __init__(self, *args, **kwargs):
        """Initialize an extended LSTMCell.

        See the official documentation of LSTMCell.
        """
        super(ExtendedLSTMCell, self).__init__(*args, **kwargs)
        self.h0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())

    def initial_state(self, n):
        return (
            self.h0.expand(n, -1).contiguous(),
            self.c0.expand(n, -1).contiguous()
        )


class ExtendedLSTM(nn.LSTM):
    """Extended LSTM with LSTM cells that have learnable initial states."""

    def __init__(self, *args, **kwargs):
        """Initialize a LSTM.

        See the official documentation of LSTM.
        """
        super(ExtendedLSTM, self).__init__(*args, **kwargs)
        bi = 2 if self.bidirectional else 1
        self.h0 = nn.Parameter(torch.FloatTensor(
            self.num_layers*bi, 1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(
            self.num_layers*bi, 1, self.hidden_size).zero_())

    def initial_state(self, n):
        return (
            self.h0.expand(-1, n, -1).contiguous(),
            self.c0.expand(-1, n, -1).contiguous()
        )

    def forward(self, x, hx=None):
        if hx is None:
            if isinstance(x, PackedSequence):
                hx = self.initial_state(x.batch_sizes[0])
            else:  # tensor
                hx = self.initial_state(
                    len(x) if self.batch_first else x.shape[1])
        return super(ExtendedLSTM, self).forward(x, hx=hx)


class ExtendedGRUCell(nn.GRUCell):
    """Extended GRUCell with learnable initial states."""

    def __init__(self, *args, **kwargs):
        """Initialize an extended GRUCell.

        See nn.GRUCell for input/output instructions.

        """
        super(ExtendedGRUCell, self).__init__(*args, **kwargs)
        self.h0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())

    def initial_state(self, n):
        return self.h0.expand(n, -1).contiguous()


class ExtendedGRU(nn.GRU):
    """Extended GRU with GRU cells that have learnable initial states."""

    def __init__(self, *args, **kwargs):
        """Initialize an extended GRU.

        See nn.GRU for input/output instructions.

        """
        super(ExtendedGRU, self).__init__(*args, **kwargs)
        bi = 2 if self.bidirectional else 1
        self.h0 = nn.Parameter(torch.FloatTensor(
            self.num_layers*bi, 1, self.hidden_size).zero_())

    def initial_state(self, n):
        return self.h0.expand(-1, n, -1).contiguous()

    def forward(self, x, hx=None):
        if hx is None:
            if isinstance(x, PackedSequence):
                hx = self.initial_state(x.batch_sizes[0])
            else:  # tensor
                hx = self.initial_state(
                    len(x) if self.batch_first else x.shape[1])
        return super(ExtendedGRU, self).forward(x, hx=hx)


class ConcatPool(nn.Module):
    """Pooling by concatenating consecutive time frames."""

    def __init__(self, decimate):
        """Instantiate a pooling layer with decimation."""
        super(ConcatPool, self).__init__()
        assert decimate > 1, "Invalid decimation factor."
        self.decimate = int(decimate)

    def concatdn(self, seq, padvalue):
        """Decimate a PackedSequence by conocatenating consecutive frames."""
        oldframes = seq.size(0)
        newframes = (oldframes+self.decimate-1) // self.decimate
        padlen = newframes*self.decimate
        # NOTE: only supporting 2-D sequence for now
        padseq = seq.data.new(padlen, seq.size(1))
        padseq[:oldframes, :] = seq
        padseq[oldframes:] = padvalue
        # get ending index for each frame after concatenation
        rng = seq.data.new(newframes).long()
        torch.arange(self.decimate-1, padlen, self.decimate, out=rng)
        catseq = torch.cat(
            [padseq[rng-ii] for ii in range(self.decimate-1, -1, -1)], 1)

        return catseq

    def forward(self, packedseq, padvalue=0.):
        assert isinstance(packedseq, PackedSequence)
        out = []
        for seq in UnpackedSequence(packedseq):
            out.append(self.concatdn(seq, padvalue))

        return pack_sequence(out)


class PyramidalLSTM(ExtendedLSTM):
    """Pyramidal LSTM could reduce the sequence length by half."""

    def __init__(self, *args, decimate=2, **kwargs):
        """See ExtendedLSTM."""
        super(PyramidalLSTM, self).__init__(*args, **kwargs)
        self.shuffle = ConcatPool(decimate)

    def forward(self, x, hx=None):
        return super(PyramidalLSTM, self).forward(self.shuffle(x), hx=hx)


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
        """Overwrite this method, which prevents PyTorch from putting all weight
        into a large chunk.
        """
        self._data_ptrs = []

    def forward(self, inp, hx=None):
        self.weight_hh_l0 = self.dropout_layer(self.old_weight_hh_l0)

        if self.training and self.dropout_inp != 0:
            input, batch_size = inp
            between_layer_mask = self.generate_mask(
                self.dropout_inp, input)
            droppedinput = input * between_layer_mask
            inp = PackedSequence(droppedinput, batch_size)

        return super(WeightdropLSTM, self).forward(inp, hx=hx)


class ResRNN(torch.nn.Module):
    """GRU or LSTM with residual connection between layers."""

    def __init__(self, indim, residues, gru=True):
        """Instantiate a ResRNN network.

        Parameters
        ----------
        indim: int
            Input feature dimension. This will also be the output dimension.
        residues: tuple(bool)
            Whether there is residual connection in each hidden layer.

        Keyword Parameters
        ------------------
        gru: bool, True
            If True, use GRU as the recurrent unit. Otherwise use LSTM.

        """
        super(ResRNN, self).__init__()
        self.indim = self.outdim = indim
        self.use_gru = gru
        cell = ExtendedGRU if gru else ExtendedLSTM
        self.rnns = nn.ModuleList([cell(indim, indim) for _ in residues])
        self.residues = residues

    def forward(self, x):
        """Assume x is a PackedSequence."""
        assert isinstance(x, PackedSequence), "Input is not PackedSequence!"

        for rnn, res in zip(self.rnns, self.residues):
            h, hn = rnn(x)
            if res:
                x = PackedSequence(data=h.data+x.data,
                                   batch_sizes=h.batch_sizes)
            else:
                x = h

        return x, hn


class ARMA(Module):
    """A RNN mimicking the AutoRegressive Moving Averge linear filter."""

    def __init__(self, indim, arsize, masize, hiddims=[], bias=True):
        """Instantiate an ARMA RNN.

        Parameters
        ----------
        indim: int
            Input dimension (per time sample).
        arsize: int
            Autoregressive filter tap.
        masize: int
            Moving average filter tap.
        hiddims: list ofint, optional
            Hidden dimensions specified in a list. Default to empty list,
            which means no hidden layers.
        bias: bool, optional
            Enable bias in forward pass. Default to True.

        See Also
        --------
        nn.MLP

        """
        super(ARMA, self).__init__()
        self.indim = indim
        self.arsize = arsize if arsize > 0 else None
        self.masize = masize if masize > 0 else 0
        if self.arsize:
            self.arnet = MLP(indim*self.arsize, indim, hiddims, bias=bias,
                             activate_hid=nn.Tanh(),
                             activate_out=nn.Tanh())
        self.manet = MLP(indim*(self.masize+1), indim, hiddims, bias=bias,
                         activate_hid=nn.Tanh(),
                         activate_out=nn.Tanh())
        # For final combination
        self.activate_out = nn.Tanh()

    def init_hidden(self):
        """Initialize initial conditions of the filter."""
        self.hidden = torch.zeros(self.arsize*self.indim)

    def forward(self, x_t, x_tm=None, y_tm=None):
        """One forward pass given input signal and initial condition.

        Each provided tensor must follow this convention:
            [batch_size x feature_dimension]

        Parameters
        ----------
        x_t: torch.tensor
            Samples in batch of x at time frame t.
            Assume x_t is a tensor of size [batch x feat].
        x_tm: torch.tensor
            Samples in batch of x at time frame [t-masize, ..., t-1].
            If provided, must be of size [batch x (masize*feat)]
        y_tm: torch.tensor, optional
            Initial conditions for y.
            If provided, must be of size [batch x (arsize*feat)].

        Returns
        -------
        y_t: torch.tensor
            Filtered signal of size [batch x feat].

        """
        if (y_tm is None) and self.arsize:
            y_tm = torch.zeros(x_t.size(0), self.arsize*self.indim)
        if (x_tm is None) and self.masize:
            x_tm = torch.zeros(x_t.size(0), self.masize*self.indim)

        ar = 0 if self.arsize is None else self.arnet(y_tm)

        ma = self.manet(x_t) if self.masize == 0 else self.manet(
            torch.cat((x_tm, x_t), 1))

        # might consider more complex combination here
        return self.activate_out(ma+ar)

    def filter(self, x, zi=None):
        """Filter a signal by the ARMA-RNN.

        Parameters
        ----------
        x: torch.tensor
            Input signal in batch of size [batch x seq x feat].
        zi: torch.tensor, optional
            Initial condition.
            Must be of size [batch x (arsize*feat)]. Assume to be all 0s.

        """
        if zi is None:
            zi = x.new_zeros(x.size(0), self.arsize*self.indim)
        y = x.new_empty(x.shape)
        for tt in range(x.size(1)):
            if (tt-self.masize < 0):
                x_tm = torch.cat(
                    (x.new_zeros(x.size(0), (self.masize-tt)*self.indim),
                     x[:, :tt, :].reshape((x.size(0), tt*self.indim))), 1)
            else:
                x_tm = x[:, (tt-self.masize):tt, :].reshape(
                    (x.size(0), self.masize*self.indim))

            if (tt-self.arsize < 0):
                y_tm = torch.cat(
                    (x.new_zeros(x.size(0), (self.arsize-tt)*self.indim),
                     y[:, :tt, :].reshape((x.size(0), tt*self.indim))), 1)
            else:
                y_tm = y[:, (tt-self.arsize):tt, :].reshape(
                    (x.size(0), self.arsize*self.indim))

            y_t = self.forward(x[:, tt, :], x_tm, y_tm)
            y_t, = detach((y_t,))

            y[:, tt, :] = y_t

        return y

    def extra_repr(self):
        """Vebose info."""
        return 'arsize={}, masize={}'.format(self.arsize, self.masize)
