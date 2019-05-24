"""A collection of recurrent neural networks for processing time series."""
import torch
from torch.nn import Module
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.utils.rnn import PackedSequence

from .nn import MLP
from .util import detach


class ExtendedLSTMCell(nn.LSTMCell):
    """Extended LSTMCell with learnable initial states."""

    def __init__(self, *args, **kwargs):
        """Initialize an extended LSTMCell.

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
        super(ExtendedLSTM, self).__init__(*args, **kwargs)
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

    def forward(self, x, hx=None):
        if hx is None:
            if isinstance(x, PackedSequence):
                hx = self.initial_state(x.batch_sizes[0])
            else:  # tensor
                hx = self.initial_state(
                    len(x) if self.batch_first else x.shape[1])
        return super(ExtendedLSTM, self).forward(x, hx=hx)


class UnpackedSequence(object):
    """Unpack a PackedSequence to original (unpadded) examples."""
    def __init__(self, ps):
        """Construct an unpacked sequence object."""
        self.packed_sequence = ps
        lencnt = [int(n) for n in ps.batch_sizes[:-1]-ps.batch_sizes[1:]] + [1]
        self.seqlengths = []  # seqlengths[i] contains length of example i
        for num, ll in zip(lencnt[::-1], range(len(lencnt), 0, -1)):
            self.seqlengths.extend([ll] * num)

    def __len__(self):
        """Return number of examples in this batch."""
        return self.packed_sequence.batch_sizes[0]

    def __getitem__(self, i):
        """Get original idx-th item in the batch."""
        idx = torch.LongTensor(self.seqlengths[i])
        idx[0] = i
        idx[1:] = self.packed_sequence.batch_sizes[:self.seqlengths[i]-1]
        ei = self.packed_sequence.data[idx.cumsum(0)]  # example i
        return ei


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


class PyramidalLSTM(ExtendedLSTM):
    """Pyramidal LSTM could reduce the sequence length by half."""

    def __init__(self, *args, **kwargs):
        """See ExtendedLSTM."""
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
