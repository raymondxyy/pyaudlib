"""A collection of recurrent neural networks for processing time series."""
import torch
from torch.nn import Module
import torch.nn as nn

from .nn import MLP
from .util import detach


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
        hiddims: list of int, optional
            Hidden dimension for the ARMA processing. Default to [], meaning
            no hidden layers.
        bias: bool, optional
            Enable bias in forward pass. Default to True.

        """
        super(ARMA, self).__init__()
        self.indim = indim
        self.arsize = arsize if arsize > 0 else None
        self.masize = masize if masize > 0 else 0
        if self.arsize:
            self.arnet = MLP(arsize*indim, indim, hiddims=hiddims, bias=bias,
                             activate_hid=nn.Tanh(),
                             activate_out=nn.Tanh())
        self.manet = MLP((self.masize+1)*indim, indim, hiddims=hiddims,
                         bias=bias,
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
        if x_tm is None and self.masize:
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
