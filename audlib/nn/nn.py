"""Neural Network modules."""
import torch
from torch.nn.parameter import Parameter
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
        self.layers.append(nn.Linear(indims[-1], outdims[-1], bias=bias))
        if activate_out is not None:
            self.layers.append(activate_out)

    def forward(self, x):
        """One forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x
