"""STRFNet."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import MLP
from .signal import hilbert
from ..sig.util import nextpow2


class STRFConv(nn.Module):
    """Spectrotemporal receptive field (STRF)-based convolution."""

    def __init__(self, fr, bins_per_octave, suptime, supoct, nkern,
                 rates=None, scales=None, phis=None, thetas=None):
        """Instantiate a STRF convolution layer.

        Parameters
        ----------
        fr: int
            Frame rate of the incoming spectrogram in Hz.
            e.g. spectrogram with 10ms hop size has frame rate 100Hz.
        bins_per_octave: int
            Number of frequency dimensions per octave in the spectrogram.
        suptime: float
            Maximum time support in seconds.
            All kernels will span [0, suptime) seconds.
        supoct: float
            Maximum frequency support in number of octaves.
            All kernels will span [-supoct, supoct] octaves.
        nkern: int
            Number of learnable STRF kernels.
        rates: array_like, (None)
            Init. for learnable stretch factor in time.
            Dimension must match `nkern` if specified.
        scales: int or float, (None)
            Init. for learnable stretch factor in frequency.
            Dimension must match `nkern` if specified.
        phis: float, (None)
            Init. for learnable phase shift of spectral evolution in radians.
            Dimension must match `nkern` if specified.
        thetas: float, (None)
            Init. for learnable phase shift of time evolution in radians.
            Dimension must match `nkern` if specified.

        """
        super(STRFConv, self).__init__()

        # For printing
        self.__rep = f"""STRFConv(fr={fr}, bins_per_octave={bins_per_octave},
            suptime={suptime}, supoct={supoct}, nkern={nkern},
            rates={rates}, scales={scales}, phis={phis},
            thetas={thetas})"""

        # Determine time & frequency support
        _fsteps = int(supoct * bins_per_octave)  # spectral step on one side
        self.supf = torch.linspace(-supoct, supoct, steps=2*_fsteps+1)
        _tsteps = int(fr*suptime)
        if _tsteps % 2 == 0:  # force odd number
            _tsteps += 1
        self.supt = torch.arange(_tsteps).type_as(self.supf)/fr
        self.padding = (_tsteps//2, _fsteps)

        # Set up learnable parameters
        for param in (rates, scales, phis, thetas):
            assert (not param) or len(param) == nkern
        if not rates:
            rates = torch.rand(nkern) * 10
        if not scales:
            scales = torch.rand(nkern) / 5
        if not phis:
            phis = 2*math.pi * torch.rand(nkern)
        if not thetas:
            thetas = 2*math.pi * torch.rand(nkern)
        self.rates_ = nn.Parameter(torch.Tensor(rates))
        self.scales_ = nn.Parameter(torch.Tensor(scales))
        self.phis_ = nn.Parameter(torch.Tensor(phis))
        self.thetas_ = nn.Parameter(torch.Tensor(thetas))

    @staticmethod
    def _hs(x, scale):
        """Spectral evolution."""
        sx = scale * x
        return scale * (1-(2*math.pi*sx)**2) * torch.exp(-(2*math.pi*sx)**2/2)

    @staticmethod
    def _ht(t, rate):
        """Temporal evolution."""
        rt = rate * t
        return rate * rt**2 * torch.exp(-3.5*rt) * torch.sin(2*math.pi*rt)

    def strfgen(self):
        """Generate downward and upward STRFs using current parameters."""
        if self.supt.device != self.rates_.device:  # for first run
            self.supt = self.supt.to(self.rates_.device)
            self.supf = self.supf.to(self.rates_.device)
        K, S, T = len(self.rates_), len(self.supf), len(self.supt)
        # Construct STRFs
        hs = self._hs(self.supf, self.scales_.view(K, 1))
        ht = self._ht(self.supt, self.rates_.view(K, 1))
        hsa = hilbert(hs)
        hta = hilbert(ht)
        hirs = hs * torch.cos(self.phis_.view(K, 1)) \
            + hsa[..., 1] * torch.sin(self.phis_.view(K, 1))
        hirt = ht * torch.cos(self.thetas_.view(K, 1)) \
            + hta[..., 1] * torch.sin(self.thetas_.view(K, 1))
        hirs_ = hilbert(hirs)  # K x S x 2
        hirt_ = hilbert(hirt)  # K x T x 2

        # for a single strf:
        # strfdn = hirt_[:, 0] * hirs_[:, 0] - hirt_[:, 1] * hirs_[:, 1]
        # strfup = hirt_[:, 0] * hirs_[:, 0] + hirt_[:, 1] * hirs_[:, 1]
        rreal = hirt_[..., 0].view(K, T, 1) * hirs_[..., 0].view(K, 1, S)
        rimag = hirt_[..., 1].view(K, T, 1) * hirs_[..., 1].view(K, 1, S)
        strfs = torch.cat((rreal-rimag, rreal+rimag), 0)  # 2K x T x S

        return strfs

    def forward(self, sigspec):
        """Convolve a spectrographic representation with all STRF kernels.

        Parameters
        ----------
        sigspec: `torch.Tensor` (batch_size, time_dim, freq_dim)
            Batch of spectrograms.
            The frequency dimension should be logarithmically spaced.

        Returns
        -------
        features: `torch.Tensor` (batch_size, nkern, time_dim, freq_dim)
            Batch of STRF activatations.

        """
        if len(sigspec.shape) == 2:  # expand batch dimension if single eg
            sigspec = sigspec.unsqueeze(0)
        strfs = self.strfgen().unsqueeze(1).type_as(sigspec)
        return F.conv2d(sigspec.unsqueeze(1), strfs, padding=self.padding)

    def __repr__(self):
        return self.__rep


class STRFLayerFC(nn.Module):
    """A Fully Convolutional layer with STRF kernels at the bottom."""
    def __init__(self, fr, bins_per_octave, suptime, supoct, nkern,
                 indim, outdim, kernel_size, nchan_conv, nchan_out):
        """Construct a STRF layer from STRFConv and nonlinearities.

        The layer structure follows this flow:
        0. N x T x F batch comes in, where:
            N = batch size
            T, F = indim
        1. N x K x T x F batch comes out of STRFConv, where:
            K = 2 x nkern (upward and downward drifting pairs)
            The STRF kernel size is determined by fr, bpo, suptime, and supoct.
        2. N x K x T x F batch comes out of BatchNorm.
        3. N x K x T x F batch comes out of nonlinearity.
        4. N x K' x T'x F' batch comes out of Conv2d, where:
            T' = largest integer below T that is an integer power of 2
            F' = largest integer below F that is an integer power of 2
            K' = nchan_conv
            The Conv2d kernel size is determined by kernel_size.
        5. N x K' x T' x F' batch comes out of LayerNorm and ReLU.
        6. Repeat step 4 and 5 until T', F' == outdim. Should that happen, also
           make the output channel == nchan_out. This output will be the input
           to MLP for classification.

        CAVEATS:
        `outdim` must be integer power of 2. This is for the convenience of
        taking a stride of 2 to halve the feature dimension in each conv layer.

        """
        super(STRFLayerFC, self).__init__()
        # Calculate input & output sizes in all layers
        itdim, ifdim = indim
        otdim, ofdim = outdim
        assert (nextpow2(otdim) == otdim) and (nextpow2(ofdim) == ofdim),\
            "Output dimensions must be integer power of 2"
        iosizes = [(itdim, ifdim)]  # input/output sizes in all layers
        while (itdim != otdim) or (ifdim != ofdim):
            if itdim != otdim:
                itdim = nextpow2(itdim)//2
            if ifdim != ofdim:
                ifdim = nextpow2(ifdim)//2
            iosizes.append((itdim, ifdim))
        nlayers = len(iosizes) - 1  # exclude STRFConv
        nchans = [2*nkern] + [nchan_conv]*(nlayers-1) + [nchan_out]

        # Construct all layers; start with STRF + pooling
        iit, iif = iosizes[0]
        oot, oof = iosizes[1]
        if type(kernel_size) == int:
            kernel_size = kernel_size, kernel_size

        # Start with the STRFConv layer
        self.strfconv = STRFConv(
            fr, bins_per_octave, suptime, supoct, nkern,
            rates=[(n % 3 + 1) for n in range(nkern)],
            scales=[.3*(n % 5 + 1) for n in range(nkern)])

        # Then fully convolutional layers here
        fullyconv = []
        iiconv = 0
        for ll in range(iiconv, nlayers):
            iit, iif = iosizes[ll]
            oot, oof = iosizes[ll+1]
            kkt, kkf = kernel_size  # changing kernel size doesn't help much
            sst = 1 if iit == oot else 2
            ssf = 1 if iif == oof else 2
            fconv = [nn.Conv2d(nchans[ll], nchans[ll+1], (kkt, kkf),
                     stride=(sst, ssf),
                     padding=(self.padsize(iit, oot, kkt, sst),
                     self.padsize(iif, oof, kkf, ssf))),
                     nn.LayerNorm([nchans[ll+1], oot, oof]),
                     nn.ReLU()
                     ]
            fullyconv.extend(fconv)

        self.fullyconv = nn.ModuleList(fullyconv)

    def forward(self, x):
        x = self.strfconv(x)
        for layer in self.fullyconv:
            x = layer(x)
        return x

    @staticmethod
    def padsize(isize, osize, ksize, stride):
        """Calculate desired padding size given input and output dimensions.

        Follows PyTorch's definition, ignoring dilation:
            osize = floor((isize + 2*padsize - (ksize-1) - 1)/stride) + 1
        =>  stride(osize-1) = isize + 2*padsize - (ksize-1) - 1
        =>  padsize = ceil((stride(osize-1)-isize+ksize) / 2)
        """
        psize = math.ceil((stride*(osize-1) - isize + ksize) / 2)
        oosize = math.floor((isize + 2*psize - ksize)/stride + 1)
        assert oosize == osize,\
            f"{isize}-{osize}-{ksize}-{stride}-{psize}-{oosize}"
        return psize

    @staticmethod
    def outsize(isize, psize, ksize, stride):
        """Calculate output size given input and other configs."""
        return math.floor((isize + 2*psize - ksize)/stride + 1)


class STRFNetWASPAA2019(nn.Module):
    """A CNN+MLP classifier that has STRFLayers in the CNN."""

    @staticmethod
    def is_strf_param(nm):
        """Check if a parameter name string is one of STRF parameters."""
        return any(n in nm for n in ("rates_", "scales_", "phis_", "thetas_"))

    def __init__(self, fr, bins_per_octave, suptimes, supocts, nkerns,
                 num_classes, indim, mlp_hidden, outdim=(4, 4), kernel_size=5,
                 nchan_conv=60, nchan_out=32):
        """Instantiate a CNN with STRF extractors in the bottom layer.

        Parameters
        ----------
        fr: int
            Frame rate in Hz.
        bins_per_octave: int
            Number of bins per octave.
        tdim: int
            Time dimension of spectrogram feature.
        fdim: int
            Frequency dimension of spectrogram feature.
        suptimes: list of float
            Time support of STRFs.
        supocts: list of float
            Frequency support of STRFs.
        nkerns: list of int
            Number of STRFs per setup.
        num_classes: int
            N-way classification.
        indim: tuple(int, int)
            Feature input dimension.
        mlp_hidden: list(int)
            MLP hidden layer dimensions in a list.
        outdim: tuple(int, int), (4, 4)
            Output feature dimension coming out of convolutional layers.
        kernel_size: int or tuple(int, int), (5, 5)
            Kernel size of fully convolutional layer.
        nchan_conv: int, 60
            Number of output channels in each fully convolutional layer.
        nchan_out: int, 32
            Number of output channels in the final convolutional layer.

        See Also
        --------
        STRFConv, STRFLayer

        """
        super(STRFNetWASPAA2019, self).__init__()
        assert len(suptimes) == len(supocts) == len(nkerns)
        self.strflayers = nn.ModuleList(
            [STRFLayerFC(fr, bins_per_octave, t, f, k, indim, outdim,
                         kernel_size, nchan_conv, nchan_out)
                for t, f, k in zip(suptimes, supocts, nkerns)])
        # Input to MLP will be flattened CNN output
        inmlp = outdim[0] * outdim[1] * nchan_out * len(supocts)
        self.mlp = MLP(inmlp, num_classes, hiddims=mlp_hidden,
                       activate_out=nn.LogSoftmax(dim=1),
                       batchnorm=[True]*len(mlp_hidden))

    def forward(self, x):
        out = torch.cat([
            layer(x).view(len(x), -1) for layer in self.strflayers], -1)
        out = self.mlp(out)
        return out
