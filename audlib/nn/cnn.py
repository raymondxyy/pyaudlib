"""Convolutional Neural Network Layers."""
import math

import torch
import torch.nn.functional as F
import torch.nn as nn

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
            Frame rate of the spectrogram in Hz.
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

        See Also
        --------
        sig.spectemp.strf

        """
        super(STRFConv, self).__init__()

        # For printing
        self.__rep = f"""STRF(fr={fr}, bins_per_octave={bins_per_octave},
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

    def strfs(self):
        """Make STRFs using current parameters."""
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

        return strfs.unsqueeze(1).to(self.rates_.device)

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
            sigspec.unsqueeze_(0)
        strfs = self.strfs().type_as(sigspec)
        return F.conv2d(sigspec.unsqueeze(1), strfs, padding=self.padding)

    def __repr__(self):
        return self.__rep


class STRFLayer(nn.Module):
    """A fully convolutional layer with STRF kernels at the bottom."""
    # NOTE: This version is UNSTABLE. Use with caution.
    def __init__(self, fr, bins_per_octave, suptime, supoct, nkern,
                 indim, outdim, kernel_size, nchan):
        """Construct a STRF layer from STRFConv and nonlinearities.

        The layer structure follows this flow:
        0. N x T x F batch comes in.
        1. N x K x T x F batch comes out of STRFConv.
        2. N x K x T x F batch comes out of BatchNorm.
        3. N x K x T x F batch comes out of nonlinearity.
        4. N x K x T'x F' batch comes out of Conv2d, where:
            T' = largest integer below T that is an integer power of 2
            F' = largest integer below F that is an integer power of 2
            If T != F, pool on the higher dimension only until they equal.
        5. N x K x T' x F' batch comes out of Conv2d.
        6. N x K x T''x F'' batch comes out of pooling, where:
            T'' = T'//2; F'' = F'//2
        7. Repeat step 5 and 6 until T'' and F'' reach 8.
        """
        super(STRFLayer, self).__init__()
        # Calculate input & output sizes in all layers
        itdim, ifdim = indim
        otdim, ofdim = outdim
        assert (nextpow2(otdim) == otdim) and (nextpow2(ofdim) == ofdim),\
            "Output dimensions must be integer multiples of 2"
        iosizes = [(itdim, ifdim)]  # input/output sizes in all layers
        while (itdim != otdim) or (ifdim != ofdim):
            if itdim != otdim:
                itdim = nextpow2(itdim)//2
            if ifdim != ofdim:
                ifdim = nextpow2(ifdim)//2
            iosizes.append((itdim, ifdim))
        nlayers = len(iosizes) - 1  # exclude STRF layer
        # linearly space between 2*nkern and nchan
        nchans = [max((2*nkern)/(2**n), nchan) for n in range(len(iosizes))]
        assert nchans[-1] >= nchan

        # Construct all layers
        layers = [STRFConv(fr, bins_per_octave, suptime, supoct, nkern)]
        for ll in range(nlayers):
            # fully convolutional layers here
            iit, iif = iosizes[ll]
            oot, oof = iosizes[ll+1]
            sst = 1 if iit == oot else 2
            ssf = 1 if iif == oof else 2
            fconv = [nn.Conv2d(nchans[ll], nchans[ll+1], kernel_size,
                     stride=(sst, ssf),
                     padding=(self.padsize(iit, oot, kernel_size, sst),
                     self.padsize(iif, oof, kernel_size, ssf))),
                     #nn.BatchNorm2d(nchans[ll+1]),
                     nn.ReLU()
                     ]
            layers.extend(fconv)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
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
        assert osize == math.floor((isize + 2*psize - ksize)/stride + 1)
        return psize
