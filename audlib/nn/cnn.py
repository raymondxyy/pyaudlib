"""Convolutional Neural Network Layers."""
import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from .signal import hilbert


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

        # Determine time & frequency support
        _fsteps = int(supoct * bins_per_octave)  # spectral step on one side
        self.supf = torch.linspace(-supoct, supoct, steps=2*_fsteps+1)
        self.supt = torch.arange(int(fr*suptime)).type_as(self.supf)/fr
        self.padding = (0, _fsteps)

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
            sigspec.unsqueeze_(0)
        sigspec.unsqueeze_(1)
        strfs = self.strfs().type_as(sigspec).unsqueeze(1).to(sigspec.device)
        return F.conv2d(sigspec, strfs, padding=self.padding)
