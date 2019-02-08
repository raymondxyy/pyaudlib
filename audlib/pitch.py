"""Implementations of Pitch (or sometimes called F0) Tracking."""
import numpy as np
from scipy.signal import lfilter

from .sig.transform import stacf
from .sig.stproc import numframes


class HistoPitch(object):
    """This class implements Mike Seltzer's histogram-based pitch tracker.

    Since the work is unpublished, please contact Dr. Richard Stern for the
    original manuscript.
    """

    def __init__(self, fbank, sr, wind, hop):
        """Instantiate a histogram-based pitch tracker.

        Parameters
        ----------
        fbank: sig.fbank class
            Filterbank specification.

        """
        super(HistoPitch, self).__init__()
        self.sr = sr
        self.fbank = fbank
        self.numframes = lambda sig: numframes(sig, sr, wind, hop)
        self.laglen = len(wind)
        self.stacf = lambda sig: stacf(sig, sr, wind, hop, norm=True)

    def t0hist(self, sig):
        """Compute the fundamental period histogram."""
        # Step 1: bandpass filter signal
        bpsig = (self.fbank.filter(sig, k) for k in range(len(self.fbank)))

        # Step 1.5: lowpass the each bandpass signal
        lpsig = (self.lowpass(sig) for sig in bpsig)

        # Step 2: autocorrelation, peak detection, and T0 decision
        bpacf = (self.stacf(sig) for sig in lpsig)
        bpt0 = (self.findt0(acf, minlag=int(self.sr/300)) for acf in bpacf)

        # Step 3: initial pitch and voice estimate
        # This is the first time all subband signals are combined.
        t0hist = np.zeros((self.numframes(sig), self.laglen))
        for t0 in bpt0:
            for ii in range(len(t0)):
                t0hist[ii, t0[ii]] += 1

        return t0hist

    def pitchcontour(self, sig, neighbor=1, peakpercent=.28, pitchflux=.2):
        """Extract pitch contour of a signal."""
        t0hist = self.t0hist(sig)
        voicepitch = []
        _voiced, _unvoiced = True, False
        for frame in t0hist:
            t0candidate = np.argmax(frame)
            r1 = max(0, t0candidate-neighbor)
            r2 = min(len(frame), t0candidate+neighbor+1)
            percent = frame[r1:r2].sum() / len(self.fbank)
            label = _voiced if percent > peakpercent else _unvoiced
            voicepitch.append((label, self.sr/t0candidate if label else 0))

        # Step 4: smoothing and final pitch and voice estimate
        # 4.1 Merge undecided frames if there's adjacent voiced frame with
        # similar pitch
        vuseg0 = [0]  # record alternating boundaries between voice/unvoiced
        for ii, (ll, pp) in enumerate(voicepitch):
            if ll == _unvoiced:
                if ii > 0:  # inspect previous frame
                    lpre, ppre = voicepitch[ii-1]
                    if lpre == _voiced and (np.abs(pp-ppre)/ppre) < .1:
                        voicepitch[ii] = _voiced, ppre
                if ii < len(voicepitch)-1:  # inspect next frame
                    lpos, ppos = voicepitch[ii+1]
                    if lpos == _voiced and (np.abs(pp-ppos)/ppos) < .1:
                        voicepitch[ii] = _voiced, ppos

            if ii > 0 and (voicepitch[ii][0] != voicepitch[ii-1][0]):
                vuseg0.append(ii)

        # 4.2 Move isolated voiced frames to unvoiced
        vuseglen0 = np.diff(vuseg0 + [len(voicepitch)])
        label = voicepitch[0][0]
        vuseg1 = []
        ii = 0
        while ii < len(vuseg0):
            idx, seglen = vuseg0[ii], vuseglen0[ii]
            if label == _voiced and seglen < 3:
                for jj in range(idx, idx+seglen):
                    voicepitch[jj] = (_unvoiced, 0)
                ii += 2  # merge 3 segments into 1
            else:
                vuseg1.append(idx)
                label = not label
                ii += 1

        # 4.3 Move isolated unvoiced frames to voiced, with interpolated pitch
        vuseglen1 = np.diff(vuseg1 + [len(voicepitch)])
        label = voicepitch[0][0]
        vuseg2 = []
        ii = 0
        while ii < len(vuseg1):
            idx, seglen = vuseg1[ii], vuseglen1[ii]
            if label == _unvoiced and seglen < 3:
                # Find adjacent pitches
                if ii == 0:
                    ppre = None
                else:
                    idxpre, seglenpre = vuseg1[ii-1], vuseglen1[ii-1]
                    ppre = voicepitch[idxpre+seglenpre-1][1]
                if ii == len(vuseg1)-1:
                    ppos = None
                else:
                    idxpos = vuseg1[ii+1]
                    ppos = voicepitch[idxpos][1]
                # Determine pitch by linear interpolation
                if ppos is None:
                    pp = ppre
                elif ppre is None:
                    pp = ppos
                else:
                    pp = (ppre+ppos)/2.
                for jj in range(idx, idx+seglen):
                    voicepitch[jj] = (_voiced, pp)
                ii += 2  # merge 3 segments into 1
            else:
                vuseg2.append(idx)
                label = not label
                ii += 1

        # 4.4 Smooth voiced frames
        # ADDED: Smooth voiced frames to prevent jumps
        vuseglen2 = np.diff(vuseg2 + [len(voicepitch)])
        label = voicepitch[0][0]
        ii = 0
        while ii < len(vuseg2):
            ss, sl = vuseg2[ii], vuseglen2[ii]
            if label == _voiced:
                # F0 mean of voiced segment
                f0mean = np.mean([p for (_, p) in voicepitch[ss:ss+sl]])
                # smooth left edge
                if (np.abs(voicepitch[ss][1]-f0mean)/f0mean) > pitchflux:
                    voicepitch[ss] = _voiced, voicepitch[ss+1][1]
                # smooth right edge
                if (np.abs(voicepitch[ss+sl-1][1]-f0mean)/f0mean) > pitchflux:
                    voicepitch[ss+sl-1] = _voiced, voicepitch[ss+sl-2][1]
                for jj in range(ss+1, ss+sl-1):
                    ppre, pc, ppo = [p for (_, p) in voicepitch[jj-1:jj+2]]
                    pcompare = np.asarray([ppre, ppo])
                    if np.all(np.abs(pc-pcompare)/pcompare > pitchflux):
                        voicepitch[jj] = _voiced, (ppre+ppo)/2.
            ii += 1
            label = not label

        return voicepitch

    @staticmethod
    def findt0(stacf, minlag=None, maxlag=None):
        """Find fundamental period T0 given the short-time ACF."""
        def findpeaks(stsig):
            """Extract local maxima location of a short-time signal.

            Returns
            -------
                Peak location as a mask.

            """
            nbleft = np.zeros_like(stsig)
            nbleft[:, 1:] = stsig[:, :-1]
            nbright = np.zeros_like(stsig)
            nbright[:, :-1] = stsig[:, 1:]
            return np.logical_and(stsig > nbleft, stsig > nbright)

        pkloc = findpeaks(stacf)

        # Apply optional mask to control search range
        if minlag or maxlag:
            lagmask = np.zeros_like(pkloc)
            lagmask[:, 0] = True  # preserve peak at lag=0
            r1 = max(minlag, 0) if minlag else 0
            r2 = min(maxlag+1, pkloc.shape[1]) if maxlag else pkloc.shape[1]
            lagmask[:, r1:r2] = True
            pkloc = np.logical_and(pkloc, lagmask)
        numpeaks = pkloc[:, 1:].sum(axis=1)  # excluding lag=0

        t0 = np.zeros_like(numpeaks)
        for ii, (nn, pkframe) in enumerate(zip(numpeaks, pkloc)):
            if nn == 0:
                continue
            elif nn == 1:
                t0[ii] = np.argmax(pkframe)
            elif nn == 2:
                lags, lagvals = np.where(pkframe)[0], stacf[ii, pkframe]
                t0[ii] = lags[0] if lagvals[0] > lagvals[1] else lags[1]
            else:  # multiple peaks
                lags, lagvals = np.where(pkframe)[0], stacf[ii, pkframe]
                assert lagvals[0] > lagvals[1]
                jj = 1
                while lagvals[jj] < lagvals[jj-1]:  # find valley
                    jj += 1
                    if jj == len(lags):
                        break
                if jj == len(lags):
                    # No valley; use first max peak
                    t0[ii] = lags[1]
                else:  # Second peak exists; find max peak after valley
                    t0[ii] = lags[jj:][np.argmax(lagvals[jj:])]

        return t0

    @staticmethod
    def lowpass(sig):
        """Lowpass filter each bandpass signal."""
        b = np.array([2.2189438e-07, 2.6627326e-06,
                      1.4645029e-05, 4.8816764e-05,
                      1.0983772e-04, 1.7574035e-04,
                      2.0503041e-04, 1.7574035e-04,
                      1.0983772e-04, 4.8816764e-05,
                      1.4645029e-05, 2.6627326e-06,
                      2.2189438e-07])
        a = np.array([1.0000000e+00, -6.8844083e+00,
                      2.2434055e+01, -4.5510026e+01,
                      6.3772549e+01, -6.4850120e+01,
                      4.8964312e+01, -2.7609199e+01,
                      1.1521776e+01, -3.4661888e+00,
                      7.1278837e-01, -8.9879358e-02,
                      5.2513147e-03])

        return lfilter(b, a, sig)
