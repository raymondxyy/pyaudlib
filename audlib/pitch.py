"""Implementations of Pitch (or sometimes called F0) Tracking."""
import math
import numpy as np
from scipy.signal import lfilter

from .sig.stproc import numframes, stana
from .sig.temporal import lpc_parcor, lpcerr, xcorr


class HistoPitch(object):
    """This class implements Mike Seltzer's histogram-based pitch tracker.

    Since the work is unpublished, please contact Dr. Richard Stern for the
    original manuscript.
    """

    def __init__(self, fbank, sr, wind, hop, lpc_order=0):
        """Instantiate a histogram-based pitch tracker.

        Parameters
        ----------
        fbank: sig.fbank class
            Filterbank specification.
        sr: int
            Sampling rate.
        wind: array_like
            Window function.
        hop: float/int
            Hop fraction or samples

        See Also
        --------
        sig.fbanks, sig.stproc, sig.window

        """
        super(HistoPitch, self).__init__()
        self.sr = sr
        self.fbank = fbank
        self.numframes = lambda sig: numframes(sig, sr, wind, hop, center=True)
        self.laglen = len(wind)

        def __stacf(sig):
            """Proceed ACF with LPC."""
            frames = stana(sig, sr, wind, hop, center=True)
            if not lpc_order:
                return np.asarray([xcorr(f, norm=True) for f in frames])
            out = np.empty_like(frames)
            for ii, frame in enumerate(frames):
                alphas, gain = lpc_parcor(frame, lpc_order)
                xerr = lpcerr(frame, alphas, gain=gain)
                out[ii] = xcorr(xerr, norm=True)
            return out

        self.stacf = __stacf

    def t0hist(self, sig, fmin=40, fmax=400):
        """Compute the fundamental period histogram."""
        # Step 1: bandpass filter signal
        bpsig = (self.fbank.filter(sig, k) for k in range(len(self.fbank)))

        # Step 1.5: lowpass the each bandpass signal
        lpsig = (self.lowpass(sig) for sig in bpsig)

        # Step 2: autocorrelation, peak detection, and T0 decision
        bpacf = (self.stacf(sig) for sig in lpsig)
        lmin = int(self.sr/fmax) if fmax else None
        lmax = math.ceil(self.sr/fmin) if fmin else None
        bpt0 = (self.findt0(acf, lmin=lmin, lmax=lmax) for acf in bpacf)

        # Step 3: initial pitch and voice estimate
        # This is the first time all subband signals are combined.
        t0hist = np.zeros((self.numframes(sig), self.laglen))
        for t0 in bpt0:
            for ii in range(len(t0)):
                t0hist[ii, t0[ii]] += 1

        return t0hist

    def pitchcontour(self, t0hist, fmin=40, fmax=400, neighbor=.02,
                     peakpercent=.28, pitchflux=.2):
        """Extract pitch contour of a signal."""
        uv, pitch = [], []
        _voiced, _unvoiced = True, False
        lmin = int(self.sr/fmax) if fmax else 0
        lmax = math.ceil(self.sr/fmin) if fmin else t0hist.shape[1]
        for frame in t0hist:
            t0candidate = np.argmax(frame)
            r1 = max(lmin, int(t0candidate*(1-neighbor)))
            r2 = min(lmax, int(t0candidate*(1+neighbor))+1)
            percent = frame[r1:r2].sum() / len(self.fbank)
            label = _voiced if percent > peakpercent else _unvoiced
            uv.append(label)
            pitch.append(self.sr/t0candidate if label else 0)

        # Step 4: smoothing and final pitch and voice estimate
        # 4.1 Merge undecided frames if there's adjacent voiced frame with
        # similar pitch
        vuseg0 = [0]  # record alternating boundaries between voice/unvoiced
        for ii, (ll, pp) in enumerate(zip(uv, pitch)):
            if ll == _unvoiced:
                if ii > 0:  # inspect previous frame
                    lpre, ppre = uv[ii-1], pitch[ii-1]
                    if lpre == _voiced and (np.abs(pp-ppre)/ppre) < .1:
                        uv[ii], pitch[ii] = _voiced, ppre
                if ii < len(pitch)-1:  # inspect next frame
                    lpos, ppos = uv[ii+1], pitch[ii+1]
                    if lpos == _voiced and (np.abs(pp-ppos)/ppos) < .1:
                        uv[ii], pitch[ii] = _voiced, ppos

            if ii > 0 and (uv[ii] != uv[ii-1]):
                vuseg0.append(ii)

        # 4.2 Move isolated voiced frames to unvoiced
        vuseglen0 = np.diff(vuseg0 + [len(pitch)])
        label = uv[0]
        vuseg1 = []
        ii = 0
        while ii < len(vuseg0):
            idx, seglen = vuseg0[ii], vuseglen0[ii]
            if label == _voiced and seglen < 3:
                for jj in range(idx, idx+seglen):
                    uv[jj], pitch[jj] = _unvoiced, 0
                ii += 2  # merge 3 segments into 1
            else:
                vuseg1.append(idx)
                label = not label
                ii += 1

        # 4.3 Move isolated unvoiced frames to voiced, with interpolated pitch
        vuseglen1 = np.diff(vuseg1 + [len(pitch)])
        label = uv[0]
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
                    ppre = pitch[idxpre+seglenpre-1]
                if ii == len(vuseg1)-1:
                    ppos = None
                else:
                    idxpos = vuseg1[ii+1]
                    ppos = pitch[idxpos]
                # Determine pitch by linear interpolation
                if ppos is None:
                    pp = ppre
                elif ppre is None:
                    pp = ppos
                else:
                    pp = (ppre+ppos)/2.
                for jj in range(idx, idx+seglen):
                    uv[jj], pitch[jj] = _voiced, pp
                ii += 2  # merge 3 segments into 1
            else:
                vuseg2.append(idx)
                label = not label
                ii += 1

        # 4.4 Smooth voiced frames
        # ADDED: Smooth voiced frames to prevent jumps
        vuseglen2 = np.diff(vuseg2 + [len(pitch)])
        label = uv[0]
        ii = 0
        while ii < len(vuseg2):
            ss, sl = vuseg2[ii], vuseglen2[ii]
            if label == _voiced:
                # F0 mean of voiced segment
                f0mean = np.mean(pitch[ss:ss+sl])
                # smooth left edge
                if (np.abs(pitch[ss]-f0mean)/f0mean) > pitchflux:
                    uv[ss], pitch[ss] = _voiced, pitch[ss+1]
                # smooth right edge
                if (np.abs(pitch[ss+sl-1]-f0mean)/f0mean) > pitchflux:
                    uv[ss+sl-1], pitch[ss+sl-1] = _voiced, pitch[ss+sl-2]
                for jj in range(ss+1, ss+sl-1):
                    ppre, pc, ppo = pitch[jj-1:jj+2]
                    pcompare = np.asarray([ppre, ppo])
                    if np.all(np.abs(pc-pcompare)/pcompare > pitchflux):
                        uv[jj], pitch[jj] = _voiced, (ppre+ppo)/2.
            ii += 1
            label = not label

        return uv, pitch

    def pitchcontour2(self, t0hist, uvdecision, fmin=40, fmax=400, neighbor=.02):
        """Alternative method by making use of the original U/V decision."""
        pitch = []
        ii = 0
        while ii < len(uvdecision):
            if not uvdecision[ii]:
                pitch.append(0)
                ii += 1
                continue
            # Process each voiced segment as a whole
            vstart = ii
            vend = ii+1
            while vend < len(uvdecision) and uvdecision[vend]:
                vend += 1
            t0path = self.bestpath(t0hist[vstart:vend],
                                   fmin=fmin, fmax=fmax,
                                   neighbor=neighbor)
            pitch.extend(self.sr/t0 if t0 > 0 else 0 for t0 in t0path)
            ii = vend

        return pitch

    @staticmethod
    def runningvar(uvdecision, pitch):
        """Calculate the running variance of pitch. Presever U/V decisions."""
        var = []
        ii = 0
        while ii < len(uvdecision):
            if not uvdecision[ii]:
                var.append(0)
                ii += 1
                continue
            # Process each voiced segment as a whole
            vstart = ii
            vend = ii+1
            while vend < len(uvdecision) and uvdecision[vend]:
                vend += 1

            vs = [np.log(p) for p in pitch[vstart:vend]]
            for jj in range(len(vs)):
                f0mean = sum(vs[:jj+1])/(jj+1)
                var.append(
                    np.sqrt(sum((f0-f0mean)**2 for f0 in vs[:jj+1])/(jj+1)))

            ii = vend

        return var

    def bestpath(self, t0hist, fmin=None, fmax=None, neighbor=.02):
        """Find the best path in a T0 histogram.

        Note that this function assumes `t0hist` is a voiced frame.
        """
        lmin = int(self.sr/fmax) if fmax else 0
        lmax = math.ceil(self.sr/fmin) if fmin else t0hist.shape[1]

        t0sums = []
        # Each entry is (value, T0, T0_tm1)
        t0sums.append([(v if (lmin <= t0 < lmax) else 0, t0, None)
                       for t0, v in enumerate(t0hist[0])])
        for tt in range(1, len(t0hist)):
            t0sums.append([])
            hist = t0hist[tt]
            for ll in range(len(hist)):
                rmin = max(0, int(ll*(1-neighbor)))
                rmax = min(len(hist), int(ll*(1+neighbor))+1)
                nbhist = [v for v, _, _ in t0sums[tt-1][rmin:rmax]]
                max_tm1 = max(nbhist)
                maxidx_tm1 = nbhist.index(max_tm1) + rmin
                val = hist[ll] if (lmin <= ll < lmax) else 0
                t0sums[tt].append((max_tm1 + val, ll, maxidx_tm1))

        # Backtrack to find full path
        ii = -1
        _, t0, parent = max(t0sums[ii])
        bestpath = [t0]
        while parent is not None:
            ii -= 1
            _, t0, parent = t0sums[ii][parent]
            bestpath.append(t0)

        return bestpath[::-1]

    @staticmethod
    def findt0(stacf, lmin=None, lmax=None):
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
        if lmin or lmax:
            lagmask = np.zeros_like(pkloc, dtype='bool_')
            lagmask[:, 0] = True  # preserve peak at lag=0
            r1 = max(lmin, 0) if lmin else 0
            r2 = min(lmax+1, pkloc.shape[1]) if lmax else pkloc.shape[1]
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
