"""Implementations of PITCH tracking algorithms."""
from collections import deque
import math
import numpy as np
import numpy.fft as fft
from scipy.signal import lfilter

from .sig.stproc import numframes, stana
from .sig.temporal import lpc_parcor, lpcerr, xcorr
from .sig.util import clipcenter, clipcenter3lvl, nextpow2

ACF_FLOOR = 1e-4


class ACF(object):
    """Pitch detection using the autocorrelation function."""

    def __init__(self, sr, wind, hop, lpc_order=0):
        """Instantiate a ACF pitch tracker.

        Parameters
        ----------
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
        super(ACF, self).__init__()
        self.sr = sr
        self.numframes = lambda sig: numframes(sig, wind, hop, center=True)
        self.laglen = len(wind)

        def __stacf(sig):
            """Proceed ACF with LPC."""
            frames = stana(sig, wind, hop, center=True)
            if not lpc_order:
                return np.asarray([xcorr(f, norm=True) for f in frames])
            out = np.empty_like(frames)
            for ii, frame in enumerate(frames):
                alphas, gain = lpc_parcor(frame, lpc_order)
                xerr = lpcerr(frame, alphas, gain=gain)
                out[ii] = xcorr(xerr, norm=True)
            return out

        self.stacf = __stacf

    def pitchcontour(self, sig, fmin=40, fmax=400):
        """Extract pitch contour of a signal."""
        stacf = self.stacf(sig)
        lmax = math.ceil(self.sr / fmin) if fmin else math.ceil(self.sr/40)
        lmin = int(self.sr/fmax) if fmax else int(self.sr/400)
        pitch = []
        for acf in stacf:
            pklag = np.argmax(acf[lmin:lmax])+lmin
            pitch.append(self.sr/pklag)

        return pitch


class HistoPitch(object):
    """This class implements Mike Seltzer's histogram-based pitch tracker.

    Since the work is unpublished, please contact Dr. Richard Stern for the
    original manuscript.
    """

    def __init__(self, fbank, sr, wind, hop, lpc_order=0, acf_floor=ACF_FLOOR):
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
        lpc_order: int, optional
            Option to extract LPC error signal and extract pitch from there.
            Default to 0, which means no LPC.
        acf_floor: float, optional
            Floor below which the normalized autocorrelation function value
            will be deemed 0. This value is determined empirically, although
            it's vital in the peak detection stage.
            Default to ACF_FLOOR.

        See Also
        --------
        sig.fbanks, sig.stproc, sig.window

        """
        super(HistoPitch, self).__init__()
        self.sr = sr
        self.fbank = fbank
        self.numframes = lambda sig: numframes(sig, wind, hop, center=True)
        self.laglen = len(wind)
        self.acf_floor = acf_floor

        def __stacf(sig):
            """Proceed ACF with LPC."""
            frames = stana(sig, wind, hop, center=True)
            out = np.empty_like(frames)
            if not lpc_order:
                for ii, frame in enumerate(frames):
                    out[ii] = xcorr(frame, norm=True)
                return out
            for ii, frame in enumerate(frames):
                alphas, gain = lpc_parcor(frame, lpc_order)
                xerr = lpcerr(frame, alphas, gain=gain)
                out[ii] = xcorr(xerr, norm=True)
            return out

        self.stacf = __stacf

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

        def __lowpass(sig):
            """Lowpass filter each bandpass signal."""
            return lfilter(b, a, sig)

        self.lowpass = __lowpass

    def t0hist(self, sig, fmin=40, fmax=400, clipmode=None, clipratio=0):
        """Compute the fundamental period histogram.

        Parameters
        ----------
        sig: array_like
            Signal to be analyzed.
        fmin: float, 40
            Minimum F0 in Hz.
        fmax: float, 400
            Maximum F0 in Hz.
        clipmode: str, None
            Optional center clipping before ACF calculation.
            Available options are:
            * 'center': Normal center clipping
            * '3level': 3-level center clipping
        clipratio: float, 0
            Center clipping ratio of the maximum absolute amplitude.
            This parameter is only used when clipmode is not None.

        """
        # Step 0.5: Lowpass filter the signal
        sig = self.lowpass(sig)

        # Step 1: bandpass filter signal
        bpsig = (self.fbank.filter(sig, k, cascade=False)
                 for k in range(len(self.fbank)))

        # Step 2: autocorrelation, peak detection, and T0 decision
        if clipmode is None:
            bpacf = (self.stacf(sig) for sig in bpsig)
        elif clipmode == '3level':
            bpacf = (self.stacf(clipcenter3lvl(
                sig, clipratio*np.abs(sig).max())) for sig in bpsig)
        elif clipmode == 'center':
            bpacf = (self.stacf(clipcenter(
                sig, clipratio*np.abs(sig).max())) for sig in bpsig)
        else:
            raise ValueError(f"Unsupported clipmode: [{clipmode}].")
        lmin = int(self.sr/fmax) if fmax else None
        lmax = math.ceil(self.sr/fmin) if fmin else None
        bpt0 = (self.findt0(acf, lmin=lmin, lmax=lmax,
                            acf_floor=self.acf_floor) for acf in bpacf)

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
        """
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
        """

        return uv, pitch

    def pitchcontour2(self, t0hist, voiced, fmin=40, fmax=400, neighbor=.02):
        """Alternative method by making use of the original U/V decision."""
        pitch = []
        ii = 0
        while ii < len(voiced):
            if not voiced[ii]:
                pitch.append(0)
                ii += 1
                continue
            # Process each voiced segment as a whole
            vstart = ii
            vend = ii+1
            while vend < len(voiced) and voiced[vend]:
                vend += 1
            t0path = self.bestpath(t0hist[vstart:vend],
                                   fmin=fmin, fmax=fmax,
                                   neighbor=neighbor)
            pitch.extend(self.sr/t0 if t0 > 0 else 0 for t0 in t0path)
            ii = vend

        return pitch

    @staticmethod
    def runningvar(voiced, pitch):
        """Calculate the running variance of pitch. Presever U/V decisions."""
        var = []
        ii = 0
        while ii < len(voiced):
            if not voiced[ii]:
                var.append(0)
                ii += 1
                continue
            # Process each voiced segment as a whole
            vstart = ii
            vend = ii+1
            while vend < len(voiced) and voiced[vend]:
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
    def findt0(stacf, lmin=0, lmax=None, acf_floor=ACF_FLOOR):
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
            return np.logical_and(
                stsig > acf_floor,
                np.logical_and(stsig > nbleft, stsig > nbright))

        pkloc = findpeaks(stacf)

        # Apply optional mask to control search range
        if 0 < lmin < pkloc.shape[1]:
            pkloc[:, 1:lmin] = False  # preserve peak at lag=0
        if lmax and (lmax+1 < pkloc.shape[1]):
            pkloc[:, (lmax+1):] = False

        numpeaks = pkloc[:, 1:].sum(axis=1)  # excluding lag=0

        t0 = np.empty_like(numpeaks)
        for ii, (nn, pkframe) in enumerate(zip(numpeaks, pkloc)):
            if nn == 0:
                t0[ii] = 0
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
                    t0[ii] = lags[1:][np.argmax(lagvals[1:])]
                else:  # Second peak exists; find max peak after valley
                    t0[ii] = lags[jj:][np.argmax(lagvals[jj:])]

        return t0

    @staticmethod
    def findt0_v2(stacf, lmin=0, lmax=None, acf_floor=ACF_FLOOR):
        """Alternative method to find fundamental period."""

        def findpeaks(sig, imin=0, imax=None, floor=acf_floor):
            """Find peak location and ampitude of a signal."""
            ii = 1 if imin <= 1 else imin
            jj = len(sig)-1 if not imax else imax+1
            seg = sig[ii:jj]
            nbl, nbr = sig[ii-1:jj-1], sig[ii+1:jj+1]
            pkmask = np.logical_and(seg > nbl, seg > nbr)
            if floor is not None:
                pkmask = np.logical_and(pkmask, seg > floor)
            if np.any(pkmask):
                return np.nonzero(pkmask)[0]+ii, seg[pkmask]
            else:
                return (), ()

        def pk2t0(pkloc, pkamp):
            """Find true T0 candidate given local maxima."""
            if len(pkloc) == 0:
                return 0
            elif len(pkloc) == 1:
                return pkloc[0]
            elif len(pkloc) == 2:
                return pkloc[0] if pkamp[0] > pkamp[1] else pkloc[1]
            else:  # Find peak in peak contour
                pvloc, pvamp = findpeaks(-pkamp, floor=None)
                if len(pvloc) == 0:  # No valley; use first max peak
                    return pkloc[np.argmax(pkamp)]
                else:  # Second peak exists; find max peak after valley
                    vv = pvloc[0]
                    return pkloc[vv+np.argmax(pkamp[vv:])]

        t0 = np.array([pk2t0(*findpeaks(acf)) for acf in stacf])

        return t0


class PraatPitch(object):
    """Pitch dection used in Praat.

    Boersma, Paul. “Accurate Short-Term Analysis of the Fundamental Frequency
    and the Harmonics-to-Noise Ratio of a Sampled Sound.”
    In Proc. Institute of Phonetic Sciences, 17:97–110, 1993.
    """
    def __init__(self, sr, timestep=0.01, n_candidates=5,
                 min_pitch=75, max_pitch=600,
                 voicing_threshold=0.45, silence_threshold=0.03,
                 octave_cost=.01, voiced_unvoiced_cost=.14, octave_jump_cost=.35,
                 ):
        """Instantiate a PraatPitch class.

        Parameters
        ----------
        sr: int
            Sampling rate.

        Keyword Parameters
        ------------------
        timestep: float, 0.01
            Duration between frames; 1/timestep is the frame rate.
        n_candidates: int, 4
            Number of pitch candidates per frame, including the unvoiced.
        min_pitch: float, 75
            Minimum pitch to be detected in Hz.
            Window length is set to be 3x or 6x (for HNR) max period.

        """
        assert min_pitch < max_pitch < (sr / 2), "Wrong pitch parameters!"

        self.sr = sr
        self.timestep = timestep

        self.window = self.make_window(3 / min_pitch, 'hanning')
        self.rww = self.xcorr(self.window)[0]
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.n_candidates = n_candidates

        self.voicing_threshold = voicing_threshold
        self.silence_threshold = silence_threshold
        self.octave_cost = octave_cost
        self.voiced_unvoiced_cost = voiced_unvoiced_cost
        self.octave_jump_cost = octave_jump_cost

    def xcorr(self, frames):
        """Normalized autocorrelation."""
        if len(frames.shape) == 1:
            frames = frames[None, ...]
        length = nextpow2(math.ceil(frames.shape[-1] * 1.5))
        if length > frames.shape[-1]:
            zp = np.zeros((len(frames), length - frames.shape[-1]))
            frames = np.concatenate((frames, zp), axis=1)
        spec = fft.fft(frames)
        corr = fft.ifft(spec.real**2 + spec.imag**2).real
        return corr / corr[:, 0][..., None]

    def make_window(self, duration, window_type='gaussian'):
        if window_type == 'gaussian':
            ts = np.linspace(
                -duration / 2, 3*duration / 2, int(duration * self.sr),
                endpoint=False
            )
            return (np.exp(-12 * (ts / duration - .5)**2) - np.exp(-12))\
                / (1 - np.exp(-12))
        elif window_type == 'hanning':
            ts = np.linspace(
                0, duration, int(duration * self.sr),
                endpoint=False
            )
            return 1/2 - 1/2 * np.cos(2*np.pi * ts / duration)
        else:
            raise ValueError(f'Invalid window type: {window_type}')

    def stana(self, sig):
        frames = stana(sig, self.window, int(self.sr*self.timestep),
                       apply_window=False)
        frames -= frames.mean(1)[..., None]
        return frames * self.window

    def lag_strength(self, frame, indices):
        """Estimate precise lag and strength given peak indices.

        NOTE: Do parabolic interpolation for now.
        """
        r = frame[indices]
        r_m1 = frame[indices - 1]
        r_p1 = frame[indices + 1]
        tau_max = 1 / self.sr * (indices + 0.5 * (r_p1 - r_m1)\
                             / (2*r - r_m1 - r_p1))
        r_max = r + (r_p1 - r_m1)**2 / (8 * (2*r - r_m1 - r_p1))
        return tau_max, r_max

    def find_peaks(self, frames):
        """Find fixed number of local maxima within candidate range."""
        minlag = int(self.sr // self.max_pitch - 1)
        maxlag = int(self.sr // self.min_pitch + 1)  # exclusive
        assert 0 < minlag < maxlag <= frames.shape[-1]
        out = []  # out[i] = [(lag [s], value), ...] of frame i
        is_local_maxima = np.logical_and(
            frames[:, minlag:maxlag] > frames[:, minlag-1:maxlag-1],
            frames[:, minlag:maxlag] > frames[:, minlag+1:maxlag+1]
        )
        for frame, maxima in zip(frames, is_local_maxima):
            maxima_index = np.argwhere(maxima)[:, 0] + minlag
            lags, strengths = self.lag_strength(frame, maxima_index)
            idx = np.argsort(strengths)[::-1][:self.n_candidates-1]
            out.append((lags[idx], strengths[idx]))

        # Decide voicing
        global_peak = max([vv[0] if len(vv) else 0 for _, vv in out])
        voiced = []
        for _, vv in out:
            if not len(vv):
                voiced.append(False)
                continue
            voiced.append(
                (vv[0] > self.voicing_threshold) \
                and (vv[0] >= self.silence_threshold * global_peak)
            )

        return global_peak, out, voiced

    def transition_cost(self, f1, f2):
        if f1 == f2 == 0:
            return 0
        if (f1 == 0) ^ (f2 == 0):
            return self.voiced_unvoiced_cost

        return self.octave_jump_cost * np.abs(np.log2(f1 / f2))

    def vad(self, local_peaks, global_peak):
        frames = self.stana(sig)
        corr = self.xcorr(frames) / self.rww
        _, _, voicing = self.find_peaks(corr)
        return voicing

    def find_path(self, frequencies, strengths):
        """Shortest path finding using dynamic programming."""
        def _min_pitch(freqs, costs):
            return min(zip(costs, freqs))[1]

        nframes = len(strengths)
        if nframes == 1:
            return [_min_pitch(
                frequencies[0], [-s for s in strengths[0]])
            ]

        costs = deque([[-s for s in strengths[0]]])
        best_child = deque([[None] * len(frequencies[0])])  # for backtracing
        for ii in range(1, nframes):
            f_m1 = frequencies[ii-1]
            c_m1 = costs[ii-1]
            costs.append([])
            best_child.append([])
            for jj, ff in enumerate(frequencies[ii]):
                ss = strengths[ii][jj]
                cc = [
                    cp + self.transition_cost(
                        fp, ff) - ss for fp, cp in zip(f_m1, c_m1)
                ]
                min_cc, min_freq_idx = min(zip(cc, range(len(cc))))
                costs[ii].append(min_cc)
                best_child[ii].append(min_freq_idx)

        _ , freq_idx = min(zip(costs[-1], range(len(costs[-1]))))
        pitches = [frequencies.pop()[freq_idx]]
        while True:
            freq_idx = best_child.pop()[freq_idx]
            if freq_idx is None:
                break
            pitches.append(frequencies.pop()[freq_idx])

        return pitches[::-1]

    def __call__(self, sig):
        frames = self.stana(sig)
        corr = self.xcorr(frames)
        corr[:, :len(self.window//2+1)] /= self.rww[:len(self.window//2+1)]
        global_peak, candidates, voicing = self.find_peaks(corr)

        # Compute strengths
        strengths = deque([])
        frequencies = deque([])
        for lag, val in candidates:
            if not len(val):  # no peak
                strengths.append([self.voicing_threshold + 2])
                frequencies.append([0])
                continue
            unvoiced = self.voicing_threshold + max(
                0, 2 - val[0] / global_peak /\
                (self.silence_threshold / (1 + self.voicing_threshold))
            )
            voiced = val - self.octave_cost * np.log2(self.min_pitch * lag)
            strengths.append([unvoiced] + voiced.tolist())
            frequencies.append([0] + (1 / lag).tolist())

        pitch = self.find_path(frequencies, strengths)

        return pitch
