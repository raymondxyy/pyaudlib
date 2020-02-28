"""Data encapsulation suitable for batch processing."""


class Audio(object):
    """A class for any processing that requires only signal and sr."""
    __slots__ = 'signal', 'samplerate'

    def __init__(self, signal=None, samplerate=None):
        self.signal = signal
        self.samplerate = samplerate


class GenericAudio(Audio):
    """For those data structure that needs extra fields."""
    __slots__ = '__dict__'


class AudioSpeaker(Audio):
    """A class useful for speaker identification tasks."""
    __slots__ = 'speaker'

    def __init__(self, signal=None, samplerate=None, speaker=None):
        super(AudioSpeaker, self).__init__(signal, samplerate)
        self.speaker = speaker


class AudioPitch(Audio):
    """A class useful for pitch extraction tasks."""
    __slots__ = 'pitch', 'egg'

    def __init__(self, signal=None, samplerate=None, pitch=None, egg=None):
        super(AudioPitch, self).__init__(signal, samplerate)
        self.pitch = pitch
        self.egg = egg


class NoisySpeech(object):
    """Data structure for noisy speech resulted from (additive) noise."""
    __slots__ = 'noisy', 'clean', 'noise', 'snr', 'vad'

    def __init__(self, noisy=None, clean=None, noise=None, snr=None, vad=None):
        self.noisy = noisy
        self.clean = clean
        self.noise = noise
        self.snr = snr
        self.vad = vad


class SpeechTranscript(Audio):
    """Data structure for automatic speech recognition."""
    __slots__ = 'transcript', 'label'

    def __init__(self, signal=None, samplerate=None,
                 transcript=None, label=None):
        super(SpeechTranscript, self).__init__(signal, samplerate)
        self.transcript = transcript
        self.label = label
