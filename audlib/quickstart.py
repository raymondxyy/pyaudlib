"""Pre-load some examples to quickstart an experiment."""
import os


def welcome():
    """Load Rich Stern's 'Welcome to DSP 1' utterance."""
    from .io.audio import audioread
    path = os.path.abspath(
        os.path.dirname(__file__)+'/../samples/welcome16k.wav')
    assert os.path.exists(path), "Example audio not available!"
    return audioread(path)
