"""Pre-load some examples to quickstart an experiment."""
import os
from .io.audio import audioread


def welcome():
    """Load Rich Stern's 'Welcome to DSP 1' utterance."""
    path = os.path.abspath(
        os.path.dirname(__file__)+'/../samples/welcome16k.wav')
    assert os.path.exists(path), "Example audio not available!"
    return audioread(path)


def babble():
    """Load a segment of babble noise sample in Loizou's NOIZEUS dataset."""
    path = os.path.abspath(
        os.path.dirname(__file__)+'/../samples/babble16k.wav')
    assert os.path.exists(path), "Example audio not available!"
    return audioread(path)
