"""Pre-load some examples to quickstart an experiment."""
import os
from .io.audio import audioread


def welcome():
    """Load Rich Stern's 'Welcome to DSP 1' utterance."""
    path = os.path.join(os.path.dirname(__file__), 'samples/welcome16k.wav')
    return audioread(path)


def arctic():
    """Load one example from the CMU ARCTIC dataset."""
    path = os.path.join(os.path.dirname(__file__), 'samples/arctic_a0001.wav')
    return audioread(path)
