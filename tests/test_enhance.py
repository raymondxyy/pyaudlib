"""Test enhancement functions."""
from audlib.quickstart import welcome
from audlib.sig.window import hamming

WELCOME, SR = welcome()
HOP = .25
WIND = hamming(SR*.025, HOP, synth=True)


def test_SSFEnhancer():
    from audlib.enhance import SSFEnhancer
    enhancer = SSFEnhancer(SR, WIND, HOP, 512)
    sigssf = enhancer(WELCOME, .4)  # sounds okay
    return


if __name__ == '__main__':
    test_SSFEnhancer()
