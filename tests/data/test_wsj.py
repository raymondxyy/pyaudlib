import os
import pytest

import audlib
from audlib.data.wsj import WSJ0, WSJ1, ASRWSJ0, ASRWSJ1
from audlib.asr.util import PhonemeMap


pmap = PhonemeMap(os.path.dirname(audlib.__file__)+"/misc/cmudict-0.7b")


@pytest.mark.skipif('WSJ0_ROOT' not in os.environ,
                    reason='ENV $WSJ0_ROOT unspecified.')
def test_wsj0():
    train = WSJ0(os.environ['WSJ0_ROOT'])
    test = WSJ0(os.environ['WSJ0_ROOT'], train=False)
    assert len(train) == 43177
    assert len(test) == 4122
    print(train, test)
    test_asr = ASRWSJ0(test, pmap)
    print(test_asr)


@pytest.mark.skipif('WSJ1_ROOT' not in os.environ,
                    reason='ENV $WSJ1_ROOT unspecified.')
def test_wsj1():
    train = WSJ1(os.environ['WSJ1_ROOT'])
    test = WSJ1(os.environ['WSJ1_ROOT'], train=False)
    assert len(train) == 30278
    assert len(test) == 503
    print(train, test)
    test_asr = ASRWSJ1(test, pmap)
    print(test_asr)
