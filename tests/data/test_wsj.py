import os

import audlib
from audlib.data.wsj import WSJ0, WSJ1, ASRWSJ0, ASRWSJ1
from audlib.asr.util import PhonemeMap

WSJ0_HOME = '/home/xyy/data/wsj0'
WSJ1_HOME = '/home/xyy/data/wsj1'

pmap = PhonemeMap(os.path.dirname(audlib.__file__)+"/misc/cmudict-0.7b")


def test_wsj0():
    train = WSJ0(WSJ0_HOME)
    test = WSJ0(WSJ0_HOME, train=False)
    assert len(train) == 43177
    assert len(test) == 4122
    print(train, test)
    test_asr = ASRWSJ0(test, pmap)
    print(test_asr)


def test_wsj1():
    train = WSJ1(WSJ1_HOME)
    test = WSJ1(WSJ1_HOME, train=False)
    assert len(train) == 30278
    assert len(test) == 503
    print(train, test)
    test_asr = ASRWSJ1(test, pmap)
    print(test_asr)


if __name__ == '__main__':
    test_wsj0()
    test_wsj1()
