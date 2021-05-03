import os
import timeit

import audlib
from audlib.asr.util import PhonemeMap, CharacterMap, levenshtein


def test_levenshtein():
    """Test edit distance."""
    assert levenshtein('', '')[0] == (0, 0, 0)
    assert levenshtein('aaa', '')[0] == (3, 0, 0)
    assert levenshtein('a', 'aaaaa')[0] == (0, 4, 0)
    assert levenshtein('abba', 'abba')[0] == (0, 0, 0)

    #  | Source  -- Target
    #  [Insert]; -Delete-; \ Substitute or correct (N)
    #     _     B      A     B     A
    #  _  0,N*  1,IB*  2,IA  3,IB  4,IA
    #  A  1,DA  1,SAB  1,N*  2,IB  3,IA
    #  B  2,DB  1,N    2,DB* 1,N   2,IA
    #  B  3,DB  2,DB   2,SBA 2,N*  2,SBA
    #  A  4,DA  3,DA   2,N   3,DA  2,N*  <- ([B]A-B-BA)
    costs, _ = levenshtein('abba', 'baba')
    assert costs == (1, 1, 0)

    costs, edits = levenshtein('kitten', 'sitting')
    for ee in edits:
        print(ee)

def test_PhonemeMap():
    pmap = PhonemeMap(os.path.dirname(audlib.__file__)+"/misc/cmudict-0.7b")
    print("Dictionary contains [{}] valid words.".format(len(pmap.dict)))
    print("[{}] valid phonemes.".format(len(pmap)))

    # Test a random sentence
    sentence = "This is a sentence to be processed by SPHINX"
    if not pmap.transcribable(sentence):
        print("OOV in [{}]: [{}]".format(sentence, pmap.trans2oov(sentence)))
    else:
        print("[{}] ---> [{}]".format(sentence, pmap.trans2vocab(sentence)))
        print("[{}] ---> [{}]".format(sentence, pmap.trans2label(sentence)))

def test_CharacterMap():
    # Test character map
    cmap = CharacterMap("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    print("Dictionary contains [{}] valid characters.".format(
        len(cmap)))

    # Test a random sentence
    sentence = "This is a sentence to be processed by SPHINX"
    if not cmap.transcribable(sentence):
        print("OOV in [{}]: [{}]".format(sentence, cmap.trans2oov(sentence)))
    else:
        print("[{}] ---> [{}]".format(sentence, cmap.trans2vocab(sentence)))
        print("[{}] ---> [{}]".format(sentence, cmap.trans2label(sentence)))

if __name__ == '__main__':
    test_levenshtein()
    #test_PhonemeMap()
    #test_CharacterMap()
