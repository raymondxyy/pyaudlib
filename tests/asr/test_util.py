import os
import timeit

import audlib
from audlib.asr.util import PhonemeMap, CharacterMap, levenshtein


def test_levenshtein():
    """Test edit distance."""
    assert levenshtein('', '') == (0, 0, 0, 0)
    assert levenshtein('aaa', '') == (3, 3, 0, 0)
    assert levenshtein('a', 'aaaaa') == (4, 0, 4, 0)
    assert levenshtein('abba', 'abba') == (0, 0, 0, 0)

    #     _    B    A    B    A
    #  _  0,N  1,I  2,I  3,I  4,I
    #  A  1,D  1,S  1,S  2,I  3,I
    #  B  2,D  1,S  2,D  1,S  2,I
    #  B  3,D  2,D  2,S  2,D  2,S
    #  A  4,D  3,D  2,S  3,D  2,S
    assert levenshtein('abba', 'baba') == (2, 1, 1, 0)


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
