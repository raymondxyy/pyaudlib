import os

import audlib
from audlib.asr.util import PhonemeMap, CharacterMap

pmap = PhonemeMap(os.path.dirname(audlib.__file__)+"/misc/cmudict-0.7b")
print("Dictionary contains [{}] valid words.".format(len(pmap.dict)))
print("[{}] valid phonemes.".format(len(pmap.phonedict)))

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
    len(cmap.chardict)))

# Test a random sentence
sentence = "This is a sentence to be processed by SPHINX"
if not cmap.transcribable(sentence):
    print("OOV in [{}]: [{}]".format(sentence, cmap.trans2oov(sentence)))
else:
    print("[{}] ---> [{}]".format(sentence, cmap.trans2vocab(sentence)))
    print("[{}] ---> [{}]".format(sentence, cmap.trans2label(sentence)))