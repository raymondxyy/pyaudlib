from audlib.data.wsj import WSJ0
from audlib.asr.util import PhonemeMap, CharacterMap
from pprint import pprint

pmap = PhonemeMap('/home/xyy/onhold/tutorial/wsj0/etc/cmudict.0.6d',
                  phonepath='/home/xyy/onhold/tutorial/wsj0/etc/wsj0.phone')
print("Dictionary contains [{}] valid words.".format(len(pmap.dict)))
print("[{}] valid phonemes.".format(len(pmap.phonedict)))
pprint(pmap.phonedict)

# Test a random sentence
sentence = "This is a sentence to be processed by SPHINX"
if not pmap.transcribable(sentence):
    pprint("OOV in [{}]: [{}]".format(sentence, pmap.trans2oov(sentence)))
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
    pprint("OOV in [{}]: [{}]".format(sentence, cmap.trans2oov(sentence)))
else:
    print("[{}] ---> [{}]".format(sentence, cmap.trans2vocab(sentence)))
    print("[{}] ---> [{}]".format(sentence, cmap.trans2label(sentence)))
