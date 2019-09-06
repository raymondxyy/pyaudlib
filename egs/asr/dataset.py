"""Dataset preparation script.

Put data-related change here. Main functions in `airbus` will run import
variables in here.
"""
import os

import numpy as np

from audlib.asr.util import CharacterMap
from audlib.data.wsj import ASRWSJ0, WSJ0
from audlib.data.dataset import Subset
from audlib.nn.transform import Compose

from transforms import Melspec, FinalTransform

# Global variables
WSJ0_ROOT = '/home/xyy/data/wsj0'
CHARS = "&*ABCDEFGHIJKLMNOPQRSTUVWXYZ #'-/@_"  # All allowed vocabularies
REPLMAP = {  # special WSJ transcripts to be replaced
           "[LOUD_BREATH]": "",
           "[LOUD_BEATH]": "",
           "[LIP_SMACK]": "",
           "[TONGUE_CLICK]": "",
           "[<DOOR_SLAM]": "",
           "[MISC_NOISE>]": "",
           "[DOOR_SLAM]": "",
           "[DOOR_SLAM>]": "",
           "[EXHALATION]": "",
           "[LOUD-BREATH]": "",
           "[MICROPHONE_MVT]": "",
           "[MICROPHONE_MVT>]": "",
           "[<MICROPHONE_MVT>]": "",
           "[/MICROPHONE_MVT]": "",
           "[MICROPHONE_MVT/]": "",
           "[CHAIR_SQUEAK>]": "",
           "[<CHAIR_SQUEAK]": "",
           "[<TAP]": "",
           "[TAP>]": "",
           "[<TAP>]": "",
           "[DOOR_OPEN/]": "",
           "[/DOOR_OPEN]": "",
           "[DOOR_OPEN>]": "",
           "[BEEP/]": "",
           "[/BEEP]": "",
           "[BEEP>]": "",
           "[BEEP]": "",
           "[SIGH]": "",
           "[CROSS_TALK]": "",
           "[CROSS_TALK/]": "",
           "[/CROSS_TALK]": "",
           "[CROSS_TALK>]": "",
           "[TAP]": "",
           "[THROAT_CLEAR]": "",
           "[UNINTELLIGIBLE]": "",
           "[DISK_NOISE/]": "",
           "[/DISK_NOISE]": "",
           "[DISK_NOISE]": "",
           "[CHAIR_SQUEAK]": "",
           "[MISC_NOISE]": "",
           "[MISC_NOISE/]": "",
           "[/MISC_NOISE]": "",
           "[PHONE_RING/]": "",
           "[PHONE_RING]": "",
           "[<PHONE_RING]": "",
           "[/PHONE_RING]": "",
           "[PAPER_RUSTLE/]": "",
           "[PAPER_RUSTLE]": "",
           "[<BEEP]": "",
           "[MOVEMENT/]": "",
           "[/MOVEMENT]": "",
           "[MOVEMENT]": "",
           "[TYPING/]": "",
           "[/TYPING]": "",
           "[<THUMP]": "",
           "[SNIFF]": "",
           "[UH]": "",
           "[UM]": "",
           "[LAUGHTER/]": "",
           "[/LAUGHTER]": "",
           "\\?QUESTION\\-MAR(K)-": "QUESTION MAR",
           "FORECAS(TS)-": "FORECAS",
           "\\.PER(IOD)-": "",
           "-(PER)IOD": "IOD",
           "FL(OW)-": "FL",
           "D(EPORTATIONS)-": "",  # checked
           "F(IVE)-": "",
           "REVO(LUTIONARY)-": "",
           "EC(CONOMISTS)-": "",
           "FINAN(CIALLY)-": "",
           "AN(NOUNCED)-": "",
           "PRIST(INE)-": "",
           "W(ORLD)-": "",
           "\\~NEW\\-GRAPH": "NEW GRAPH",
           "\\/SLASH": "SLASH",
           '\\"OPEN\\-QUOTES': 'OPEN QUOTES',
           '\\"CLOSE\\-QUOTES': 'CLOSE QUOTES',
           '\\"OPEN\\-QUOTE': 'OPEN QUOTE',
           '\\"CLOSE\\-QUOTE': 'CLOSE QUOTE',
           '\\"QUOTES': 'QUOTES',
           '\\"QUOTE': 'QUOTE',
           '\\"UNQUOTE': 'UNQUOTE',
           '\\"END\\-QUOTE': 'END QUOTE',
           "\\!EXCLAMATION\\-POINT": "EXCLAMATION POINT",
           "\\!EXCLAMATION-POINT": "EXCLAMATION POINT",
           "\\.PERIOD": "PERIOD",
           "\\,COMMA": "COMMA",
           "\\%PERCENT": "PERCENT",
           "\\-HYPHEN": "HYPHEN",
           '\\"DOUBLE\\-QUOTE': 'DOUBLE QUOTE',
           '\\"DOUBLE-QUOTE': "DOUBLE QUOTE",
           "\\?QUESTION\\-MARK": "QUESTION MARK",
           "\\?QUESTION-MARK": "QUESTION MARK",
           "\\-\\-DASH": "DASH",
           "\\&AMPERSAND": "AMPERSAND",
           "\\:COLON": "COLON",
           "\\;SEMI\\-COLON": "SEMI COLON",
           "\\;SEMI-COLON": "SEMI COLON",
           "\\(LEFT\\-PAREN": "LEFT PAREN",
           "\\)RIGHT\\-PAREN": "RIGHT PAREN",
           "\\(LEFT-PAREN": "LEFT PAREN",
           "\\)RIGHT-PAREN": "RIGHT PAREN",
           "\\(PARENTHESES": "PARENTHESE",
           "\\)CLOSE\\-PARENTHESES": "CLOSE PARENTHESE",
           "\\(OPEN\\-PAREN": "OPEN PAREN",
           "\\)CLOSE\\-PAREN": "CLOSE PAREN",
           "\\\'SINGLE\\-QUOTE": "SINGLE QUOTE",
           "\\{LEFT\\-BRACE": "LEFT BRACE",
           "\\{LEFT-BRACE": "LEFT BRACE",
           "\\}RIGHT\\-BRACE": "RIGHT BRACE",
           "\\}RIGHT-BRACE": "RIGHT BRACE",
           "\\$DOLLAR\\-SIGN": "DOLLAR SIGN",
           "\\\'": "",  # as in they\'ve
           "\\.": "",  # as in Mr\.
           '\.': '',  # as in U\. S\.
           "\\'": "",  # as in can\'t
           "\\`": "",  # REPUBLICBANK\\`S
           ":": "",  # M:ONEY
           ".": "",  # BEACH . CALIFORNIA
           "~": "",
           "*": "",  # *AMORTIZE*
           "<D>": "",
           "<A>": "",
           "!": ""
           }
FEATDIM = 40  # Feature dimension

# Global variables that are not set by user
assert os.path.exists(WSJ0_ROOT), "Invalid WSJ path -- Configure [WSJ0_ROOT]!"
_signal_transform = Melspec(16000, nmel=FEATDIM)
CHARMAP = CharacterMap(CHARS, replacemap=REPLMAP)
print(CHARMAP)
_train_transforms = Compose([_signal_transform,
                             FinalTransform(CHARMAP, '&', '*', train=True)])
_test_transforms = Compose([_signal_transform,
                            FinalTransform(CHARMAP, '&', '*', train=False)])


def _filt(path): return path.endswith(".wv1")  # only process .wv1


_wsj_train = ASRWSJ0(WSJ0(WSJ0_ROOT, train=True, filt=_filt,
                          transform=_train_transforms),
                     CHARMAP, verbose=True)
WSJ_TEST = ASRWSJ0(WSJ0(WSJ0_ROOT, train=False, filt=_filt,
                        transform=_test_transforms),
                   CHARMAP, verbose=True)
print(_wsj_train, WSJ_TEST)
_num_train = int(len(_wsj_train)*.9)  # split training set

# These are the three actual datasets passed into NNs
print("Using [{}] for training, [{}] for validation.".format(
    _num_train, len(_wsj_train)-_num_train))
WSJ_TRAIN = Subset(_wsj_train, range(_num_train))
WSJ_VALID = Subset(_wsj_train, range(_num_train, len(_wsj_train)))

print("Accumulating vocabulary histogram.")
VOCAB_HIST = np.asarray(_wsj_train.vocab_hist)
for ii, cnt in enumerate(VOCAB_HIST):
    print("""{}: {}""".format(_wsj_train.transmap.labeldict[ii], cnt))


if __name__ == "__main__":
    for ii, (feat, inseq, target) in enumerate(WSJ_TRAIN):
        print(feat.shape)
