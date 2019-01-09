"""Utilities for speech recognition."""


class TranscriptMap(object):
    """An abstract class representing a map from transcripts to labels.

    All other datasets should subclass it. All subclasses should override
    ``trans2label``.
    """

    def trans2label(self, transcript):
        """Convert a transcript to (a sequence of) labels."""
        raise NotImplementedError

    def trans2oov(self, transcript):
        """Convert a transcript to out-of-vocabulary words."""
        raise NotImplementedError

    def trans2vocab(self, transcript):
        """Convert a transcript to a list of vocabularies."""
        raise NotImplementedError

    def label2vocab(self, labels):
        """Convert a label sequence to a vocabulary sequence."""
        raise NotImplementedError

    def vocab2trans(self, vocabs):
        """Convert a vocabulary sequence to a string."""
        raise NotImplementedError

    def transcribable(self, transcript):
        """Determine if transcript is transcribable."""
        return len(self.trans2oov(transcript)) == 0

    def __len__(self):
        """Return number of vocabularies in the map."""
        raise NotImplementedError

    def __str__(self):
        """Print transcript map info."""
        report = """
            [{}]: Total [{}] available vocabularies.
        """.format(self.__class__.__name__,
                   len(self))
        return report


class PhonemeMap(TranscriptMap):
    """Construct a phoneme set, filler set, and a dictionary for speech rec."""

    def __init__(self, dictpath, phonepath=None, fillerpath=None,
                 replacemap=None):
        """Construct a phoneme (and filler) map from a dictionary.

        Optionally, if `phonepath` and `fillerpath` are predefined, use them
        as the default map and filter out-of-vocabulary words in dictionary.
        """
        if phonepath is not None:  # use user-defined phone list
            pset = self.load_phoneset(phonepath)
            if fillerpath is not None:
                pset.union(self.load_fillerset(fillerpath))
            # Construct dictionary
            self.dict, _ = self.load_dictionary(dictpath, phoneset=pset)
        else:  # infer phone list from dictionary
            self.dict, pset = self.load_dictionary(dictpath, phoneset=None)

        self.vocabdict = {p: ii for ii, p in enumerate(sorted(pset))}
        self.replacedict = replacemap

    def load_phoneset(self, path):
        """Load phoneme set from `path`.

        Assume the file has the following format:
        <phone 1>
        <phone 2>
        ...
        <phone N>
        """
        plist = []
        with open(path) as fp:
            for line in fp.readlines():
                plist.append(line.strip())
        return set(plist)

    def load_fillerset(self, path):
        """Load filler set from `path`.

        Assume the file has the following format:
        <filler 1>
        <filler 2>
        ...
        <filler N>
        """
        return self.load_phoneset(path)

    def load_dictionary(self, path, phoneset=None):
        """Load dictionary from `path`.

        Assume the file has the following format:
        <word 1>    <phone 1> <phone 2> ...
        <word 2>    <phone 1> <phone 2> ...
        ...         ...
        <word N>    <phone 1> <phone 2> ...

        Returns the dictionary and the phoneme set.
        """
        d = {}
        if phoneset is None:  # construct pset on-the-fly
            pset = set([])
        else:  # assume existing pset; returns nothing
            pset = None
        with open(path) as fp:
            for line in fp.readlines():
                line = line.strip().split()
                word, plist = line[0].upper(), line[1:]
                assert word not in d, "Redefinition of word: [{}]".format(d)
                if phoneset is None:  # no default set - construct one
                    d[word] = plist
                    pset.update(set(plist))
                else:  # filter out OOV words
                    oov = False
                    for p in plist:
                        if p not in phoneset:
                            oov = True
                            break
                    if not oov:
                        d[word] = plist

        return d, pset

    def validate_phone(self, vocabset, dictionary):
        """Validate the vocabulary set and dictionary.

        This should be called upon loading a phoneset and dictionary.
        """
        for word, vlist in dictionary.items():
            for v in vlist:
                assert v in vocabset, "[{}] of [{}] not in vocabset!".format(
                    v, word)

    def trans2label(self, transcript):
        """Convert a transcript string to phoneme indices.

        Assuming all CAPITALIZED words in transcript exist in dictionary.
        """
        out = []
        for word in self._replace(transcript.upper()).split():
            out.extend([self.vocabdict[p] for p in self.dict[word]])
        return out

    def trans2oov(self, transcript):
        """Convert a transcript string to a list of out-of-vocabulary words.

        Assuming dictionary words are CAPITALIZED.
        """
        out = []
        for word in self._replace(transcript.upper()).split():
            if word.upper() not in self.dict:
                out.append(word)
        return {w: out.count(w) for w in set(out)}

    def trans2vocab(self, transcript):
        """Convert a transcript string to a list of phonemes.

        Assuming dictionary words are CAPITALIZED.
        """
        out = []
        for word in self._replace(transcript.upper()).split():
            out.extend(self.dict[word.upper()])
        return out

    def _replace(self, transcript):
        """Replace a transcript using `replacedict`."""
        if self.replacedict is None:
            return transcript
        for orig, repl in self.replacedict.items():
            transcript = transcript.replace(orig.upper(), repl.upper())
        return transcript

    def __len__(self):
        """Return number of phones in the map."""
        return len(self.vocabdict)


class CharacterMap(TranscriptMap):
    """Construct a character map for speech rec."""

    def __init__(self, charset, replacemap=None):
        """Construct a character map from a character set.

        Optionally, take a replacement dictionary in which some particular
        strings are mapped to another string. This is useful for removing
        special characters and replacing special strings such as ?QUESTIONMARK.
        """
        sorted_char = sorted(charset)
        self.vocabdict = {c.upper(): ii for ii, c in enumerate(sorted_char)}
        self.labeldict = {ii: c.upper() for ii, c in enumerate(sorted_char)}
        self.replacedict = None
        if replacemap is not None:
            self.replacedict = replacemap

    def trans2label(self, transcript):
        """Convert a transcript string to a sequence of labels."""
        out = []
        for c in self._replace(transcript.upper()):
            out.append(self.vocabdict[c])
        return out

    def trans2vocab(self, transcript):
        """Convert a transcript string to a sequence of characters."""
        out = []
        for c in self._replace(transcript.upper()):
            out.append(c)
        return out

    def label2vocab(self, labels):
        """Convert a label sequence to character sequence."""
        return [self.labeldict[l] for l in labels]

    def vocab2trans(self, vocab):
        """Convert a sequence of characters to a string."""
        return ''.join(vocab)

    def trans2oov(self, transcript):
        """Convert a transcript string to a list of oov chars."""
        out = []
        for c in self._replace(transcript.upper()):
            if c not in self.vocabdict:
                out.append(c)
        return {w: out.count(w) for w in set(out)}

    def _replace(self, transcript):
        """Replace a transcript using `replacedict`."""
        if self.replacedict is None:
            return transcript
        for orig, repl in self.replacedict.items():
            transcript = transcript.replace(orig.upper(), repl.upper())
        return transcript

    def __len__(self):
        """Return number of characters in the map."""
        return len(self.vocabdict)


def levenshtein(src, tar):
    """Compute the Levenshtein (edit) distance between two strings.

    The edit distance is the number of steps taken to transform `src` to
    `tar`. Break tie by preferring deletion, then insertion, and finally
    substitution.

    Parameters
    ----------
    src: str
        src string.
    tar: str
        tar string.

    Returns
    -------
    type: tuple of 4 int
    Total distance, number of deletions, insertions, and substitutions.

    """
    ss = len(src)
    tt = len(tar)
    if ss == 0:
        return tt, 0, tt, 0
    if tt == 0:
        return ss, ss, 0, 0

    UP, LEFT, UPLEFT = 'DIS'  # order is important for tiebreak
    costs = []  # holds cached cost and parent node at each (i,j)
    for ii in range(ss+1):
        if ii == 0:
            costs.append([(0, None)] + [(i, LEFT) for i in range(1, tt+1)])
            continue
        costs.append([])
        for jj in range(tt+1):
            if jj == 0:
                costs[ii].append((ii, UP))
                continue
            cost = 0 if (src[ii-1] == tar[jj-1]) else 1
            up = jj + 1 if ii-1 < 0 else costs[ii-1][jj][0]
            left = ii + 1 if jj-1 < 0 else costs[ii][jj-1][0]
            if ii-1 < 0 and jj-1 < 0:
                upleft = 0
            elif ii-1 < 0:
                upleft = jj
            elif jj-1 < 0:
                upleft = ii
            else:
                upleft = costs[ii-1][jj-1][0]

            # break tie by: deletion, insertion, substitution
            costs[ii].append(min(zip((up+1, left+1, upleft+cost),
                                     (UP, LEFT, UPLEFT))))

    # Backtrack to beginning and record types of error along the way
    insert = 0
    delete = 0
    substitute = 0
    ii, jj = ss, tt
    cost, parent = costs[ii][jj]  # start from bottom right
    while parent:
        if parent == UP:
            ii -= 1
            delete += 1
        elif parent == LEFT:
            jj -= 1
            insert += 1
        else:
            ii -= 1
            jj -= 1
            if cost != costs[ii][jj][0]:
                substitute += 1

        cost, parent = costs[ii][jj]

    return costs[-1][-1][0], delete, insert, substitute
