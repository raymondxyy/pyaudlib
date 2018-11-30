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

    def transcribable(self, transcript):
        """Determine if transcript is transcribable."""
        return len(self.trans2oov(transcript)) == 0


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

        self.phonedict = {p: ii for ii, p in enumerate(sorted(pset))}
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
            out.extend([self.phonedict[p] for p in self.dict[word]])
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


class CharacterMap(TranscriptMap):
    """Construct a character map for speech rec."""

    def __init__(self, charset, replacemap=None):
        """Construct a character map from a character set.

        Optionally, take a replacement dictionary in which some particular
        strings are mapped to another string. This is useful for removing
        special characters and replacing special strings such as ?QUESTIONMARK.
        """
        self.chardict = {c.upper(): ii for ii, c in enumerate(sorted(charset))}
        self.replacedict = None
        if replacemap is not None:
            self.replacedict = replacemap

    def trans2label(self, transcript):
        """Convert a transcript string to a sequence of labels."""
        out = []
        for c in self._replace(transcript.upper()):
            out.append(self.chardict[c])
        return out

    def trans2vocab(self, transcript):
        """Convert a transcript string to a sequence of characters."""
        out = []
        for c in self._replace(transcript.upper()):
            out.append(c)
        return out

    def trans2oov(self, transcript):
        """Convert a transcript string to a list of oov chars."""
        out = []
        for c in self._replace(transcript.upper()):
            if c not in self.chardict:
                out.append(c)
        return {w: out.count(w) for w in set(out)}

    def _replace(self, transcript):
        """Replace a transcript using `replacedict`."""
        if self.replacedict is None:
            return transcript
        for orig, repl in self.replacedict.items():
            transcript = transcript.replace(orig.upper(), repl.upper())
        return transcript


if __name__ == '__main__':
    from pprint import pprint
    pmap = PhonemeMap('/home/xyy/local/tutorial/wsj0/etc/cmudict.0.6d',
                      phonepath='/home/xyy/local/tutorial/wsj0/etc/wsj0.phone')
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
