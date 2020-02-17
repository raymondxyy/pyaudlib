# coding: utf-8

"""Dataset class for the AVspoof dataset."""
from .dataset import AudioDataset, audioread
from .datatype import Audio


class SpoofedAudio(Audio):
    """A data structure for spoofed audio.

    attacktype can only be one of the following:
        - None --> genuine audio
        - 'r' --> replay attack
        - 's' --> speech synthesis attack
        - 'c' --> voice conversion attack
    """
    __slots__ = "attacktype"

    def __init__(self, signal=None, samplerate=None, attacktype=None):
        super(SpoofedAudio, self).__init__(signal, samplerate)
        self.attacktype = attacktype


def attack_type(path):
    """Return the attack type of the audio.

    Useful as a filter for obtaining sub-partitions of the dataset.

    Parameters
    ----------
    path: str
        File path.

    Returns
    -------
    atype: str or None
        Attack type if path points to an attack file, or None if genuine.
        Attack types can only be one of the following:
            - replay
            - synthesis
            - conversion

    """
    if 'genuine' in path:
        return None
    elif 'replay' in path:
        return 'r'
    elif 'conversion' in path:
        return 'c'
    elif 'synthesis' in path:
        return 's'
    else:
        raise ValueError(f"Unknown attack type for [{path}]!")


def is_replay(path): return attack_type(path) == 'r'


def is_synthesis(path): return attack_type(path) == 's'


def is_conversion(path): return attack_type(path) == 'c'


def is_genuine(path): return attack_type(path) is None


class AVSpoof(AudioDataset):
    """Class for the AVSpoof dataset.

    The dataset directory should follow the structure below.

    AVSpoof
    ├── attacks
    │   ├── replay_laptop
    │   ├── replay_laptop_HQ_speaker
    │   ├── replay_phone1
    │   ├── replay_phone2
    │   ├── speech_synthesis_logical_access
    │   ├── speech_synthesis_physical_access
    │   ├── speech_synthesis_physical_access_HQ_speaker
    │   ├── voice_conversion_logical_access
    │   ├── voice_conversion_physical_access
    │   └── voice_conversion_physical_access_HQ_speaker
    ├── genuine
    │   ├── female
    │   └── male
    └── README.txt

    Serife Kucur Ergunay, Elie Khoury, Alexandros Lazaridis, Sebastien Marcel.
    "On the vulnerability of Speaker Verification to Realistic Voice Spoofing",
    BTAS 2015.
    """

    @staticmethod
    def isaudio(path):
        return path.endswith('.wav')

    def __init__(self, root, filt=None, read=None, transform=None):
        """Instantiate an AVSpoof dataset.

        Parameters
        ----------
        root: str
            The root directory of AVSpoof.
        filt: callable, optional
            Filters to be applied on each audio path. Default to None.
        read: callable(str) -> (array_like, int), optional
            User-defined ways to read in an audio.
            Returned values are wrapped around an `SpoofedAudio` class.
        transform: callable(SpoofedAudio) -> SpoofedAudio
            User-defined transformation function.

        Returns
        -------
        An class instance `avspoof` that has the following properties:
            - len(avspoof) == number of usable audio samples
            - avspoof[idx] == an Audio instance

        See Also
        --------
        SpoofedAudio, dataset.AudioDataset, datatype.Audio

        """
        def _read(path):
            if not read:
                sig, ssr = audioread(path)
                return SpoofedAudio(sig, ssr, attacktype=attack_type(path))
            else:
                sig, ssr = read(path)
                return SpoofedAudio(sig, ssr, attacktype=attack_type(path))

        super(AVSpoof, self).__init__(
            root, filt=self.isaudio if not filt else lambda p:
                self.isaudio(p) and filt(p),
            read=_read, transform=transform)

    def __repr__(self):
        """Representation of AVSpoof."""
        return r"""{}({}, sr={}, transform={})
        """.format(self.__class__.__name__, self.root, self.sr, self.transform)

    def __str__(self):
        """Print out a summary of instantiated dataset."""
        genuine, replay, conversion, synthesis = 0, 0, 0, 0
        for pp in self._filepaths:
            if attack_type(pp) == 'r':
                replay += 1
            elif attack_type(pp) == 'c':
                conversion += 1
            elif attack_type(pp) == 's':
                synthesis += 1
            else:
                genuine += 1
        report = """
            +++++ Summary for [{}] +++++
            Total [{}] valid files to be processed.
            Genuine files: [{}]
            Attack files: [{}]
                - [{}] replay attacks
                - [{}] voice conversion attacks
                - [{}] speech synthesis attacks
        """.format(self.__class__.__name__, len(self._filepaths),
                   genuine,
                   replay+conversion+synthesis,
                   replay, conversion, synthesis)

        return report
