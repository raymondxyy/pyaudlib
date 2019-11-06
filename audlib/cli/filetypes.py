"""Compound file definition."""


class SimpleFile(object):
    """A simple data structure to be piped through audpipe."""
    __slots__ = 'root', 'path', 'data'

    def __init__(self, root=None, path=None, data=None):
        """Specify file name, sampling rate, and signal here."""
        super(SimpleFile, self).__init__()
        self.root = root
        self.path = path
        self.data = data
