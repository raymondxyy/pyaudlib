"""I/O bridge for the SPHINX feature format."""
import numpy as np
import struct
import sys


def s3read(path, featdim):
    """
    Reads SPHINX-3 (s3) feature file to a numpy array.

    Arguments:
        path   : string
            Absolute path to feature file on disk
        featdim: int
            Feature dimension

    Returns:
        result : ndarray
            Feature with dimension `timedim` by `featdim`. `timedim` is inferred
            from file.

    """
    with open(path, 'rb') as fp:
        f = fp.read()
    tot_nums = struct.unpack(">1i", f[:4])[0]  # first 4 bytes hold total size
    if tot_nums != (len(f)-4)/4:
        raise ValueError("S3 header inconsistent with file size!")
    if not (tot_nums % featdim):
        raise ValueError("Incompatible `featdim`!")

    feat = np.array(struct.unpack(">{}f".format(tot_nums), f[4:]))
    return feat.reshape((tot_nums//featdim, featdim))


def s5read(path, featdim):
    data = []
    unpack_str = '{}f'.format(featdim)
    with open(path, 'rb') as f:
        v = f.read(4)
        head = struct.unpack('I', v)[0]
        v = f.read(featdim * 4)
        while v:
            frame = list(struct.unpack(unpack_str, v))
            data.append(frame)
            v = f.read(featdim * 4)
    data = np.array(data)
    # print data.shape, head
    assert(data.shape[0] * data.shape[1] == head)
    return data


def s3write(path, feat):
    """
    Write numpy array to file in SPHINX-3 (s3) format.

    Arguments:
        path: string
            Absolute path to feature file on disk
        feat: 1D ndarray
            Feature array (only 2D array is supported)
    """
    tot_nums = feat.size
    with open(path, 'wb') as fp:
        fp.write(struct.pack(">1i", tot_nums))
        fp.write(struct.pack(">{}f".format(tot_nums), *feat.ravel()))


def s3view(feat):
    """
    Simple printing utility similar to SPHINX's cepview. Print some feature
    values on screen.
    Args:
        feat - feature in numpy array
    """
    MAXROW = 10
    MAXCOL = 10
    rows, cols = feat.shape
    sys.stdout.write("Data dimension: [{}x{}]\n".format(rows, cols))
    for r in range(min(rows, MAXROW)):
        for c in range(min(cols, MAXCOL)):
            sys.stdout.write("{:.2f} ".format(feat[r, c]))
        if MAXCOL < cols:  # ignore rest
            sys.stdout.write("...\n")
    if MAXROW < rows:
        sys.stdout.write(".\n.\n.\n")
