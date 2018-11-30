# AudioTransformer Class for easy batch audio I/O
# Author: Raymond Xia (yangyanx@andrew.cmu.edu)
#
# Change Log:
#   * 2018/1/1:
#       - Ready for package
#       - Revised to be compatible with Python 3
#       - Implemented AudioTransformer with BatchIO
#       - Documentation added.

# Misc. libraries
import os
import random
from pdb import set_trace


class BatchIO(object):
    """This is an abstract class for batch file processing.
    The class accepts an input path and optionally an output path.
    Input path could point to a file or a directory, and so does the output
    path. The class produces a generator which, on each iteration, yields a
    path under input path that points to a qualified file. Users may define
    the operation to be done on each file path.
    Arguments:
    Yields:
    """

    def __init__(self, indir, in_op=None, in_format=None,
                 transform_op=None,
                 outdir=None, out_op=None, out_format=None,
                 flatten=False, random_order=False, repeat=1,
                 maxnum=None, verbose=False):
        if type(indir) is list:
            self.indir = list(map(os.path.abspath, indir))
        else:
            self.indir = os.path.abspath(indir)
        if in_format is None:
            self.__support__ = ()
        else:
            assert type(in_format) is tuple
            self.__support__ = in_format  # supporting format

        self.current_path = None  # for writing files

        # Output path related parameters
        self.outdir = os.path.abspath(outdir) if outdir is not None else None
        self.flatten = flatten  # Output to a flattened directory?
        if self.outdir is not None:
            if out_format is None:
                raise ValueError("Output format needs to be specified!")
            else:
                assert type(out_format) is str
                self.__out_format__ = out_format
        else:  # No output
            self.__out_format__ = None

        # Operations parameters
        self.in_op = in_op
        self.transform_op = transform_op
        self.out_op = out_op
        if self.out_op:
            assert self.outdir is not None
            assert self.__out_format__ is not None

        # Other parameters
        self.random_order = random_order
        self.repeat = repeat  # number of times indir will be walked through
        self.maxnum = maxnum  # maximum number of files to be processed
        self.verbose = verbose

        # collect good file paths first
        if len(self.__support__) == 0:  # accept any file extension
            def is_good_format(fname): return True
        else:
            is_good_format = \
                lambda fname: fname.lower().endswith(self.__support__)
        if type(self.indir) is list:
            self.good_paths = dirs2files(self.indir, is_good_format)
        else:
            self.good_paths = dir2files(self.indir, is_good_format)

        if (type(self.indir) is list) and (self.outdir is not None):  # file mode
            assert self.outdir.lower().endswith(self.__support__)
        self.__file_mode__ = (type(self.indir) is list)

        if self.verbose:
            self.__print_summary__()

    def __print_summary__(self):
        """
        Print out the function of this class.
        """
        print("#"*80)
        print('Summary of this BatchIO Class')
        print("Base Directory: [{}]".format(self.indir))
        print("Supported File format: [{}]".format(self.__support__))
        print("Number of supported files:[{}]".format(len(self.good_paths)))
        in_op_name = 'None' if self.in_op is None else self.in_op.__name__
        transform_op_name = 'None' if self.transform_op is None else self.transform_op.__name__
        out_op_name = 'None' if self.out_op is None else self.out_op.__name__
        print("Input operation to be done: [{}]".format(in_op_name))
        print("Transformation to be done: [{}]".format(transform_op_name))
        print("Output operation to be done: [{}]".format(out_op_name))
        print("#"*80)

    def __iter__(self):
        for i in range(self.repeat):  # repeat iterating all files
            # shuffle order if necessary
            if self.random_order:
                good_paths = sorted(iter(self.good_paths),
                                    key=lambda k: random.random())
            else:
                good_paths = self.good_paths
            counter = self.maxnum  # start counter
            for path in good_paths:
                if counter is not None:
                    if counter <= 0:
                        break
                    counter -= 1
                self.current_path = path
                if self.verbose:
                    print("Processing {}".format(self.current_path))

                # File -> internal representation
                if self.in_op is None:
                    rep = path  # gives full path to data only
                else:  # Otherwise process data
                    rep = self.in_op(path)

                # Internal representation -> transformed representation
                rep_t = self.transform_op(rep) if self.transform_op else rep

                # Transformed representation -> output file
                if self.out_op:  # Write to output
                    if self.__file_mode__:  # write to a single file
                        path_out = self.outdir
                    else:
                        # Form output path first
                        path_out = os.path.splitext(
                            path)[0]+self.__out_format__
                        if self.flatten:  # Write all files in a flat directory
                            path_out = os.path.join(self.outdir,
                                                    path_out.split('/')[-1])
                        else:
                            path_out = path_out.replace(
                                self.indir, self.outdir)
                    if not os.path.exists(os.path.dirname(path_out)):
                        os.makedirs(os.path.dirname(path_out))
                    self.out_op(rep_t, path_out)

                yield rep_t


def dir2files(indir, filter_fn, head_tail=False):
    """
    dir2files - Returns a list of files' full paths in `indir` that has qualified
    properties specified in `filter_fn`.
    Args:
        indir     - Input directory/path.
        filter_fn - filtering function on each file's filename.
    Returns:
        flist     - a list of full paths of file names.
    """
    #set_trace()
    indir = os.path.abspath(indir)
    if os.path.isfile(indir):  # file mode
        if filter_fn(indir):
            if head_tail:
                return [os.path.split(indir)]
            else:
                return [indir]
        else:
            return []
    flist = []
    for root, dirs, files in os.walk(indir):
        for fname in filter(filter_fn, files):
            fullpath = os.path.join(root, fname)
            if head_tail:
                flist.append((indir, os.path.relpath(fullpath, indir)))
            else:
                flist.append(fullpath)
    return flist


def dirs2files(indirs, filter_fn, head_tail=False):
    """Same as dir2file, except allowing more than 1 input directory."""
    flist = []
    for indir in indirs:
        flist.extend(dir2files(indir, filter_fn, head_tail=head_tail))
    return flist


def lst2files(lstpath, ext=None):
    """Read from a listing file and return a list of file paths."""
    with open(lstpath) as fp:
        lines = fp.readlines()
    paths = []
    for line in lines:
        fpath = line.strip()
        if ext is not None:
            fpath = "{}.{}".format(fpath, ext)
            assert os.path.isfile(fpath)
        paths.append(fpath)
    return paths

# Some useful batch configurations that are used frequently


def AudioTransformer(indir, sr, mono=True, norm=False, outdir=None,
                     flatten=False, transform=None, random_order=False,
                     repeat=1, maxnum=None, with_path=False, verbose=False):
    """
    AudioTransformer: Batch process audio in the workflow of:
        Audio path - --> representation - --> [transform] - --> [Write to output]
    Args:
        indir - Input path. Could be file path/directory, or list of directories.
        sr - sampling rate
        [mono] - force mono at reading audio
        [outdir] - output directory
        [flatten] - flatten output directory
        [transform] - transform to be applied on representation
        [random_order] - process files in random order
        [repeat] - repeat the entire process number of times
        [maxnum] - maximum number of files to be processed per repetition
        [with_path] - return full path to each audio file as well
        [verbose] - enable verbose.
    Returns:
        Iterator which iterates a transform per iteration, and optionally
        write to disk if outdir is specified.
    """
    from .audio_io import audioread, audiowrite
    if verbose:
        print("AudioTransformer sampling rate: [{}].".format(sr))
    in_format = ('.wav', '.flac', '.sph')
    out_format = '.wav'
    if with_path:
        def audread(path): return (
            audioread(path, sr, force_mono=mono, norm=norm)[0], path)
    else:
        def audread(path): return audioread(
            path, sr, force_mono=mono, norm=norm)[0]
    if outdir is None:
        out_op = None
    else:
        def out_op(x, path): return audiowrite(x, sr, path, normalize=True,
                                               verbose=verbose)
    return BatchIO(indir, in_op=audread, in_format=in_format,
                   transform_op=transform,
                   outdir=outdir, out_op=out_op, out_format=out_format,
                   flatten=flatten, random_order=random_order, repeat=repeat,
                   maxnum=maxnum, verbose=verbose)


if __name__ == "__main__":
    # Use as a standalone program
    import util
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Input file/directory', required=True)
    parser.add_argument('-sr', help='Sampling rate of the system',
                        type=int, required=False)
    parser.add_argument('-c2', help='Disable mono processing',
                        action='store_true', required=False)
    parser.add_argument('-o', help='Optional output directory',
                        required=False, default=None)
    parser.add_argument('-f', action='store_true',
                        help='Enable flattened output directory', required=False,
                        default=False)
    parser.add_argument('-max', help='Maximum number of files to be processed.',
                        type=int, required=False, default=None)
    parser.add_argument('-t',
                        help='Transformation applied to signals. Currently only supporting\
        functions from util', required=False, default=None)
    parser.add_argument('-v', action='store_true',
                        help='Enable verbose', required=False,
                        default=False)
    args = parser.parse_args()
    if args.sr is None:
        args.sr = 16000
        print(
            "Sampling rate unspecified. Take [{}] as default.".format(args.sr))
    transform = None if args.t is None else getattr(util, args.t)
    force_mono = True
    if args.c2:
        force_mono = False
    transformer = AudioTransformer(args.i, args.sr, mono=force_mono,
                                   outdir=args.o, flatten=args.f, transform=transform, maxnum=args.max,
                                   verbose=args.v)
    for rep in transformer:
        continue
