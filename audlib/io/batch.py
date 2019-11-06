"""Utility functions for batch processing."""

import os
import random


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
            self.good_paths = []
            for dd in self.indir:
                self.good_paths.extend(lsfiles(dd, is_good_format))
        else:
            self.good_paths = lsfiles(self.indir, is_good_format)

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


def genfiles(root, filt=None, relpath=False):
    """Recursively find all files within a directory and return a generator.

    Parameters
    ----------
    root: str
        Directory to look into.
    filt: callable, optional
        Filter function applied to each file path (to keep or discard).
        Default to None, which accepts all files.
    relpath: bool, optional
        Return relative path to `root` instead of full path.
        Default to False.

    Returns
    -------
    type: generator
        A valid file path is returned on each yield.

    """
    root = os.path.abspath(root)
    if os.path.isfile(root):  # single file mode
        if (filt is None) or filt(root):
            yield root
    else:
        for rr, dirs, files in os.walk(root):
            for fname in files:
                fullpath = os.path.join(rr, fname)
                if (filt is None) or filt(fullpath):
                    if relpath:
                        yield os.path.relpath(fullpath, root)
                    else:
                        yield fullpath


def lsfiles(root, filt=None, relpath=False):
    """Recursively find all files within a directory and return a list.

    See Also
    --------
    genfiles

    """
    return list(genfiles(root, filt=filt, relpath=relpath))


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
