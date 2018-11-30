# Feature extraction mimicing CMU SPHINX wave2feat style.
#
# Author: Raymond Xia (yangyanx@andrew.cmu.edu)

from audlib.sig.transform import mfcc
from audlib.io.sphinx import write_sphinx
from audlib.io.audio import audioread
from audlib.io.batch import BatchIO
import os

# Global variables (default to SPHINX default)
SAMPLE_RATE = 16000  # sampling rate in Hz
FRAME_RATE = 100   # frame rate in Hz
FRAME_LENGTH = 0.025625  # frame length in seconds
NFFT = 512  # DFT point
ALPHA = .97  # pre-emphasis coefficient
NFILTERS = 40  # number of mel filters
MINFREQ = 133.33334  # lower cutoff of filterbank
MAXFREQ = 6855.4976  # upper cutoff of filterbank
BWIDTH = 1.0  # filterbank bandwidth
NCEP = 13  # number of dct coefficients to keep
DITHER = True


def feat_extract(inpath, outpath=None, verbose=False):
    """
    Wrapper function that extracts MFCC from input wave file. The input audio
    file has to be supported (.wav, .sph, .flac).
    """
    x, fs = audioread(inpath, SAMPLE_RATE)
    feat = mfcc(x, SAMPLE_RATE, FRAME_RATE, FRAME_LENGTH, nfft=NFFT, alpha=ALPHA,
                nfilts=NFILTERS, minfrq=MINFREQ, maxfrq=MAXFREQ, sumpower=True,
                bwidth=BWIDTH, sphinx=True, numcep=NCEP, dith=DITHER)
    if outpath is not None:
        # Write to file in SPHINX format
        write_sphinx(outpath, feat)
        if verbose:
            print("Saving [{}] feature to [{}].".format(option, outpath))

    return feat


def batch_feat_extract(indir, outdir, verbose=False):
    """
    Batch feature extraction: Read in all audio of supporting format from
    `indir`, extract MFCC features, and write to `outdir` with same directory
    structure.
    """
    dataset = BatchIO(indir, in_format=('wav', 'sph'))
    indir = os.path.abspath(indir)
    outdir = os.path.abspath(outdir)

    for inpath in dataset:
        if verbose:
            print("Processing [{}]".format(inpath))
        outpath = inpath.replace(indir, outdir).split('.')[0]+'.mfc'
        outdir_tmp = os.path.dirname(outpath)
        if not os.path.exists(outdir_tmp):
            os.makedirs(outdir_tmp)
        if os.path.exists(outpath):
            print("File exists. Continue.")
            continue
        #set_trace()
        feat = feat_extract(inpath, outpath)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Input file/directory', required=True)
    parser.add_argument('-o', help='output file/directory', required=True)
    parser.add_argument('-v', action='store_true', help='Enable verbose',
                        required=False, default=False)
    args = parser.parse_args()
    if os.path.isfile(args.i):
        feat_extract(args.i, args.o, verbose=args.v)
    else:
        batch_feat_extract(args.i, args.o, verbose=args.v)
