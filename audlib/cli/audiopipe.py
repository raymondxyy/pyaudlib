#!/home/xyy/anaconda3/bin/python
# Command-line audio piping utility
#
# Author: Raymond Xia (yangyanx@andrew.cmu.edu)
#
# This utility is heavily inspired by imagepipe:
# https://github.com/pallets/click/blob/master/examples/imagepipe/imagepipe.py

import click
import os
from functools import update_wrapper
from audlib.util.cfg import cfgload
from audlib.io.audio import audioread, audiowrite, __support__
from audlib.io.batch import dir2files, lst2files
from audlib.io.sphinx import s5read, trans2chars
from audlib.sig.stproc import stccep, stlogm
from audlib.sig.window import hamming
from audlib.plot import specgram
from audlib.enh.enhance import wiener_iter, asnr, asnr_recurrent, asnr_activate
import numpy as np
from types import GeneratorType

from pdb import set_trace

# Update stft with current configuration
_cfg = cfgload()['audiopipe']
# must be one of [iter/asnr/activate/recurrent/optim]
_force_mono = bool(_cfg['force_mono'])
_sr = int(_cfg['sample_rate'])
_hop = float(_cfg['hop_fraction'])
_wsize = int(float(_cfg['window_length']) * _sr)
_wind = hamming(_wsize, _hop)
_nfft = int(_cfg['nfft'])
_trange = eval(_cfg['analysis_range'])
_zphase = bool(_cfg['zphase'])
_ncep = int(_cfg['ncep'])

# Signal types enum
AUDIO, FEAT, = range(2)

# Save type enum
SINGLE, BATCH, = range(2)

# Total number of files tracker
TOT_NUM = 0


class Audio(object):
    # TODO
    """A small class holding an audio signal.

    Available fields:
        filename - file name
        sr       - sampling rate
        sig      - raw waveform OR other representations
    """

    def __init__(self, filepath=None, sr=None, sig=None, sigtype=None,
                 savemode=None):
        """Specify file name, sampling rate, and signal here."""
        super(Audio, self).__init__()
        self.path = filepath
        self.sr = sr
        self.sig = sig
        self.sigtype = sigtype
        self.savemode = savemode


@click.group(chain=True)
def cli():
    """Process a bunch of audio files through audlib in a unix pipe.

    Example:

    * Extract feature of a single file and save to filename.

        `audiopipe open -i path/to/audio.wav ceps save --filename ceps.npy`

    * Batch extract features of files in a directory, and save to out.npy.

        `audiopipe open -i path/to/audios/ ceps save`
    """


@cli.resultcallback()
def process_commands(processors):
    """Process chained commands one by one.

    This result callback is invoked with an iterable of all the chained
    subcommands.  As in this example each subcommand returns a function
    we can chain them together to feed one into the other, similar to how
    a pipe on unix works.
    """
    # Start with an empty iterable.
    stream = ()

    # Pipe it through all stream processors.
    for processor in processors:
        stream = processor(stream)

    # Evaluate the stream and throw away the items.
    for _ in stream:
        pass


def processor(f):
    """Wrap a stream of processors."""
    def new_func(*args, **kwargs):
        def processor(stream):
            return f(stream, *args, **kwargs)
        return processor
    return update_wrapper(new_func, f)


def generator(f):
    """Similar to the :func:`processor` but passes through old values
    unchanged and does not pass through the values as parameter.
    """
    @processor
    def new_func(stream, *args, **kwargs):
        for item in stream:
            yield item
        for item in f(*args, **kwargs):
            yield item
    return update_wrapper(new_func, f)


@cli.command('open')
@click.option('-i', '--input', 'paths', type=click.Path(), default=None,
              multiple=True, help='The audio file/directory to open.')
@click.option('-ext', '--extension', 'ext', default=None,
              multiple=True, help='Extension to extension-less listing.')
@click.option('-lst', 'lst', default=None,
              help='Listing file for batch processing.')
@generator
def open_cmd(paths, ext, lst):
    """Load one or multiple files or directories for processing. Default to
    audio files.
    """
    if ext is None:
        ext = __support__
    # convert any directory to valid paths to audio if necessary
    if len(paths) > 0:
        ppaths = []
        for path in paths:
            path = os.path.abspath(path)
            if os.path.isfile(path):
                ppaths.append(os.path.split(path))
                continue
            ppaths.extend(dir2files(path, lambda s: s.endswith(ext), True))

    # Incorporate lst files if neccessary
    if lst is not None:
        lpaths = []
        with open(lst) as fp:
            lines = fp.readlines()
        for line in lines:
            fpath = line.strip()
            if not os.path.isfile(fpath):  # guess extension
                for e in ext:
                    path = "{}.{}".format(fpath, e)
                    if os.path.isfile(path):
                        lpaths.append((os.path.abspath(os.path.curdir), path))
                        break
    if len(paths) > 0:
        apaths = ppaths
    elif lst is not None:
        apaths = lpaths
    else:
        raise ValueError('No files to be processed.')

    # now process all files
    global TOT_NUM
    TOT_NUM = len(apaths)
    for ii, ht in enumerate(apaths):
        fpath = os.path.join(*ht)
        click.echo('Processing [{}/{}]: [{}]'.format(ii+1, TOT_NUM, fpath))
        yield ht


@cli.command('audioread')
@processor
def audioread_cmd(paths):
    """Read audio files."""
    for ii, path in enumerate(paths):
        fpath = os.path.join(*path)
        try:
            aud, sr = audioread(fpath, sr=_sr, force_mono=_force_mono)
            yield Audio(filepath=path, sig=aud, sr=sr, sigtype=AUDIO)
        except Exception as e:
            click.echo('Could not open file [{}]: {}'.format(
                fpath, e), err=True)


@cli.command('sphinxread')
@click.option('-d', '--dimension', 'featdim', type=int, default=40,
              help='Feature dimension')
@processor
def sphinxread_cmd(paths, featdim):
    """Read sphinx feature files."""
    for ii, path in enumerate(paths):
        fpath = os.path.join(*path)
        try:
            feat = s5read(fpath, featdim)
            yield Audio(filepath=path, sig=feat, sr=None, sigtype=FEAT)
        except Exception as e:
            click.echo('Could not open file [{}]: {}'.format(
                fpath, e), err=True)


@cli.command('trans2chars')
@processor
def trans2chars_cmd(paths):
    """Read sphinx feature files."""
    for ii, path in enumerate(paths):
        fpath = os.path.join(*path)
        try:
            feat = trans2chars(fpath)
            yield Audio(filepath=path, sig=feat, sr=None, sigtype=FEAT, savemode=SINGLE)
        except Exception as e:
            click.echo('Could not open file [{}]: {}'.format(
                fpath, e), err=True)


@cli.command('save')
@click.option('-o', '--out', default='out', type=click.Path(),
              help='The format for the filename.',
              show_default=True)
@processor
def save_cmd(audios, out):
    """Save all processed files."""
    featout = []  # for saving features to 1 file if needed
    for idx, audio in enumerate(audios):
        try:
            if (audio.sigtype == FEAT) and (audio.savemode == BATCH):
                # append all to 1 .npy file
                if isinstance(audio.sig, GeneratorType):
                    featout.append(np.array(list(audio.sig)))
                else:
                    featout.append(audio.sig)
            elif (audio.sigtype == FEAT) and (audio.savemode == SINGLE):
                # save each file to a .npy file
                outpath = os.path.join(out, audio.path[1])
                outpath, _ = os.path.splitext(outpath)
                outpath += '.npy'
                if not os.path.exists(os.path.dirname(outpath)):
                    os.makedirs(os.path.didrname(outpath))
                click.echo('Saving to [{}].'.format(outpath))
                if isinstance(audio.sig, GeneratorType):
                    yield np.save(outpath, np.array(list(audio.sig)))
                else:
                    yield np.save(outpath, np.array(audio.sig))
            elif audio.sigtype == AUDIO:  # save each file to outdir
                outpath = os.path.abspath(os.path.join(out, audio.path[1]))
                click.echo('Saving to [{}].'.format(outpath))
                yield audiowrite(audio.sig, _sr, outpath)
        except Exception as e:
            click.echo('Could not save [{}]: {}'.format(
                audio.path[1], e), err=True)
    if len(featout) > 0:
        if os.path.isdir(out):
            if not os.path.exists(out):
                os.path.makedirs(out)
            outpath = os.path.join(out, 'out.npy')
        else:
            outpath = out
        click.echo('Saving to [{}].'.format(outpath))
        yield np.save(outpath, np.array(featout))


@cli.command('lspec')
@processor
def spec_cmd(audios):
    """Compute log magnitude spectrogram of each audio signal."""
    for audio in audios:
        #click.echo("Computing log |STFT| of [{}]".format(audio.filename))
        spec = stlogm(audio.sig, _sr, _wind, _hop, _nfft, trange=_trange)
        audio.sig = spec
        yield audio


@cli.command('ceps')
@click.option('--cms', type=bool, default=True,
              help='cepstral mean subtraction.')
@processor
def ceps_cmd(audios, cms):
    """Compute complex cepstra of each audio signal."""
    for audio in audios:
        ceps = np.array(
            list(stccep(audio.sig, _sr, _wind, _hop, (_ncep//2),
                        trange=_trange)))
        if cms:
            ceps -= np.mean(ceps, axis=0)
        audio.sig = ceps
        yield audio


@cli.command('enhance')
@click.option('--method', default='iter',
              help='Speech enhancement. Must be one of \
                [iter/asnr/activate/recurrent/optim]')
@click.option('--noise', default=None,
              help='Path to noise signal for estimating PSD of noise.')
@click.option('--iters', type=int, default=3,
              help='Number of iterations for iterative Wiener filtering.')
@click.option('--rule', default='wiener',
              help='Noise suppression rule for ASNR.')
@processor
def enhance_cmd(audios, method, noise, iters, rule):
    """Enhance each speech segment using method."""
    assert iters >= 1

    def enhance(*args, **kwargs):
        if method == 'iter':
            return wiener_iter(*args, **kwargs, iters=iters)
        elif method == 'asnr':
            return asnr(*args, **kwargs, rule=rule)
        elif method == 'activate':
            return asnr_activate(*args, **kwargs, rule=rule, fn='classic')
        elif method == 'recurrent':
            return asnr_recurrent(*args, **kwargs, rule=rule)
        else:
            raise NotImplementedError
    if noise is not None:
        noise = audioread(noise, _sr, _force_mono)
    for audio in audios:
        audio.sig = enhance(audio.sig, _sr, _wind, _hop, _nfft, noise=noise,
                            zphase=_zphase)
        yield audio


if __name__ == '__main__':
    cli()
