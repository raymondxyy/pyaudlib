"""A Unix pipe-style command-line interface for audio processing.

This utility is heavily inspired by imagepipe:
https://github.com/pallets/click/blob/master/examples/imagepipe/imagepipe.py
"""
import os
from types import GeneratorType

import click
import numpy as np

from .cli import processor, generator
from .filetype import Audio, SIGTYPE, SAVEMODE
from ..cfg import cfgload
from ..io.audio import audioread, audiowrite
from ..io.batch import dir2files
from ..sig.transform import stccep, stlogm
from ..sig.window import hamming
from ..enhance import wiener_iter, asnr, asnr_recurrent, asnr_activate


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
    for proc in processors:
        stream = proc(stream)

    # Evaluate the stream and throw away the items.
    for _ in stream:
        pass


@cli.command('open')
@click.option('-i', '--input', 'paths', type=click.Path(), default=None,
              multiple=True, help='The audio file/directory to open.')
@click.option('-lst', 'lst', default=None,
              help='Listing file for batch processing.')
@generator
def open_cmd(paths, lst):
    """Load one or multiple files or directories for processing."""
    # TODO: Will need to move this extension to cfg
    ext = ('.wav', '.flac', '.sph', '.aiff', '.wv1', '.wv2')
    if len(paths) > 0:  # convert any directory to valid paths to audio
        ppaths = []
        for path in paths:
            path = os.path.abspath(path)
            if os.path.isfile(path):
                ppaths.append(os.path.split(path))
                continue
            ppaths.extend(dir2files(path, lambda s: s.endswith(ext), True))

    if lst is not None:  # Read any lst files
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
    tot_num = len(apaths)
    for ii, ht in enumerate(apaths):
        fpath = os.path.join(*ht)
        click.echo('Processing [{}/{}]: [{}]'.format(ii+1, tot_num, fpath))
        yield ht


@cli.command('read')
@processor
def read_cmd(paths):
    """Read audio files."""
    for ii, path in enumerate(paths):
        fpath = os.path.join(*path)
        try:
            aud, sr = audioread(fpath, sr=_sr, force_mono=_force_mono)
            yield Audio(filepath=path, sig=aud, sr=sr,
                        sigtype=SIGTYPE['AUDIO'])
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
            if (audio.sigtype == SIGTYPE['FEAT']) and\
                    (audio.savemode == SAVEMODE['BATCH']):
                # append all to 1 .npy file
                if isinstance(audio.sig, GeneratorType):
                    featout.append(np.array(list(audio.sig)))
                else:
                    featout.append(audio.sig)
            elif (audio.sigtype == SIGTYPE['FEAT']) and\
                    (audio.savemode == SAVEMODE['SINGLE']):
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
            elif audio.sigtype == SIGTYPE['AUDIO']:
                # save each file to outdir
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

    # redefine window for synthesis
    _wind = hamming(_wsize, hop=_hop, synth=True)

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
