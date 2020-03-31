"""A Unix pipe-style command-line interface for audio processing.

This utility is inspired by imagepipe:
https://github.com/pallets/click/blob/master/examples/imagepipe/imagepipe.py
"""
from importlib import import_module
import os

import click
import soundfile as sf

from .utils import processor, generator
from .filetypes import SimpleFile
from ..data.datatype import Audio

SOUNDFILE_FORMAT = tuple(sf.available_formats().keys())
SPH2PIPE_FORMAT = 'WV1', 'WV2', 'SPH'
LISTING_FORMAT = 'LST', 'NDX', 'TXT'
SUPPORTED_FEATURE = 'STFT', 'CQT', 'GAMMATONE', 'MFCC',\
                     'MELSPEC', 'PNCC', 'PNSPEC', 'MODSPEC'
cur_dir = os.path.dirname(__file__)
EXTRACT_CFG = os.path.abspath(f'{cur_dir}/../cfg/audpipe/extract.ini')
MAXNFRAME_SHOW = 1000


@click.group(chain=True)
def cli():
    """audpipe: Batch-process AUDio through audlib in a PIPEline.

    Chain commands in any sensible sequence, starting with `open`.

    Examples:

    * Open a single file

        $ audpipe open path/to/audio.wav

    * Read all audio files in a (nested) directory

        $ audpipe open /some/dir/ read

    * Extract MFCC of a single file and save to a npy file in ./mfcc/

        $ audpipe open a.wav read extract mfcc save -o ./mfcc

    * Batch-extract PNCC of a directory using a custom cfg, and save.

        $ audpipe open ./SWITCHBOARD/ read extract pncc --cfg 8khz.ini save

    * Extract and display spectrogram of a dataset one at a time

        $ audpipe open ./WSJ/ read extract stft show

    * List all options of a command

        $ audpipe <command> --help

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
        click.echo('')

    click.echo("Done.")


@cli.command('open')
@click.argument('path', nargs=1, type=click.Path(exists=True))
@generator
def open_cmd(path):
    """Load one or multiple files or directories for processing."""
    def isaudio(p):
        return p.upper().endswith(SOUNDFILE_FORMAT) or\
            p.upper().endswith(SPH2PIPE_FORMAT)

    def islst(p): return p.upper().endswith(LISTING_FORMAT)

    if os.path.isdir(path):
        # root = path; path = rest/of/the/paths.wav
        genfiles = import_module('audlib.io.batch').genfiles
        for pp in genfiles(path, filt=isaudio, relpath=True):
            click.echo(f"OPEN [{os.path.join(path, pp)}] | ", nl=False)
            yield SimpleFile(root=path, path=pp)
    elif isaudio(path):
        # root = dirname(path); path = basename(path)
        dd, bb = os.path.split(path)
        click.echo(f"OPEN [{path}] | ", nl=False)
        yield SimpleFile(root=dd, path=bb)
    elif islst(path):
        # root = cwd; path = each line
        cwd = os.getcwd()
        with open(path) as fp:
            for pp in fp:
                pp = pp.rstrip()
                assert os.path.exists(os.path.join(cwd, pp))
                click.echo(f"OPEN [{os.path.join(cwd, pp)}] | ", nl=False)
                yield SimpleFile(root=cwd, path=pp)
    else:
        raise ValueError("Unrecognizable path.")


@cli.command('read')
@processor
def read_cmd(files):
    """Read audio files."""
    audioread = import_module('audlib.io.audio').audioread
    for ff in files:
        fpath = os.path.join(ff.root, ff.path)
        ff.data = Audio(*audioread(fpath))
        click.echo(f"READ [{len(ff.data.signal)/ff.data.samplerate:.3f} s] | ",
                   nl=False)
        yield ff


@cli.command('extract')
@click.argument('feature', nargs=1,
                type=click.Choice(SUPPORTED_FEATURE, case_sensitive=False))
@click.option('--cfg', default=EXTRACT_CFG, show_default=True,
              help=f'Config file. See default as example.')
@processor
def extract_cmd(files, feature, cfg):
    """Extract features."""
    assert os.path.exists(cfg), f"Invalid configuration path: {cfg}"
    assert feature is not None
    import configparser
    import audlib.sig.callables as callables
    cc = configparser.ConfigParser()
    cc.read(cfg)
    feature = feature.lower()
    if feature in ['stft', 'gammatone', 'mfcc', 'melspec', 'pncc', 'pnspec']:
        # Need STFT
        sr = int(cc['STFT']['sampling_rate'])
        windowlen = float(cc['STFT']['window_length'])
        hop = float(cc['STFT']['hop_fraction'])
        nfft = int(cc['STFT']['nfft'])
        STFT = callables.STFT(sr, windowlen, hop, nfft)
    if feature in ['gammatone', 'pncc', 'pnspec']:
        # Need gammatonespec
        nchan = int(cc['GammatoneSpec']['nchannels'])
        GammatoneSpec = callables.GammatoneSpec(STFT, nchan)

    if feature == 'stft':
        Feature = STFT
    elif feature == 'cqt':
        sr = int(cc['CQT']['sampling_rate'])
        fr = int(cc['CQT']['frame_rate'])
        fc_min = float(cc['CQT']['minimum_center_frequency'])
        bpo = int(cc['CQT']['bins_per_octave'])
        Feature = callables.CQT(sr, fr, fc_min, bpo)
    elif feature == 'gammatone':
        Feature = GammatoneSpec
    elif feature == 'melspec':
        nchan = int(cc['MFCC']['nchannels'])
        Feature = callables.MFCC(STFT, nchan, 0)
    elif feature == 'mfcc':
        nchan = int(cc['MFCC']['nchannels'])
        ncep = int(cc['MFCC']['ncepstra'])
        cmn = cc['MFCC']['cepstral_mean_normalization'].lower() == 'true'
        Feature = callables.MFCC(STFT, nchan, ncep, cmn)
    elif feature == 'pncc':
        ncep = int(cc['PNCC']['ncepstra'])
        cmn = cc['PNCC']['cepstral_mean_normalization'].lower() == 'true'
        Feature = callables.PNCC(GammatoneSpec, ncep, cmn)
    elif feature == 'pnspec':
        Feature = callables.PNCC(GammatoneSpec, 0)
    elif feature == 'modspec':
        cc = cc['ModulationSpec']
        sr = int(cc['sampling_rate'])
        fr = int(cc['frame_rate'])
        nchan = int(cc['nchannels'])
        fc_mod = float(cc['modulation_frequency'])
        norm = cc['long_term_normalization'].lower() == 'true'
        Feature = callables.ModulationSpec(sr, fr, nchan, fc_mod, norm)
    else:
        raise ValueError("Invalid feature.")

    for ff in files:
        click.echo(
            f"EXTRACT [{feature.upper()}] | ", nl=False)
        ff.data = Feature(ff.data.signal)
        yield ff


@cli.command('show')
@click.option('--interactive/--no-interactive', ' /-I', default=False)
@processor
def show_cmd(files, interactive):
    """Display features using matplotlib."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.ion()
    plt.show()
    cbar = None
    for ff in files:
        data = ff.data
        if len(data) > MAXNFRAME_SHOW:
            data = data[:MAXNFRAME_SHOW]
        click.echo(f"SHOW [{len(data)} frames] | ", nl=False)
        ax.clear()
        im = ax.pcolormesh(list(range(len(data))),
                           list(range(data.shape[1])),
                           data.T, cmap='jet'
                           )
        ax.set_title(f'File [{os.path.join(ff.root, ff.path)}]')
        if not cbar:
            cbar = fig.colorbar(im, ax=ax)
        fig.canvas.draw()
        if interactive:
            input("Press a key to continue...")
        else:
            plt.pause(2)
        yield ff

    plt.close(fig)


@cli.command('save')
@click.option('-o', '--output-directory', 'outdir', default='aud.pipe',
              type=click.Path(),
              show_default=True)
@processor
def save_cmd(files, outdir):
    """Save all processed files."""
    import numpy as np
    for ff in files:
        outpath = os.path.join(outdir, ff.path)
        if not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))

        if isinstance(ff.data, Audio):
            # save each audio to a .wav file
            outpath = os.path.splitext(outpath)[0] + '.wav'
            click.echo(f"SAVE [WAV] in [{outpath}] | ", nl=False)
            sf.write(outpath, ff.data.signal, ff.data.samplerate)
        elif isinstance(ff.data, np.ndarray):
            # Interpret as feature to be saved
            outpath = os.path.splitext(outpath)[0] + '.npy'
            click.echo(f"SAVE [NPY] in [{outpath}] | ", nl=False)
            np.save(outpath, ff.data)
        else:
            raise NotImplementedError

        yield ff
