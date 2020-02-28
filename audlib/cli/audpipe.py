"""A Unix pipe-style command-line interface for audio processing.

This utility is heavily inspired by imagepipe:
https://github.com/pallets/click/blob/master/examples/imagepipe/imagepipe.py

Currentlly registered commands:
"""
from importlib import import_module
import os

import click

from .utils import processor, generator
from .filetypes import SimpleFile
from ..data.datatype import Audio
from .cfg import AUDPIPE_DEFAULT as _cfg


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
@click.argument('path', nargs=1, type=click.Path(exists=True))
@generator
def open_cmd(path):
    """Load one or multiple files or directories for processing."""
    def isaudio(p):
        return p.lower().endswith(
            ('.wav', '.flac', '.sph', '.aiff', '.wv1', '.wv2')
        )

    def islst(p): return p.lower().endswith(('.lst', '.ndx'))

    if os.path.isdir(path):
        # root = path; path = rest/of/the/paths.wav
        genfiles = import_module('audlib.io.batch').genfiles
        for pp in genfiles(path, filt=isaudio, relpath=True):
            click.echo(f"Open root:[{path}], path:[{pp}]")
            yield SimpleFile(root=path, path=pp)
    elif isaudio(path):
        # root = dirname(path); path = basename(path)
        dd, bb = os.path.split(path)
        click.echo(f"Open root:[{dd}], path:[{bb}]")
        yield SimpleFile(root=dd, path=bb)
    elif islst(path):
        # root = cwd; path = each line
        cwd = os.getcwd()
        with open(path) as fp:
            for pp in fp:
                pp = pp.rstrip()
                assert os.path.exists(os.path.join(cwd, pp))
                click.echo(f"Open root:[{cwd}], path:[{pp}]")
                yield SimpleFile(root=cwd, path=pp)
    else:
        raise ValueError("Unrecognizable path.")


@cli.command('read')
@click.option('-r', '--rate', 'sr', type=int, default=_cfg['rate'])
@click.option('--mono/--no-mono', default=_cfg['force-mono'])
@processor
def read_cmd(files, sr, mono):
    """Read audio files.

    TODO: Implement resample.
    TODO: Implement mono.
    """
    audioread = import_module('audlib.io.audio').audioread
    for ff in files:
        fpath = os.path.join(ff.root, ff.path)
        ff.data = Audio(*audioread(fpath))
        yield ff


@cli.command('save')
@click.option('-o', '--output-directory', 'out', default='aud.pipe',
              type=click.Path(),
              show_default=True)
@click.option('-r', '--rate', 'sr', type=int, default=_cfg['rate'])
@processor
def save_cmd(files, out, sr):
    """Save all processed files."""
    for ff in files:
        if isinstance(ff.data, Audio):
            # save each audio to a .wav file
            audiowrite = import_module('audlib.io.audio').audiowrite
            outpath = os.path.join(out, ff.path)
            outpath = os.path.splitext(outpath)[0] + '.wav'
            assert os.path.abspath(outpath) != os.path.abspath(ff.path),\
                "Output will overwrite original file!"
            if not os.path.exists(os.path.dirname(outpath)):
                os.makedirs(os.path.dirname(outpath))
            click.echo('Saving to [{}].'.format(outpath))
            yield audiowrite(
                ff.data.signal, sr if sr else ff.data.samplerate, outpath)
        else:
            raise NotImplementedError
