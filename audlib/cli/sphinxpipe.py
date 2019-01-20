"""A Unix pipe-style command-line interface for SPHINX files."""

import os

import click

from .cli import cli, processor
from .filetype import Audio, SIGTYPE, SAVEMODE
from ..io.sphinx import s5read, trans2chars


@cli.command('read')
@click.option('-d', '--dimension', 'featdim', type=int, default=40,
              help='Feature dimension')
@processor
def read(paths, featdim):
    """Read sphinx feature files."""
    for ii, path in enumerate(paths):
        fpath = os.path.join(*path)
        try:
            feat = s5read(fpath, featdim)
            yield Audio(filepath=path, sig=feat, sr=None,
                        sigtype=SIGTYPE['FEAT'])
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
            yield Audio(filepath=path, sig=feat, sr=None,
                        sigtype=SIGTYPE['FEAT'], savemode=SAVEMODE['SINGLE'])
        except Exception as e:
            click.echo('Could not open file [{}]: {}'.format(
                fpath, e), err=True)
