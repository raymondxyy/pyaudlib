#!/home/xyy/anaconda3/bin/python

from ..cfg import cfgload, cfgstatus, cfgreset
from ..cfg import __active__, __cfglist__
import subprocess
import os
import click


@click.group()
def cli():
    """Read or write configurations for audlib.

    Example:

    \b
    * Show current configurations.
        audiocfg show
    * Reset configurations to [default].
        audiocfg reset
    * Show all configuration files loaded in the system.
        audiocfg status
    """


@cli.command('status')
def status_cmd():
    """Show current configuration list and star the one in use."""
    cfgstatus()
    return


@cli.command('reset')
def reset_cmd():
    """Reset configuration to `default.cfg`."""
    cfgreset()


@cli.command('show')
@click.option('--filename', default=None, type=click.Path(),
              help='The format for the filename.',
              show_default=True)
def show_cmd(filename):
    """Print configuration file on screen."""
    if filename is None:
        fpath = __active__
    elif not os.path.isfile(filename):  # search from cfgdir
        fpaths, fnames = __cfglist__()
        if filename in fnames:
            fpath = fpaths[fnames.index(filename)]
    subprocess.call(['cat', fpath])


@cli.command('load')
@click.option('--cfgname', default=None, type=click.Path(),
              help='The format for the filename.',
              show_default=True)
def load_cmd(cfgname):
    """Load configuration."""
    cfgload(cfgname, verbose=True)
    cfgstatus()
    return
