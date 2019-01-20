"""Simple CLI for converting embedded-shorten files to wav."""

import click
import os
from ..io.audio import sphereread, audiowrite
from ..io.batch import dir2files


@click.command()
@click.argument('inpaths', type=click.Path(exists=True), nargs=-1)
@click.argument('outdir', default='out', type=click.Path(), nargs=1)
def sph2wav(inpaths, outdir):
    """Convert a list of (possibly compressed) sphere (.wv*) files to wav."""
    # convert any directory to valid paths to audio if necessary
    supported = ('wv', 'wv1', 'wv2', 'sph')
    apaths = []
    for path in inpaths:
        if os.path.isdir(path):
            apaths.extend(
                dir2files(path, lambda s: s.endswith(supported), True))
        elif os.path.isfile(path):
            if path.endswith(supported):
                apaths.append(os.path.split(path))
            elif path.endswith(('.txt', '.lst', 'fileids')):
                with open(path) as fp:
                    lines = fp.readlines()
                for line in lines:
                    fpath = line.strip()
                    if not os.path.isfile(fpath):  # guess extension
                        for ext in supported:
                            if os.path.isfile(fpath+ext):
                                fpath += ext
                                break
                    apaths.append(os.path.split(fpath))
            else:
                print('Invalid input name [{}]. Ignore.'.format(path))
        else:
            click.echo('File [{}] does not exist. Skipping.'.format(path))
            continue

    # now process all files
    tot = len(apaths)
    for ii, (indir, relpath) in enumerate(apaths):
        inpath = os.path.join(indir, relpath)
        outpath, _ = os.path.splitext(os.path.join(outdir, relpath))
        outpath += '.wav'
        try:
            click.echo(
                'Processing [{}/{}]: [{}]--->[{}]'.format(ii+1, tot, inpath,
                                                          outpath))
            x, xsr = sphereread(inpath)
            audiowrite(x, xsr, outpath)
        except Exception as e:
            click.echo('Could not open file [{}]: {}'.format(
                path, e), err=True)
