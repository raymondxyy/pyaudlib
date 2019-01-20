"""Some helper functions for reading configuration files in audlib.cfg."""

# Use cases:
#   1. Users run scripts and call functions blindly. By default, the current
#      active configuration `__active__.cfg` will be loaded and used. The active
#      configuration could be:
#      * pyaudiolib's default configuration `default.cfg`
#      * any configuration that's loaded in the previous call of `cfgload`
#   2. Users call `cfgreset`. This resets cfg to `default.cfg`.
#   3. Users call `cfgload` with argument. This loads a configuration specified
#      by users to `__active__.cfg`. The specification could be:
#      * a full path
#      * a name without `.cfg` - this will be parsed as cfg file in `cfg/`
#   4. Users call `cfgadd` with appropriate path to cfg and name. This copies
#      user-specified cfg file to `cfg/` and give it the corresponding name.
#   5. Users call `cfgdel` with appropriate name. This attempts to delete the
#      cfg file with the corresponding name.
#
import configparser as CP
import os
import subprocess
from pprint import pprint
from .io.batch import dir2files

__cfgdir__ = os.path.abspath(os.path.join(os.path.dirname(__file__), '../cfg'))
__active__ = os.path.join(__cfgdir__, '__active__.cfg')
__default__ = os.path.join(__cfgdir__, 'default.cfg')

# A "private" INFO field for __active__.cfg
_info = 'INFO'
_name = 'name'


def cfgload(cpath=None, verbose=False):
    """
    Load a configuration file specified by `cpath`.

    Arguments:
        [cpath]: string
            Path to a configuration file. There are 3 options:
                1. full path to a valid .cfg file.
                2. a name stored in cfg/.
                3. None. This reads default.cfg.
        [verbose]: boolean
            Print `cpath` if set to True.

    Returns a config object.
    """
    if cpath is None:
        # Load current __active__.cfg
        # only copy default.cfg to __active__.cfg if not present
        if verbose:
            print("Using active configuration.")
        if not os.path.exists(os.path.abspath(__active__)):
            return cfgreset(verbose=verbose)
        else:  # use existing __active__.cfg
            cfile = cfgread(__active__)
            if verbose:
                cname = cfile.get(_info, _name)
                print("Using configuration [{}].".format(cname))
                cfgshow(cfile)
            return cfile

    elif '.cfg' not in cpath:  # file stored in cfg/. search by name
        cpath = os.path.join(__cfgdir__, "{}.cfg".format(cpath))
        assert os.path.exists(cpath)
    else:  # full path
        cpath = os.path.abspath(cpath)
    cfile = __activate__(cpath)
    if verbose:
        print("Reading configuration from [{}].".format(cpath))
    return cfile


def cfgreset(verbose=False):
    """Reset __active__.cfg to default.cfg."""
    cfile = __activate__(__default__)
    if verbose:
        print("Resetting configuration to [default].")
        cfgshow(cfile)
    return cfile


def cfgadd(cpath, cfgname=None):
    """Add a configuration file specified by `cpath` to cfg/.

    NOTE: this will not activate this file.

    Arguments:
        cpath: string
            Path to a configuration file.
        [cfgname]: string
            File name to be saved as `cfgname`.cfg.
        [verbose]: boolean
            Print `cpath` if set to True.

    Returns True on success, False otherwise.
    """
    cpath = os.path.abspath(cpath)
    assert os.path.exists(cpath)
    assert (__checkpath__(cpath) and __checkfile__(cpath))

    if cfgname is None:  # User does not specify name. Use file name of cpath
        savpath = os.path.join(__cfgdir__, os.path.basename(cpath))
    else:  # user specifies a name.
        assert cfgname != ('__active__' or 'default')
        savpath = os.path.join(__cfgdir__, "{}.cfg".format(cfgname))
    if os.path.exists(savpath):
        print("[{}] exists! Use a different name.".format(savpath))
        return False

    subprocess.call(['cp', cpath, savpath])
    return True


def cfgdel(cfgname, verbose=False):
    """Delete configuration from cfg list. Returns True on success."""
    cpath = os.path.join(__cfgdir__, "{}.cfg".format(cfgname))
    if os.path.exists(cpath):
        if verbose:
            print("Deleting [{}] from cfglist.".format(cfgname))
        subprocess.call(['rm', cpath])
        return True
    else:
        print("[{}] does not exist.".format(cfgname))
        return False


def cfgshow(cdict):
    """Print configuration dictionary on screen."""
    pprint(cdict)
    return


def cfgstatus():
    """Print a list of available cfg files and highlight the one in use."""
    cpaths, cnames = __cfglist__()
    if os.path.exists(__active__):
        cfile = cfgread(__active__)
        active = cfile[_info]['name']
    else:
        active = ''
    print("List of stored configurations. Active cfg is surrounded by [.].")
    for cpath, cname in zip(cpaths, cnames):
        if cname == '__active__':
            continue
        if cname == active:
            print("\t[{}]".format(cname))
        else:
            print("\t{}".format(cname))


def __cfglist__():
    """List all existing configurations in `cfg/`. Exclude __active__.cfg."""
    cpaths = dir2files(__cfgdir__, lambda s: s.endswith('.cfg'))
    cnames = map(lambda p: os.path.basename(p).split('.')[0], cpaths)
    return cpaths, cnames


def __checkpath__(cpath):
    """Check if cfg file specified by path is a valid configuration path."""
    cname, cext = os.path.basename(cpath).split('.')
    if cext != 'cfg':
        print("Configuration file extension must be [.cfg].")
        return False
    if cname == '__active__':
        print("Configuration file name cannot be [__active__].")
        return False
    if os.path.dirname(cpath) == __cfgdir__:
        print("Do not directly manipulate [cfg/].")
        return False
    return True


def __checkfile__(cdict):
    """Check if cdict is a valid configuration dictionary."""
    # TODO
    return True


def __activate__(cpath):
    """Read from a configuration file and save to __active__.cfg.

    Add a INFO field with name if necessary.
    """
    assert os.path.dirname(cpath) == __cfgdir__
    cfile = cfgread(cpath)
    if _info not in cfile:
        cfile.add_section(_info)
    if _name not in cfile[_info]:  # add a name field
        cfile[_info][_name] = os.path.basename(cpath).split('.')[0]
    with open(__active__, 'w') as wfp:
        cfile.write(wfp)
    return cfile


def cfgread(cpath):
    settings = CP.ConfigParser()
    settings._interpolation = CP.ExtendedInterpolation()
    settings.read(cpath)
    return settings
