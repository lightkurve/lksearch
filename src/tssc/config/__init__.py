import os
import warnings
import glob
import shutil

import astropy.config as astropyconfig


ROOTNAME = "tssc"


class ConfigNamespace(astropyconfig.ConfigNamespace):
    rootname = ROOTNAME


class ConfigItem(astropyconfig.ConfigItem):
    rootname = ROOTNAME


def get_config_dir():
    """
    Determines the package configuration directory name and creates the
    directory if it doesn't exist.

    This directory is typically ``$HOME/.tssc/config``, but if the
    XDG_CONFIG_HOME environment variable is set and the
    ``$XDG_CONFIG_HOME/tssc`` directory exists, it will be that directory.
    If neither exists, the former will be created and symlinked to the latter.

    Returns
    -------
    configdir : str
        The absolute path to the configuration directory.

    """
    return astropyconfig.get_config_dir(ROOTNAME)


def get_cache_dir():
    """
    Determines the default TSSC cache directory name and creates the
    directory if it doesn't exist. If the directory cannot be access or created,
    then it returns the current directory (``"."``).

    This directory is typically ``$HOME/.tssc/cache``, but if the
    XDG_CACHE_HOME environment variable is set and the
    ``$XDG_CACHE_HOME/tssc`` directory exists, it will be that directory.
    If neither exists, the former will be created and symlinked to the latter.

    The value can be also configured via ``cache_dir`` configuration parameter.

    Returns
    -------
    cachedir : str
        The absolute path to the cache directory.

    See :ref:`configuration <api.config>` for more information.
    """
    from .. import conf

    cache_dir = conf.cache_dir
    if cache_dir is None or cache_dir == "":
        cache_dir = astropyconfig.get_cache_dir(ROOTNAME)
    cache_dir = _ensure_cache_dir_exists(cache_dir)
    cache_dir = os.path.abspath(cache_dir)

    return cache_dir


def _ensure_cache_dir_exists(cache_dir):
    if os.path.isdir(cache_dir):
        return cache_dir
    else:
        # if it doesn't exist, make a new cache directory
        try:
            os.mkdir(cache_dir)
        # user current dir if OS error occurs
        except OSError:
            warnings.warn(
                "Warning: unable to create {} as cache dir "
                " (for downloading MAST files, etc.). Use the current "
                "working directory instead.".format(cache_dir)
            )
            cache_dir = "."
        return cache_dir


def clearcache(test=True):
    # Check to see if default download dir/mastDownload exists
    mastdir = f"{get_cache_dir()}/mastDownload"
    if os.path.isdir(mastdir):
        files = glob.glob(f"{mastdir}/*")
        if test:
            print("Running in test mode, rerun with test=False to clear cache")
        for f in files:
            if test:
                print(f"removing {f}")
            else:
                shutil.rmtree(f)
