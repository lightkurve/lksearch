import os
import warnings
import glob
import shutil
import logging

import astropy.config as _config

ROOTNAME = "lksearch"
log = logging.getLogger(ROOTNAME)


class ConfigNamespace(_config.ConfigNamespace):
    rootname = ROOTNAME


class ConfigItem(_config.ConfigItem):
    rootname = ROOTNAME


def get_config_dir():
    """
    Determines the package configuration directory name and creates the
    directory if it doesn't exist.

    This directory is typically ``$HOME/.lksearch/config``, but if the
    XDG_CONFIG_HOME environment variable is set and the
    ``$XDG_CONFIG_HOME/lksearch`` directory exists, it will be that directory.
    If neither exists, the former will be created and symlinked to the latter.

    Returns
    -------
    configdir : str
        The absolute path to the configuration directory.

    """
    return _config.get_config_dir(ROOTNAME)


def get_config_file():
    return f"{get_config_dir()}/{ROOTNAME}.cfg"


def create_config_file(overwrite: bool = False):
    """Creates a default configuration file in the config directory"""

    from .. import conf

    # check if config file exists
    path_to_config_file = get_config_file()
    cfg_exists = os.path.isfile(path_to_config_file)

    if not cfg_exists or (cfg_exists and overwrite):
        with open(path_to_config_file, "w", encoding="utf-8") as f:
            for item in conf.items():
                f.write(f"## {item[1].description} \n")
                f.write(f"# {item[0]} = {item[1].defaultvalue} \n")
                f.write("\n")
    else:
        log.error("Config file exists and overwrite set to {overwrite}")


def get_cache_dir():
    """
    Determines the default lksearch cache directory name and creates the
    directory if it doesn't exist. If the directory cannot be access or created,
    then it returns the current directory (``"."``).

    This directory is typically ``$HOME/.lksearch/cache``, but if the
    XDG_CACHE_HOME environment variable is set and the
    ``$XDG_CACHE_HOME/lksearch`` directory exists, it will be that directory.
    If neither exists, the former will be created and symlinked to the latter.

    The value can be also configured via ``cache_dir`` configuration parameter.

    Returns
    -------
    cachedir : str
        The absolute path to the cache directory.

    See `~lksearch.Conf` for more information.
    """
    from .. import conf

    cache_dir = conf.cache_dir
    if cache_dir is None or cache_dir == "":
        cache_dir = _config.get_cache_dir(ROOTNAME)
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
    """Deletes all downloaded files in the lksearch download directory

    Parameters
    ----------
    test : bool, optional
       perform this in test mode, printing what folders will be deleted, by default True.
       Set test=False to delete cache
    """
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
