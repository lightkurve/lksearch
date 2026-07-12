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

    from . import config

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

    cache_dir = config.cache_dir
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


class Config(_config.ConfigNamespace):
    """
    Configuration parameters for `search`.

    Refer to `astropy.config.ConfigNamespace` for API details.

    Refer to `Astropy documentation <https://docs.astropy.org/en/stable/config/index.html#accessing-values>`_
    for usage.

    The attributes listed below are the available configuration parameters.

    Parameters
    ----------
    search_result_display_extra_columns
        List of extra columns to be included when displaying a SearchResult object.

    cache_dir
        Default cache directory for data files downloaded, etc. Defaults to ``~/.lksearch/cache`` if not specified.

    CLOUD_ONLY
        If False (default), will download a file whether the source is the cloud or MAST archives.
        If True, will only download/return uris for data located in the cloud (Amazon S3).

    PREFER_CLOUD
        Use Cloud-based data product retrieval where available (primarily Amazon S3 buckets for MAST holdings)

    DOWNLOAD_CLOUD
       Download cloud based data. If False, download() will return a pointer to the cloud based data instead of
       downloading it - intended usage for cloud-based science platforms (e.g. TIKE)

    CHECK_CACHED_FILE_SIZES
        Toggles whether to send requests to check whether the size of files in the local cache match the expected online file.

    """

    # Note: when using list or string_list datatype,
    # the behavior of astropy's parsing of the config file value:
    # - it does not accept python list literal
    # - it accepts a comma-separated list of string
    #   - for a single value, it needs to be ended with a comma
    # see: https://configobj.readthedocs.io/en/latest/configobj.html#the-config-file-format
    search_result_display_extra_columns = _config.ConfigItem(
        [],
        "List of extra columns to be included when displaying a SearchResult object.",
        cfgtype="string_list",
        module="lksearch",
    )

    cache_dir = _config.ConfigItem(
        None,
        "Default cache directory for data files downloaded, etc.",
        cfgtype="string",
        module="lksearch.config",
    )

    CLOUD_ONLY = _config.ConfigItem(
        False,
        "Only Download cloud based data."
        "If False, will download all data"
        "If True, will only download data located on a cloud (Amazon S3) bucket",
        cfgtype="boolean",
    )

    PREFER_CLOUD = _config.ConfigItem(
        True,
        "Prefer Cloud-based data product retrieval where available",
        cfgtype="boolean",
    )

    DOWNLOAD_CLOUD = _config.ConfigItem(
        True,
        "Download cloud based data."
        "If False, download() will return a pointer to the cloud based data"
        "instead of downloading it - intended usage for cloud-based science platforms (e.g. TIKE)",
        cfgtype="boolean",
    )

    CHECK_CACHED_FILE_SIZES = _config.ConfigItem(
        True,
        "Whether to send requests to check the size of files in the cache match the expected online file."
        "If False, lksearch will assume files within the cache are complete and will not check their file size."
        "Setting to True will create a modest speed up to retrieving paths for cached files, but will be lest robust, and return an 'UNKNOWN' status message",
        cfgtype="boolean",
    )


config = Config()
