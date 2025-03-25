#!/usr/bin/env python
from __future__ import absolute_import
from . import config as _config
import logging
import os

from importlib.metadata import version, PackageNotFoundError  # noqa


def get_version():
    try:
        return version("lksearch")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))


class Conf(_config.ConfigNamespace):
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


conf = Conf()
log = logging.getLogger("lksearch")

from .MASTSearch import MASTSearch  # noqa
from .TESSSearch import TESSSearch  # noqa
from .KeplerSearch import KeplerSearch  # noqa
from .K2Search import K2Search  # noqa

# from .catalogsearch import *  # noqa
from . import catalogsearch  # noqa
