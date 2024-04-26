#!/usr/bin/env python
from __future__ import absolute_import
from . import config as _config

import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
PREFER_CLOUD = True # Do you prefer URIs pointing to the Amazon bucket when available?
DOWNLOAD_CLOUD = True # TODO: Ask Tyler if this only downloads if there is cloud data?

from .version import __version__


class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `search`.

    Refer to `astropy.config.ConfigNamespace` for API details.

    Refer to `Astropy documentation <https://docs.astropy.org/en/stable/config/index.html#accessing-values>`_
    for usage.

    The attributes listed below are the available configuration parameters.

    Attributes
    ----------
    search_result_display_extra_columns
        List of extra columns to be included when displaying a SearchResult object.

    cache_dir
        Default cache directory for data files downloaded, etc. Defaults to ``~/.newlk_search/cache`` if not specified.

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
        module="newlk_search.search"
    )

    cache_dir = _config.ConfigItem(
        None,
        "Default cache directory for data files downloaded, etc.",
        cfgtype="string",
        module="newlk_search.config"
    )



conf = Conf()

from .MASTSearch import MASTSearch
from .TESSSearch import TESSSearch
from .KeplerSearch import KeplerSearch
from .K2Search import K2Search
