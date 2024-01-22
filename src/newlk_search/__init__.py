#!/usr/bin/env python

from . import config as _config

import os
PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for `lightkurve`.

    Refer to `astropy.config.ConfigNamespace` for API details.

    Refer to `Astropy documentation <https://docs.astropy.org/en/stable/config/index.html#accessing-values>`_
    for usage.

    The attributes listed below are the available configuration parameters.

    Attributes
    ----------
    search_result_display_extra_columns
        List of extra columns to be included when displaying a SearchResult object.

    cache_dir
        Default cache directory for data files downloaded, etc. Defaults to ``~/.lightkurve/cache`` if not specified.

    warn_legacy_cache_dir
        If set to True, issue warning if the legacy default cache directory exists. Default is True.
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
        module="lightkurve.search"
    )

    cache_dir = _config.ConfigItem(
        None,
        "Default cache directory for data files downloaded, etc.",
        cfgtype="string",
        module="lightkurve.config"
    )

    warn_legacy_cache_dir = _config.ConfigItem(
        True,
        "If set to True, issue warning if the legacy default cache directory exists.",
        cfgtype="boolean",
        module="lightkurve.config"
    )

conf = Conf()