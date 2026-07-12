#!/usr/bin/env python
from __future__ import absolute_import
from .config import config, get_cache_dir, get_config_dir, get_config_file


# import astropy.config as _config
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

REPR_COLUMNS = [
    "target_name",
    "pipeline",
    "mission",
    "exptime",
    "distance",
    "year",
    "description",
]


log = logging.getLogger("lksearch")


from .mast import MASTSearch  # noqa
from .tess import TESSSearch  # noqa
from .kepler import KeplerSearch  # noqa
from .k2 import K2Search  # noqa

# from .catalogsearch import *  # noqa
from . import catalogsearch  # noqa
