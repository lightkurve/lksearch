import os
import sys

from . import config

default_download_dir = config.get_cache_dir()


from functools import wraps


class SearchError(Exception):
    pass


class SearchWarning(Warning):
    pass


def suppress_stdout(f, *args, **kwargs):
    """A simple decorator to suppress function print outputs."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        # redirect output to `null`
        with open(os.devnull, "w") as devnull:
            old_out = sys.stdout
            sys.stdout = devnull
            try:
                return f(*args, **kwargs)
            # restore to default
            finally:
                sys.stdout = old_out

    return wrapper
