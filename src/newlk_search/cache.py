import tempfile
import configparser
import os

from functools import lru_cache, wraps
from joblib import Memory

import logging  # noqa: E402
from rich.logging import RichHandler  # noqa: E402

from IPython.display import display, Pretty


def get_logger():
    """Configure and return a logger with RichHandler."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARN)

    # Add RichHandler
    rich_handler = RichHandler()
    rich_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(rich_handler)
    return logger


log = get_logger()


def check_cache_dir(cache_dir):
    if not os.path.exists(cache_dir):
        try:
            os.makedirs(cache_dir)  # Try creating the cache directory
        except OSError as e:
            raise PermissionError(
                f"Unable to create cache directory at {cache_dir}."
            ) from e

    # Check if the directory is writable
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as fp:
            fp.write(b"Hello world!")
            fp.close()
            with open(fp.name, mode="rb") as f:
                f.read()
    except OSError as e:
        raise PermissionError(
            f"The cache directory at {cache_dir} is not writable."
        ) from e
    return cache_dir


def make_default_config():
    config = configparser.ConfigParser()
    config["CacheSettings"] = {
        "use_disk_cache": "False",
        "cache_dir": _default_cache_dir,
    }
    with open(_config_file_path, "w") as f:
        config.write(f)


def load_config():
    config = configparser.ConfigParser()
    if not os.path.exists(_config_file_path):
        make_default_config()
    else:
        config.read(_config_file_path)
    return config


def save_config(config):
    with open(_config_file_path, "w") as f:
        config.write(f)


def show_config():
    config = load_config()

    def parse_section(section):
        return "\n".join([f'\t{c[0]}: "{c[1]}"' for c in section.items()])

    display(
        Pretty(
            "\n".join(
                [
                    f"{section}: \n{parse_section(config[section])}"
                    for section in config.sections()
                ]
            )
        )
    )


def set_use_disk_cache(use_disk_cache: bool):
    """Set whether to cache on disk. Default is False."""
    try:
        config = load_config()
        if not isinstance(use_disk_cache, bool):
            raise ValueError("`use_disk_cache` must been type `bool`.")
        config["CacheSettings"]["use_disk_cache"] = f"{use_disk_cache}"
        save_config(config)
    except:
        log.warning(
            "Can not update config parameters. Setting `USE_DISK_CACHE` in this session."
        )
    global USE_DISK_CACHE
    USE_DISK_CACHE = use_disk_cache


def set_disk_cache_directory(cache_directory):
    try:
        config = load_config()
        if not isinstance(cache_directory, str):
            raise ValueError("`cache_directory` must been type `str`.")
        check_cache_dir(cache_directory)
        config["CacheSettings"]["cache_dir"] = cache_directory
        save_config(config)
    except:
        log.warning(
            "Can not update config parameters. Setting `CACHE_DIR` in this session."
        )
    global CACHE_DIR
    CACHE_DIR = cache_directory


_default_cache_dir = check_cache_dir(
    os.path.join(os.path.expanduser("~"), ".newlk_search")
)
_config_file_path = os.path.join(_default_cache_dir, "config.ini")

USE_DISK_CACHE = load_config()["CacheSettings"]["use_disk_cache"].lower() in [
    "true",
    "y",
    "yes",
    "on",
    "disk",
]
CACHE_DIR = load_config()["CacheSettings"]["cache_dir"]

if USE_DISK_CACHE:
    memory = Memory(CACHE_DIR, verbose=0)
else:
    memory = None


def cache(lru_cache_size=32):
    """
    Decorator to cache either on disk or in memory
    """
    if USE_DISK_CACHE:
        if not "memory" in globals():
            global memory
            memory = Memory(CACHE_DIR, verbose=0)
        if memory is None:
            memory = Memory(CACHE_DIR, verbose=0)
        if globals()["memory"].location != CACHE_DIR:
            memory = Memory(CACHE_DIR, verbose=0)

        def disk_cache_decorator(func):
            cached_func = memory.cache(func)
            return cached_func

        return disk_cache_decorator
    else:

        def memory_cache_decorator(func):
            @wraps(func)
            @lru_cache(maxsize=lru_cache_size)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return memory_cache_decorator
