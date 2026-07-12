import os
import sys
import numpy as np
from functools import wraps

from . import get_cache_dir

default_download_dir = get_cache_dir()


class SearchError(Exception):
    pass


class SearchWarning(Warning):
    pass


class SearchDeprecationError(SearchError):
    """Class for all lksearch deprecation warnings."""

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


# List of keys in a search result table
table_keys = np.array(
    [
        "intentType",
        "obs_collection_obs",
        "provenance_name",
        "instrument_name",
        "project_obs",
        "filters_obs",
        "wavelength_region",
        "target_name",
        "target_classification",
        "obs_id",
        "s_ra",
        "s_dec",
        "dataproduct_type_obs",
        "proposal_pi",
        "calib_level_obs",
        "t_min",
        "t_max",
        "exptime",
        "em_min",
        "em_max",
        "obs_title",
        "t_obs_release",
        "proposal_id_obs",
        "proposal_type",
        "sequence_number",
        "s_region",
        "jpegURL",
        "dataURL",
        "dataRights_obs",
        "mtFlag",
        "srcDen",
        "obsid",
        "objID",
        "objID1",
        "distance",
        "obsID",
        "obs_collection",
        "dataproduct_type",
        "description",
        "type",
        "dataURI",
        "productType",
        "productGroupDescription",
        "productSubGroupDescription",
        "productDocumentationURL",
        "project",
        "prvversion",
        "proposal_id",
        "productFilename",
        "size",
        "parent_obsid",
        "dataRights",
        "calib_level_prod",
        "filters_prod",
        "pipeline",
        "mission",
        "year",
        "start_time",
        "end_time",
        "targetid",
    ]
)
