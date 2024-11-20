"""Catalog class to search various catalogs for missions"""

from typing import Union
import numpy as np

from astropy.coordinates import Angle, SkyCoord, Distance
from astropy.table import Table
from astropy.time import Time
import astropy.units as u

from astroquery.vizier import Vizier
from astroquery.utils.tap.core import TapPlus
from astroquery.mast import MastClass
from astroquery.simbad import Simbad

import pandas as pd

import warnings
import time
import json

from . import log
from . import PACKAGEDIR


# This is a lits of VizieR catalogs and their input parameters to be used in the
# query_skycatalog function
def _load_cat_config():
    with open(f"{PACKAGEDIR}/data/catalog_config.json", "r") as j:
        cat_dict = json.loads(j.read())
    for key in cat_dict.keys():
        cat_dict[key]["equinox"] = Time(
            cat_dict[key]["equinox"], format="jyear", scale="tt"
        )
    return cat_dict


_Catalog_Dictionary = _load_cat_config()

# Connect to the Vizier TAP server here so that we only do this once
VizTap = TapPlus(url="http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")

# TODO Swap this out with a configuration parameter maybe?
# Or None and Raise Exception
# Make this an optional keword argument for debugging/doc
_default_catalog = "tic"


# use simbad to get name/ID crossmatches
def IDLookup(search_input: Union[str, list[str]], match_catalog: str = None):
    match_list = _Catalog_Dictionary.keys()
    """Uses the Simbad name resolver and ids to disambiguate the search_input string or list. 

    Parameters
    ----------
    search_input: Union[str, list[str]]
        A string or list of strings to query simbad for ID disambiguation
    
    match_catalog: str = None
        Short name of catalog to parse the simbad id results for 

    Returns
    -------
    result: Table, list[Table]
        Results from the `~astroquery.simbad.Simbad` ID query in `~astropy.table.Table` format. 

    """
    #    match_str = None only usable in bleeding edge astroquery

    if match_catalog is not None:
        if match_catalog.lower() in match_list:
            match_str = _Catalog_Dictionary[match_catalog.lower()]["SIMBAD_match_like"]

    if isinstance(search_input, list):
        result = []
        for item in search_input:
            log.warning("Throttling query limit to Simbad's: max 5/s")
            result_iter = _IDLookup(item)
            time.sleep(0.2)
            result.append(result_iter)
    else:
        result = _IDLookup(search_input)

    return result


def _IDLookup(search_item):
    # Construct exact ID TAP queries for various surveys
    #    result_table = Simbad.query_objectids(search_item, criteria = match_str)
    result_table = Simbad.query_objectids(search_item)
    return result_table


def _get_TAP_Query(catalog: str, ID: str, max_results: int = None, id_column=None):
    if catalog not in _Catalog_Dictionary.keys():
        raise ValueError(f"{catalog} not found in TAP catalogs list")
    if max_results is None:
        max_results = len(ID.split(","))
    if id_column is None:
        id_column = _Catalog_Dictionary[catalog]["default_id_column"]

    query_model = f""" SELECT TOP {max_results} * 
    FROM "{_Catalog_Dictionary[catalog]["catalog"]}" 
    WHERE {id_column} IN({ID}) """

    return query_model


def _parse_search_input(search_input, catalog: str = None):
    if isinstance(search_input, SkyCoord):
        search_object = search_input
    elif isinstance(search_input, tuple):
        search_object = SkyCoord(search_input[0], search_input[1], unit="deg")
    elif isinstance(search_input, str):
        # Try to turn the passed string into a SkyCoord.
        try:
            search_object = SkyCoord(search_input)
        except ValueError:
            # If this fails assume we were given an object name
            search_object = search_input
            # If no catalog is specified, see if we can infer a preffered catalog
            if catalog is None:
                search_catalog, _ = _match_target_catalog(search_input)
    else:
        raise TypeError("Cannot Resolve type of search input")

    # If a catalog is passed make this the catalog to search
    # Else use a default catalog(tic?) if none is assigned
    if isinstance(catalog, str):
        search_catalog = catalog
    elif catalog is None:
        search_catalog = None
    return search_object, search_catalog


def _match_target_catalog(search_input):
    search = search_input.strip().replace(" ", "").lower()
    if search_input.isnumeric():
        # If string is purelt numbers, make no assumptions
        search_string = search_input
        search_catalog = None
    elif search[0:3] == "tic":
        search_catalog = "tic"
        search_string = search[3:]
    elif search[0:4] == "tess":
        search_catalog = "tic"
        search_string = search[4:]
    elif search[0:3] == "kic":
        search_catalog = "kic"
        search_string = search[3:]
    elif search[0:4] == "kplr":
        search_catalog = "kic"
        search_string = search[4:]
    elif search[0:4] == "epic":
        search_catalog = "epic"
        search_string = search[4:]
    elif search[0:4] == "ktwo":
        search_catalog = "epic"
        search_string = search[4:]
    elif search[0:7] == "gaiadr3":
        search_catalog = "gaiadr3"
        search_string = search[7:]
    elif search[0:4] == "gaia" and search_input[4:6] != "dr":
        search_string = search[4:]
        search_catalog = "gaiadr3"
    else:
        # If we cannot parse a catalog, make no assumptions
        search_catalog = (None,)
        search_string = search_input
    return search_catalog, search_string


def _parse_id_list(search_object):
    ids = [_parse_id(item)[0] for item in search_object]
    id_list = ", ".join(str(id) for id in ids)
    return id_list


def _parse_id(search_item):
    if isinstance(search_item, int):
        id = str(search_item)
        scat = None
    elif isinstance(search_item, str):
        scat, id = _match_target_catalog(search_item)
    else:
        id = None
        scat = None

    return id, scat


def QueryID(
    search_object: Union[str, int, list[str, int]],
    catalog: str = None,
    input_catalog: str = None,
    max_results: int = None,
    return_skycoord: bool = False,
    epoch: Union[str, Time] = None,
):
    """Searches a catalog (TIC, KIC, EPIC, or GAIA DR3) for an exact ID match and
    returns the assosciated catalog rows.  A limited cross-match between the TIC, KIC, and gaiadr3
    catalogs is possible using the catalog, and input_catalog optional parameters.

    Parameters
    ----------
    search_object : Union[str, int, list[str, int]]
        A string or integer, or list of strings or integers, that represents
        a list of IDs from a single catalog to match.  If an integer is supplied the
        catalog optional parameter must be specified.
    catalog : str, optional
        Catalog to search for an ID match to.  If no input_catalog is
        specified catalog and input_catalog are assumed to be the same.
        If search_object is a string and catalog and is None, search_object is
        parsed to try and determine the catalog, by default None
    input_catalog : str, optional
        _description_, by default None
    max_results : int, optional
        limits the maximum rows to return, by default None
    return_skycoord : bool, optional
        If true, an `~astropy.coordinates.SkyCoord` objects is returned for each
        row in the result table, by default False
    epoch : Union[str, Time], optional
        If a return_skycoord is True, epoch can be used to specify the epoch for the
        returned SkyCoord object, by default None

    Returns
    -------
    results_table: Union[Table, SkyCoord, list[SkyCoord]]
        `~astropy.table.Table` object containing the rows of the catalog with IDs matching the search_input.
        If return_skycoord is set to True, a `~astropy.coordinates.SkyCoord` object or list of `~astropy.coordinates.SkyCoord` objects
        is instead returned.

    """
    id_column = None

    if isinstance(search_object, list):
        id_list = _parse_id_list(search_object)
    else:
        id_list, scat = _parse_id(search_object)
        # IF we can figure out the soruce catalog from context -
        # EG TIC Blah, assume the catalog to search is the catalog detected
        # And th
        if catalog is None and scat is not None:
            catalog = scat
        if input_catalog is None and scat is not None:
            input_catalog = scat

    # Assuming a 1:1 match.  TODO is this a bad assumption?
    if max_results is None:
        max_results = len(np.atleast_1d(search_object))

    if catalog is not None and input_catalog is not None:
        if catalog != input_catalog:
            max_results = max_results * 10
            if input_catalog in np.atleast_1d(
                _Catalog_Dictionary[catalog]["crossmatch_catalogs"]
            ):
                if _Catalog_Dictionary[catalog]["crossmatch_type"] == "tic":
                    # TIC is is crossmatched with gaiadr3/kic
                    # If KIC data for a gaia source or vice versa is desired
                    # search TIC to get KIC/gaia ids then Search KIC /GAIA
                    source_id_column = _Catalog_Dictionary["tic"][
                        "crossmatch_column_id"
                    ][input_catalog]
                    new_id_table = _QueryID(
                        "tic", id_list, max_results, id_column=source_id_column
                    )
                    id_list = ", ".join(
                        new_id_table[
                            _Catalog_Dictionary["tic"]["crossmatch_column_id"][catalog]
                        ].astype(str)
                        # .values
                    )
                if _Catalog_Dictionary[catalog]["crossmatch_type"] == "column":
                    # TIC is is crossmatched with gaiadr3/kic
                    # If we want TIC Info for a gaiadr3/KIC source - match appropriate column in TIC
                    id_column = _Catalog_Dictionary[catalog]["crossmatch_column_id"][
                        input_catalog
                    ]
            else:
                raise ValueError(
                    f"{input_catalog} does not have crossmatched IDs with {catalog}. {catalog} can be crossmatched with {_Catalog_Dictionary[catalog]["crossmatch_catalogs"]}"
                )
    else:
        if catalog is None:
            catalog = _default_catalog

    results_table = _QueryID(catalog, id_list, max_results, id_column=id_column)
    if return_skycoord:
        return _table_to_skycoord(results_table, epoch=epoch, catalog=catalog)
    else:
        return CatalogResult(results_table)


def _QueryID(catalog: str, id_list: str, max_results: int, id_column: str = None):
    query = _get_TAP_Query(
        catalog, id_list, max_results=max_results, id_column=id_column
    )
    async_limit = 1e3
    if max_results > async_limit:
        # we should chex max_results and if low do a synchronous query, if large async
        log.warn(
            f"Warning: Queries over {async_limit} will be done asynchronously, and may take some time"
        )
        job = VizTap.launch_job_async(query)
        job.wait_for_job_end()
        results_table = job.get_data()
    else:
        job = VizTap.launch_job(query)
        results_table = job.get_data()
    return results_table  # .to_pandas()


def QueryPosition(
    search_input: Union[str, SkyCoord, tuple, list[str, SkyCoord, tuple]],
    epoch: Union[str, Time] = None,
    catalog: str = "tic",
    radius: Union[float, u.Quantity] = u.Quantity(100, "arcsecond"),
    magnitude_limit: float = 18.0,
    max_results: int = None,
    return_skycoord: bool = False,
):
    """
    Query a catalog for a single source location, obtain nearby sources
    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord, string, tuple, or list thereof
        Coordinates around which to do a radius query. If passed a string, will first try to resolve string as a coordinate using `~astropy.coordinates.SkyCoord`, if this fails then tries to resolve the string as a name using '~astroquery.mast.MastClass.resolve_object'.
    epoch: astropy.time.Time
        The time of observation in JD.
    catalog: str
        The catalog to query, either 'kepler', 'k2', or 'tess', 'gaia'
    radius : float or astropy quantity
        Radius in arcseconds to query
    magnitude_limit : float
        A value to limit the results in based on the Tmag/Kepler mag/K2 mag or Gaia G mag. Default, 18.
    return_skycoord: bool
        Whether to return an astropy.coordinates.SkyCoord object. Default is False.
    Returns
    -------
    result: Table or astropy.coordinates.SkyCoord
        By default returns a  pandas dataframe of the sources within radius query, corrected for proper motion. Optionally will return astropy.coordinates.SkyCoord object.

    """

    coord, search_catalog = _parse_search_input(search_input, catalog=catalog)

    # Check to make sure that user input is in the correct format
    if not isinstance(coord, SkyCoord):
        if isinstance(coord, str):
            coord = MastClass().resolve_object(coord)
        else:
            raise TypeError(f"could not resolve {coord} to SkyCoord")
    if epoch is not None:
        if not isinstance(epoch, Time):
            try:
                epoch = Time(epoch, format="jd")
            except ValueError:
                raise TypeError(
                    "Must pass an `astropy.time.Time object` or parsable object."
                )
            raise TypeError(
                "Must pass an `astropy.time.Time object` or parsable object."
            )
    if not coord.isscalar:
        raise ValueError("must pass one target only.")

    # Here we check to make sure that the radius entered is in arcseconds
    # This also means we do not need to specify arcseconds in our catalog query
    try:
        radius = u.Quantity(radius, "arcsecond")
    except u.UnitConversionError:
        raise

    # Check to make sure that the catalog provided by the user is valid for this function
    if search_catalog.lower() not in _Catalog_Dictionary.keys():
        raise ValueError(f"Can not parse catalog name '{catalog}'")
    catalog_meta = _Catalog_Dictionary[search_catalog.lower()]

    # Get the Vizier catalog name
    catalog_name = catalog_meta["catalog"]

    # Get the appropriate column names and filters to be applied
    filters = Vizier(
        columns=catalog_meta["columns"],
        column_filters={catalog_meta["column_filters"]: f"<{magnitude_limit}"},
    )
    # The catalog can cut off at 50 - we dont want this to happen
    if max_results is not None:
        filters.ROW_LIMIT = max_results
    else:
        filters.ROW_LIMIT = -1
    # Now query the catalog
    result = filters.query_region(coord, catalog=catalog_name, radius=Angle(radius))
    if len(result) == 0:
        # Make an empty Table
        empty_table = pd.DataFrame(
            columns=[
                *catalog_meta["columns"],
                "ID",
                "RA",
                "Dec",
                "Separation",
                "Relative_Flux",
            ]
        )
        # Make Sure Columns are consistently renamed for the catalog
        empty_table = empty_table.rename(
            {
                i: o
                for i, o in zip(catalog_meta["rename_in"], catalog_meta["rename_out"])
            },
            axis=1,
        )
        # Make sure we have an index set
        empty_table = empty_table.set_index("ID")
        empty_table = Table.from_pandas(empty_table)
        return empty_table

    result = result[catalog_name]
    # Rename the columns so that the output is uniform
    result.rename_columns(
        catalog_meta["rename_in"],
        catalog_meta["rename_out"],
    )
    if catalog_meta["prefix"] is not None:
        prefix = catalog_meta["prefix"]
        result["ID"] = [f"{prefix} {id}" for id in result["ID"]]
    if epoch is None:
        epoch = catalog_meta["equinox"]
    c = _table_to_skycoord(
        table=result,
        equinox=catalog_meta["equinox"],
        epoch=epoch,
        catalog=search_catalog,
    )
    ref_index = np.argmin(coord.separation(c).arcsecond)
    sep = c[ref_index].separation(c)
    if return_skycoord:
        s = np.argsort(sep.deg)
        return c[s]
    result["RA"] = c.ra.deg
    result["Dec"] = c.dec.deg
    result["Separation"] = sep.arcsecond
    # Calculate the relative flux
    result["Relative_Flux"] = 10 ** (
        (
            result[catalog_meta["default_mag"]]
            - result[catalog_meta["default_mag"]][ref_index]
        )
        / -2.5
    )
    # Now sort the table based on separation
    result.sort(["Separation"])
    # return result
    # result = result.to_pandas().set_index("ID")
    return CatalogResult(result[_get_return_columns(result.columns)])


def _get_return_columns(columns):
    """Convenience function to reorder columns and remove motion columns."""
    downselect_columns = list(
        set(columns)
        - set(
            [
                "ID",
                "RA",
                "Dec",
                "RAJ2000",
                "DEJ2000",
                "Plx",
                "pmRA",
                "pmDE",
                "Separation",
                "Relative_Flux",
            ]
        )
    )
    downselect_columns = np.hstack(
        [
            np.sort([i for i in downselect_columns if i.endswith("mag")]),
            np.sort([i for i in downselect_columns if not i.endswith("mag")]),
        ]
    )
    new_columns = [
        "ID",
        "RA",
        "Dec",
        "Separation",
        "Relative_Flux",
        "pmRA",
        "pmDE",
        *downselect_columns,
    ]
    return new_columns


def _table_to_skycoord(
    table: Table, equinox: Time = None, epoch: Time = None, catalog=None
) -> SkyCoord:
    """
    Convert a table input to astropy.coordinates.SkyCoord object

    Parameters
    ----------
    table : astropy.table.Table
        astropy.table.Table which contains the coordinates of targets and proper motion values
    equinox: astropy.time.Time
        The equinox for the catalog
    epoch : astropy.time.Time
        Desired time of the observation

    Returns
    -------
    coords : astropy.coordinates.SkyCoord
       SkyCoord object with RA, Dec, equinox, and proper motion parameters.
    """

    if equinox is None and catalog is None:
        _, catalog = _parse_id(table[0][0])
    if equinox is None and catalog is not None:
        equinox = _Catalog_Dictionary[catalog]["equinox"]
    if epoch is None and catalog is not None:
        epoch = equinox

    # We need to remove any nan values from our proper  motion list
    # Doing this will allow objects which do not have proper motion to still be displayed
    if "pmRA" in table.keys():
        table["pmRA"] = np.ma.filled(table["pmRA"].astype(float), 0.0)
    else:
        table["pmRA"] = 0.0 * u.mas / u.yr

    if "pmDE" in table.keys():
        table["pmDE"] = np.ma.filled(table["pmDE"].astype(float), 0.0)
    else:
        table["pmDE"] = 0.0 * u.mas / u.yr

    # If an object does not have a parallax then we treat it as if the object is an "infinite distance"
    # and set the parallax to 1e-7 arcseconds or 10Mpc.
    if "Plx" in table.keys():
        table["Plx"] = np.ma.filled(table["Plx"].astype(float), 1e-4)
    else:
        table["Plx"] = 1e-4 * u.mas

    if "RAJ2000" in table.keys() and "DEJ2000" in table.keys():
        ra = table["RAJ2000"]
        dec = table["DEJ2000"]
    elif "RA" in table.keys() and "Dec" in table.keys():
        ra = table["RA"] * u.deg
        dec = table["Dec"] * u.deg
    else:
        raise KeyError("RA, DEC Disambiguation failed")
    # Suppress warning caused by Astropy as noted in issue 111747 (https://github.com/astropy/astropy/issues/11747)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="negative parallaxes")

        # Get the input data from the table
        c = SkyCoord(
            ra=ra,
            dec=dec,
            distance=Distance(parallax=table["Plx"].quantity, allow_negative=True),
            pm_ra_cosdec=table["pmRA"],
            pm_dec=table["pmDE"],
            frame="icrs",
            obstime=equinox,
        )

    # Suppress warning caused by Astropy as noted in issue 111747 (https://github.com/astropy/astropy/issues/11747)
    if epoch != equinox:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="ERFA function")
            warnings.filterwarnings("ignore", message="invalid value")
            c = c.apply_space_motion(new_obstime=epoch)
    return c


class CatalogResult(Table):
    def to_SkyCoord(self, equinox: Time = None, epoch: Time = None):
        return _table_to_skycoord(Table(self), equinox=equinox, epoch=epoch)
