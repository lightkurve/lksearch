"""Catalog class to search various catalogs for missions"""

from typing import Union
import numpy as np

from astropy.coordinates import Angle, SkyCoord, Distance
from astropy.table import Table
from astropy.time import Time
import astropy.units as u

from astroquery.vizier import Vizier
from astroquery.utils.tap.core import TapPlus

import pandas as pd

import warnings

# This is a lits of VizieR catalogs and their input parameters to be used in the
# query_skycatalog function
_Catalog_Dictionary = {
    "kic": {
        "catalog": "V/133/kic",
        "columns": [
            "KIC",
            "RAJ2000",
            "DEJ2000",
            "pmRA",
            "pmDE",
            "Plx",
            "kepmag",
            "Radius",
            "Teff",
            "logg",
        ],
        "column_filters": "kepmag",
        "rename_in": ["KIC", "kepmag"],
        "rename_out": ["ID", "Kepmag"],
        "equinox": Time(2000, format="jyear", scale="tt"),
        "prefix": "KIC",
        "default_mag": "Kepmag",
        "default_id_column": "KIC",
        "crossmatch_catalogs": [
            "tic",
            "gaiadr3",
        ],  # gaia->tess->Kepler? possible but convoluted
        "crossmatch_type": "tic",
        "crossmatch_columns": None,
    },
    "epic": {
        "catalog": "IV/34/epic",
        "columns": [
            "ID",
            "RAJ2000",
            "DEJ2000",
            "pmRA",
            "pmDEC",
            "plx",
            "Kpmag",
            "logg",
            "Teff",
            "Rad",
            "Mass",
        ],
        "column_filters": "Kpmag",
        "rename_in": ["Kpmag", "pmDEC", "plx"],
        "rename_out": ["K2mag", "pmDE", "Plx"],
        "equinox": Time(2000, format="jyear", scale="tt"),
        "prefix": "EPIC",
        "default_mag": "K2mag",
        "default_id_column": "ID",
        "crossmatch_catalogs": None,
    },
    "tic": {
        "catalog": "IV/39/tic82",
        "columns": [
            "TIC",
            "RAJ2000",
            "DEJ2000",
            "pmRA",
            "pmDE",
            "Plx",
            "Tmag",
            "logg",
            "Teff",
            "Rad",
            "Mass",
        ],
        "column_filters": "Tmag",
        "rename_in": ["TIC", "Tmag"],
        "rename_out": ["ID", "TESSmag"],
        "equinox": Time(2000, format="jyear", scale="tt"),
        "prefix": "TIC",
        "default_mag": "TESSmag",
        "default_id_column": "TIC",
        "crossmatch_catalogs": ["gaiadr3", "kic"],  # WISE, TYCHO2
        "crossmatch_type": "column",
        "crossmatch_column_id": {"kic": "KIC", "gaiadr3": "GAIA", "tic": "TIC"},
    },
    "gaiadr3": {
        "catalog": "I/355/gaiadr3",
        "columns": [
            "DR3Name",
            "RAJ2000",
            "DEJ2000",
            "pmRA",
            "pmDE",
            "Plx",
            "Gmag",
            "BPmag",
            "RPmag",
            "logg",
            "Teff",
        ],
        "column_filters": "Gmag",
        "rename_in": ["DR3Name"],
        "rename_out": ["ID"],
        "equinox": Time(2016, format="jyear", scale="tt"),
        "prefix": None,
        "default_mag": "Gmag",
        "default_id_column": "Source",
        "crossmatch_catalogs": ["tic", "kic"],
        "crossmatch_type": "tic",
        "crossmatch_column_id": None,
    },
    "short": {
        "SkyCoordDict": {
            "ra": "RA",
            "dec": "Dec",
            "pmRA": "pmRA",
            "pmDE": "pmDE",
        }
    },
}

# Connect to the Vizier TAP server here so that we only do this once
VizTap = TapPlus(url="http://TAPVizieR.u-strasbg.fr/TAPVizieR/tap/")

# TODO Swap this out with a configuration parameter maybe?
# Or None and Raise Exception
# Make this an optional keword argument for debugging/doc
_default_catalog = "tic"


# Construct exact ID TAP queries for various surveys
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
    if search_input.isnumeric():
        # If string is purelt numbers, make no assumptions
        search_string = search_input
        search_catalog = None
    elif search_input[0:3].lower() == "tic":
        search_catalog = "tic"
        search_string = search_input[3:]
    elif search_input[0:4].lower() == "tess":
        search_catalog = "tic"
        search_string = search_input[4:]
    elif search_input[0:3].lower() == "kic":
        search_catalog = "kic"
        search_string = search_input[3:]
    elif search_input[0:4].lower() == "kplr":
        search_catalog = "kic"
        search_string = search_input[4:]
    elif search_input[0:4].lower() == "epic":
        search_catalog = "epic"
        search_string = search_input[4:]
    elif search_input[0:4].lower() == "ktwo":
        search_catalog = "epic"
        search_string = search_input[4:]
    elif search_input[0:7].lower() == "gaiadr3":
        search_catalog = "gaiadr3"
        search_string = search_input[7:]
    elif search_input[0:4].lower() == "gaia" and search_input[4:6].lower() != "dr":
        search_string = search_input[4:]
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
    search_object: Union[str, list[str]],
    catalog: Union[str, int, list[str, int]] = None,
    input_catalog: str = None,
    max_results: int = None,
    return_skycoord: bool = False,
    epoch: Union[str, Time] = None,
):
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
                        ]
                        .astype(str)
                        .values
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
        _table_to_skycoord(results_table, epoch=epoch, catalog=catalog)
    else:
        return CatalogResult(results_table)


def _QueryID(catalog: str, id_list: str, max_results: int, id_column: str = None):
    query = _get_TAP_Query(
        catalog, id_list, max_results=max_results, id_column=id_column
    )
    if max_results > 1e3:
        # we should chex max_results and if low do a synchronous query, if large async
        job = VizTap.launch_job_async(query)
        results_table = job.get_data()
    else:
        job = VizTap.launch_job(query)
        results_table = job.get_data()
    return results_table.to_pandas()


def QueryPosition(
    search_input: Union[str, SkyCoord, tuple, list[str, SkyCoord, tuple]],
    epoch: Union[str, Time] = None,
    catalog: str = "tic",
    radius: Union[float, u.Quantity] = u.Quantity(100, "arcsecond"),
    magnitude_limit: float = 18.0,
    return_skycoord: bool = False,
):
    """
    Query a catalog for a single source location, obtain nearby sources
    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord or string
        Coordinates around which to do a radius query. If passed a string, will resolve using `from_name`.
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
    result: pd.DataFrame or astropy.coordinates.SkyCoord
        By default returns a  pandas dataframe of the sources within radius query, corrected for proper motion. Optionally will return astropy.coordinates.SkyCoord object.
    """

    coord, search_catalog = _parse_search_input(search_input, catalog=catalog)

    # Check to make sure that user input is in the correct format
    if not isinstance(coord, SkyCoord):
        if isinstance(coord, str):
            coord = SkyCoord.from_name(coord)
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
    filters.ROW_LIMIT = -1
    # Now query the catalog
    result = filters.query_region(coord, catalog=catalog_name, radius=Angle(radius))
    if len(result) == 0:
        result = (
            pd.DataFrame(
                columns=[
                    *catalog_meta["columns"],
                    "RA",
                    "Dec",
                    "Separation",
                    "Relative_Flux",
                ]
            )
            .rename(
                {
                    i: o
                    for i, o in zip(
                        catalog_meta["rename_in"], catalog_meta["rename_out"]
                    )
                },
                axis=1,
            )
            .set_index("ID")
        )
        return result[_get_return_columns(result.columns)]
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
    result = result.to_pandas().set_index("ID")
    return CatalogResult(result[_get_return_columns(result.columns)])


def _get_return_columns(columns):
    """Convenience function to reorder columns and remove motion columns."""
    downselect_columns = list(
        set(columns)
        - set(
            [
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
        "RA",
        "Dec",
        "Separation",
        "Relative_Flux",
        *downselect_columns,
    ]
    return new_columns


def _table_to_skycoord(
    table: Table, equinox: Time = None, epoch: Time = None, catalog=None
):
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

    if equinox is None and catalog is not None:
        equinox = _Catalog_Dictionary[catalog]["equinox"]
    if epoch is None and catalog is not None:
        epoch = equinox

    catlist = ["short", "tic", "kic", "epic", "gaiadr3"]

    RA_Keys = ["RA", "RAJ2000", "RA_ICRS"]
    Dec_Keys = ["Dec", "DEJ2000", "DE_ICRS"]
    pmRA_Keys = ["pmRA"]
    pmDec_Keys = ["pmDE"]

    # We need to remove any nan values from our proper  motion list
    # Doing this will allow objects which do not have proper motion to still be displayed
    table["pmRA"] = np.ma.filled(table["pmRA"].astype(float), 0.0)
    table["pmDE"] = np.ma.filled(table["pmDE"].astype(float), 0.0)
    # If an object does not have a parallax then we treat it as if the object is an "infinite distance"
    # and set the parallax to 1e-7 arcseconds or 10Mpc.
    table["Plx"] = np.ma.filled(table["Plx"].astype(float), 1e-4)

    # Suppress warning caused by Astropy as noted in issue 111747 (https://github.com/astropy/astropy/issues/11747)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="negative parallaxes")

        # Get the input data from the table
        c = SkyCoord(
            ra=table["RAJ2000"],
            dec=table["DEJ2000"],
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


class CatalogResult(pd.DataFrame):
    def to_SkyCoord(self, equinox: Time = None, epoch: Time = None):
        return _table_to_skycoord(Table.from_pandas(self), equinox=equinox, epoch=epoch)
