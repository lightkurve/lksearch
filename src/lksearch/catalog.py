"""Functions to search various catalogs for missions"""

from typing import Union
import numpy as np
from astropy.coordinates import Angle, SkyCoord, Distance
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from astroquery.vizier import Vizier
import pandas as pd

import warnings

__all__ = ["query_KIC", "query_EPIC", "query_TIC", "query_gaia", "query_catalog"]

# This is a lits of VizieR catalogs and their input parameters to be used in the
# query_skycatalog function
_Catalog_Dictionary = {
    "kic": {
        "catalog": "V/133/kic",
        "columns": ["KIC", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx", "kepmag"],
        "column_filters": "kepmag",
        "rename_in": ["KIC", "kepmag"],
        "rename_out": ["ID", "Kepmag"],
        "equinox": Time(2000, format="jyear", scale="tt"),
        "prefix": "KIC",
        "default_mag": "Kepmag",
    },
    "epic": {
        "catalog": "IV/34/epic",
        "columns": ["ID", "RAJ2000", "DEJ2000", "pmRA", "pmDEC", "plx", "Kpmag"],
        "column_filters": "Kpmag",
        "rename_in": ["Kpmag", "pmDEC", "plx"],
        "rename_out": ["K2mag", "pmDE", "Plx"],
        "equinox": Time(2000, format="jyear", scale="tt"),
        "prefix": "EPIC",
        "default_mag": "K2mag",
    },
    "tic": {
        "catalog": "IV/39/tic82",
        "columns": ["TIC", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx", "Tmag"],
        "column_filters": "Tmag",
        "rename_in": ["TIC", "Tmag"],
        "rename_out": ["ID", "TESSmag"],
        "equinox": Time(2000, format="jyear", scale="tt"),
        "prefix": "TIC",
        "default_mag": "TESSmag",
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
        ],
        "column_filters": "Gmag",
        "rename_in": ["DR3Name"],
        "rename_out": ["ID"],
        "equinox": Time(2016, format="jyear", scale="tt"),
        "prefix": None,
        "default_mag": "Gmag",
    },
}


def _table_to_skycoord(table: Table, equinox: Time, epoch: Time):
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


def _get_return_columns(columns):
    """Convenience function to reorder columns and remove motion columns."""
    return [
        "RA",
        "Dec",
        "Separation",
        "Relative_Flux",
        *list(
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
        ),
    ]


def query_catalog(
    coord: Union[SkyCoord, str],
    epoch: Time,
    catalog: str,
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

    # Check to make sure that user input is in the correct format
    if not isinstance(coord, SkyCoord):
        if isinstance(coord, str):
            coord = SkyCoord.from_name(coord)
        raise TypeError("Must pass an `astropy.coordinates.SkyCoord` object.")
    if not isinstance(epoch, Time):
        try:
            epoch = Time(epoch, format="jd")
        except ValueError:
            raise TypeError("Must pass an `astropy.time.Time object`.")
        raise TypeError("Must pass an `astropy.time.Time object`.")
    if not coord.isscalar:
        raise ValueError("Pass one target only.")

    # Here we check to make sure that the radius entered is in arcseconds
    # This also means we do not need to specify arcseconds in our catalog query
    try:
        radius = u.Quantity(radius, "arcsecond")
    except u.UnitConversionError:
        raise

    # Check to make sure that the catalog provided by the user is valid for this function
    if catalog.lower() not in _Catalog_Dictionary.keys():
        raise ValueError(f"Can not parse catalog name '{catalog}'")
    catalog_meta = _Catalog_Dictionary[catalog.lower()]

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

    c = _table_to_skycoord(table=result, equinox=catalog_meta["equinox"], epoch=epoch)
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
    return result[_get_return_columns(result.columns)]


def query_KIC(
    coord: SkyCoord,
    epoch: Time = Time(2000, format="jyear", scale="tt"),
    radius: Union[float, u.Quantity] = u.Quantity(2, "pixel"),
    magnitude_limit: float = 18.0,
    return_skycoord: bool = False,
):
    """
    Query the Kepler Input Catalog for a single source location, obtain nearby sources

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD. Defaults to J2000.
    catalog: str
        The catalog to query, either 'kepler', 'k2', or 'tess', 'gaia'
    radius : float or astropy quantity
        Radius in arcseconds to query. Defaults to 2 pixels.
    magnitude_limit : float
        A value to limit the results in based on the Tmag/Kepler mag/K2 mag or Gaia G mag. Default, 18.
    return_skycoord: bool
        Whether to return an astropy.coordinates.SkyCoord object. Default is False.

    Returns
    -------
    result: pd.DataFrame
        A pandas dataframe of the sources within radius query, corrected for proper motion
    """
    if radius.unit == u.pixel:
        radius = (radius * (4 * u.arcsecond / u.pixel)).to(u.arcsecond)
    return query_catalog(
        coord=coord,
        epoch=epoch,
        catalog="kic",
        radius=radius,
        magnitude_limit=magnitude_limit,
        return_skycoord=return_skycoord,
    )


def query_TIC(
    coord: SkyCoord,
    epoch: Time = Time(2000, format="jyear", scale="tt"),
    radius: Union[float, u.Quantity] = u.Quantity(2, "pixel"),
    magnitude_limit: float = 18.0,
    return_skycoord: bool = False,
):
    """
    Query the TESS Input Catalog for a single source location, obtain nearby sources

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD. Defaults to J2000.
    catalog: str
        The catalog to query, either 'kepler', 'k2', or 'tess', 'gaia'
    radius : float or astropy quantity
        Radius in arcseconds to query. Defaults to 2 pixels.
    magnitude_limit : float
        A value to limit the results in based on the Tmag/Kepler mag/K2 mag or Gaia G mag. Default, 18.
    return_skycoord: bool
        Whether to return an astropy.coordinates.SkyCoord object. Default is False.

    Returns
    -------
    result: pd.DataFrame
        A pandas dataframe of the sources within radius query, corrected for proper motion
    """
    if radius.unit == u.pixel:
        radius = (radius * (21 * u.arcsecond / u.pixel)).to(u.arcsecond)
    return query_catalog(
        coord=coord,
        epoch=epoch,
        catalog="tic",
        radius=radius,
        magnitude_limit=magnitude_limit,
        return_skycoord=return_skycoord,
    )


def query_EPIC(
    coord: SkyCoord,
    epoch: Time = Time(2000, format="jyear", scale="tt"),
    radius: Union[float, u.Quantity] = u.Quantity(2, "pixel"),
    magnitude_limit: float = 18.0,
    return_skycoord: bool = False,
):
    """
    Query the Ecliptic Plane Input Catalog for a single source location, obtain nearby sources

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD. Defaults to J2000.
    catalog: str
        The catalog to query, either 'kepler', 'k2', or 'tess', 'gaia'
    radius : float or astropy quantity
        Radius in arcseconds to query. Defaults to 2 pixels.
    magnitude_limit : float
        A value to limit the results in based on the Tmag/Kepler mag/K2 mag or Gaia G mag. Default, 18.
    return_skycoord: bool
        Whether to return an astropy.coordinates.SkyCoord object. Default is False.

    Returns
    -------
    result: pd.DataFrame
        A pandas dataframe of the sources within radius query, corrected for proper motion
    """
    if radius.unit == u.pixel:
        radius = (radius * (4 * u.arcsecond / u.pixel)).to(u.arcsecond)
    return query_catalog(
        coord=coord,
        epoch=epoch,
        catalog="epic",
        radius=radius,
        magnitude_limit=magnitude_limit,
        return_skycoord=return_skycoord,
    )


def query_gaia(
    coord: SkyCoord,
    epoch: Time = Time(2016, format="jyear", scale="tt"),
    radius: Union[float, u.Quantity] = u.Quantity(10, "arcsecond"),
    magnitude_limit: float = 18.0,
    return_skycoord: bool = False,
):
    """
    Query the Gaia EDR3 catalog for a single source location, obtain nearby sources

    Parameters
    ----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD. Defaults to the epoch of Gaia DR3 (J2016).
    catalog: str
        The catalog to query, either 'kepler', 'k2', or 'tess', 'gaia'
    radius : float or astropy quantity
        Radius in arcseconds to query. Defaults to 10 arcseconds.
    magnitude_limit : float
        A value to limit the results in based on the Tmag/Kepler mag/K2 mag or Gaia G mag. Default, 18.
    return_skycoord: bool
        Whether to return an astropy.coordinates.SkyCoord object. Default is False.

    Returns
    -------
    result: pd.DataFrame
        A pandas dataframe of the sources within radius query, corrected for proper motion
    """
    return query_catalog(
        coord=coord,
        epoch=epoch,
        catalog="gaiadr3",
        radius=radius,
        magnitude_limit=magnitude_limit,
        return_skycoord=return_skycoord,
    )
