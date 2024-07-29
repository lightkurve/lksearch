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

__all__ = ['query_KIC', 'query_EPIC', 'query_TIC', 'query_gaia', 'query_catalog']

# This is a lits of VizieR catalogs and their input parameters to be used in the
# query_skycatalog function
_Catalog_Dictionary = {
    "kic": {
        "catalog": "V/133/kic",
        "columns": ["KIC", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx", "kepmag"],
        "column_filters": "kepmag",
        "rename_in": ("KIC", "pmDE", "kepmag"),
        "rename_out": ("ID", "pmDEC", "Kepler_Mag"),
        "equinox":Time(2000, format="jyear", scale="tt"),
        "prefix":"KIC"
    },
    "epic": {
        "catalog": "IV/34/epic",
        "columns": ["ID", "RAJ2000", "DEJ2000", "pmRA", "pmDEC", "plx", "Kpmag"],
        "column_filters": "Kpmag",
        "rename_in": ("Kpmag", "plx"),
        "rename_out": ("K2_Mag", 'Plx'),
        "equinox":Time(2000, format="jyear", scale="tt"),
        "prefix":"EPIC"
    },
    "tic": {
        "catalog": "IV/39/tic82",
        "columns": ["TIC", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx", "Tmag"],
        "column_filters": "Tmag",
        "rename_in": ("TIC", "pmDE", "Tmag"),
        "rename_out": ("ID", "pmDEC", "TESS_Mag"),
        "equinox":Time(2000, format="jyear", scale="tt"),
        "prefix":"TIC"
    },
    "gaiadr3": {
        "catalog": "I/355/gaiadr3",
        "columns": ["DR3Name", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "Plx", "Gmag"],
        "column_filters": "Gmag",
        "rename_in": ("DR3Name", "pmDE", "Gmag"),
        "rename_out": ("ID", "pmDEC", "Gaia_G_Mag"),
        "equinox":Time(2016, format="jyear", scale="tt"),
        "prefix":None,
    },
}


def _apply_propermotion(table:Table, equinox:Time, epoch:Time):
    """
    Returns an astropy table of sources with the proper motion correction applied

    Parameters:
    -----------
    table :
        astropy.table.Table which contains the coordinates of targets and proper motion values
    equinox: astropy.time.Time
        The equinox for the catalog
    epoch : astropy.time.Time
        Time of the observation - This is taken from the table R.A and Dec. values and re-formatted as an astropy.time.Time object

    Returns:
    ------
    table : astropy.table.Table
        Returns an astropy table with ID, corrected RA, corrected Dec, and Mag(?Some ppl might find this benifical for contamination reasons?)
    """

    # We need to remove any nan values from our proper  motion list
    # Doing this will allow objects which do not have proper motion to still be displayed
    table["pmRA"] = np.ma.filled(table["pmRA"].astype(float), 0.0)
    table["pmDEC"] = np.ma.filled(table["pmDEC"].astype(float), 0.0)
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
            pm_dec=table["pmDEC"],
            frame="icrs",
            obstime=equinox,
        )

    # Suppress warning caused by Astropy as noted in issue 111747 (https://github.com/astropy/astropy/issues/11747)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="ERFA function")
        warnings.filterwarnings("ignore", message="invalid value")
        
        # Calculate the new values
        c1 = c.apply_space_motion(new_obstime=epoch)

    # Add new data corrected RA and Dec
    table["RA"] = c1.ra.to(u.deg)
    table["DEC"] = c1.dec.to(u.deg)

    # Get the index of the targets with zero proper motions
    pmzero_index = np.where((table["pmRA"] == 0.0) & (table["pmDEC"] == 0.0))

    # In those instances replace with J2000 values
    table["RA"][pmzero_index] = table["RAJ2000"][pmzero_index]
    table["DEC"][pmzero_index] = table["DEJ2000"][pmzero_index]
    return table



def query_catalog(
    coord: SkyCoord,
    epoch: Time,
    catalog: str,
    radius: Union[float, u.Quantity] = u.Quantity(100, "arcsecond"),
    magnitude_limit: float = 18.0,
):
    """Query a catalog for a single source location, obtain nearby sources

    Parameters:
    -----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD. 
    catalog: str
        The catalog to query, either 'kepler', 'k2', or 'tess', 'gaia'
    radius : float or astropy quantity
        Radius in arcseconds to query
    magnitude_limit : float
        A value to limit the results in based on the Tmag/Kepler mag/K2 mag or Gaia G mag. Default, 18.

    Returns:
    -------
    result: pd.DataFrame
        A pandas dataframe of the sources within radius query, corrected for proper motion
    """

    # Check to make sure that user input is in the correct format
    if not isinstance(coord, SkyCoord):
        raise TypeError("Must pass an `astropy.coordinates.SkyCoord` object.")
    if not isinstance(epoch, Time):
        try:
            epoch = Time(epoch, format='jd')
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
        column_filters={
            catalog_meta[
                "column_filters"
            ]: f"<{magnitude_limit}"
        },
    )

    # The catalog can cut off at 50 - we dont want this to happen
    filters.ROW_LIMIT = -1
    # Now query the catalog
    result = filters.query_region(coord, catalog=catalog_name, radius=Angle(radius))
    if len(result) == 0:
        return pd.DataFrame(columns=[*catalog_meta["columns"], 'RA', 'DEC', 'Separation', 'Relative_Flux']).rename({i:o for i, o in zip(catalog_meta["rename_in"], catalog_meta["rename_out"])}, axis=1).set_index("ID")
    
    result = result[catalog_name]

    # Rename the columns so that the output is uniform
    result.rename_columns(
        catalog_meta["rename_in"],
        catalog_meta["rename_out"],
    )

    if catalog_meta['prefix'] is not None:
        prefix = catalog_meta["prefix"]
        result["ID"] = [f"{prefix} {id}"for id in result['ID']]

    # Based on the input coordinates pick the object with the mininmum separation as the reference star.
    c1 = SkyCoord(result["RAJ2000"], result["DEJ2000"], unit="deg")

    sep = coord.separation(c1)

    # Find the object with the minimum separation - this is our target
    ref_index = np.argmin(sep)

    # apply_propermotion
    result = _apply_propermotion(result, equinox=catalog_meta["equinox"], epoch=epoch)

    # Now we want to repete but using the values corrected for proper motion
    # First get the correct values for target
    coord_pm_correct = SkyCoord(
        result["RA"][ref_index], result["DEC"][ref_index], unit="deg"
    )

    c1_pm_correct = SkyCoord(result["RA"], result["DEC"], unit="deg")

    # Then calculate the separation based on pm corrected values
    sep_pm_correct = coord_pm_correct.separation(c1_pm_correct)

    # Provide the separation in the output table
    result["Separation"] = sep_pm_correct

    # Calculate the relative flux
    result["Relative_Flux"] = 10**(
        (
            [value for key, value in result.items() if "_Mag" in key][0]
            - [value for key, value in result.items() if "_Mag" in key][0][ref_index]
        )
        / -2.5
    )

    # Now sort the table based on separation
    result.sort(["Separation"])
    return result.to_pandas().set_index("ID")

def query_KIC(        
    coord: SkyCoord,
    epoch: Time,
    radius: Union[float, u.Quantity] = u.Quantity(100, "arcsecond"),
    magnitude_limit: float = 18.0,
):
    """Query the Kepler Input Catalog for a single source location, obtain nearby sources

    Parameters:
    -----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD. 
    catalog: str
        The catalog to query, either 'kepler', 'k2', or 'tess', 'gaia'
    radius : float or astropy quantity
        Radius in arcseconds to query
    magnitude_limit : float
        A value to limit the results in based on the Tmag/Kepler mag/K2 mag or Gaia G mag. Default, 18.
    Returns:
    -------
    result: pd.DataFrame
        A pandas dataframe of the sources within radius query, corrected for proper motion
    """
    if radius.unit == u.pixel:
        radius = (radius * (4 * u.arcsecond/u.pixel)).to(u.arcsecond)
    return query_catalog(coord=coord, epoch=epoch, catalog='kic', radius=radius, magnitude_limit=magnitude_limit)

def query_TIC(
    coord: SkyCoord,
    epoch: Time,
    radius: Union[float, u.Quantity] = u.Quantity(100, "arcsecond"),
    magnitude_limit: float = 18.0,
):
    """Query the TESS Input Catalog for a single source location, obtain nearby sources

    Parameters:
    -----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD. 
    catalog: str
        The catalog to query, either 'kepler', 'k2', or 'tess', 'gaia'
    radius : float or astropy quantity
        Radius in arcseconds to query
    magnitude_limit : float
        A value to limit the results in based on the Tmag/Kepler mag/K2 mag or Gaia G mag. Default, 18.
    Returns:
    -------
    result: pd.DataFrame
        A pandas dataframe of the sources within radius query, corrected for proper motion
    """
    if radius.unit == u.pixel:
        radius = (radius * (21 * u.arcsecond/u.pixel)).to(u.arcsecond)
    return query_catalog(coord=coord, epoch=epoch, catalog='tic', radius=radius, magnitude_limit=magnitude_limit)

def query_EPIC(
    coord: SkyCoord,
    epoch: Time,
    radius: Union[float, u.Quantity] = u.Quantity(100, "arcsecond"),
    magnitude_limit: float = 18.0,
):
    """Query the Ecliptic Plane Input Catalog for a single source location, obtain nearby sources

    Parameters:
    -----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD. 
    catalog: str
        The catalog to query, either 'kepler', 'k2', or 'tess', 'gaia'
    radius : float or astropy quantity
        Radius in arcseconds to query
    magnitude_limit : float
        A value to limit the results in based on the Tmag/Kepler mag/K2 mag or Gaia G mag. Default, 18.
    Returns:
    -------
    result: pd.DataFrame
        A pandas dataframe of the sources within radius query, corrected for proper motion
    """
    if radius.unit == u.pixel:
        radius = (radius * (4 * u.arcsecond/u.pixel)).to(u.arcsecond)
    return query_catalog(coord=coord, epoch=epoch, catalog='epic', radius=radius, magnitude_limit=magnitude_limit)

def query_gaia(
    coord: SkyCoord,
    epoch: Time,
    radius: Union[float, u.Quantity] = u.Quantity(100, "arcsecond"),
    magnitude_limit: float = 18.0,
):
    """Query the Gaia EDR3 catalog for a single source location, obtain nearby sources

    Parameters:
    -----------
    coord : astropy.coordinates.SkyCoord
        Coordinates around which to do a radius query
    epoch: astropy.time.Time
        The time of observation in JD. 
    catalog: str
        The catalog to query, either 'kepler', 'k2', or 'tess', 'gaia'
    radius : float or astropy quantity
        Radius in arcseconds to query
    magnitude_limit : float
        A value to limit the results in based on the Tmag/Kepler mag/K2 mag or Gaia G mag. Default, 18.
    Returns:
    -------
    result: pd.DataFrame
        A pandas dataframe of the sources within radius query, corrected for proper motion
    """
    return query_catalog(coord=coord, epoch=epoch, catalog='gaiadr3', radius=radius, magnitude_limit=magnitude_limit)