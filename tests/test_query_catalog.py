"""Tests catalog querying"""

import numpy as np
from astropy.coordinates import SkyCoord
import pandas as pd
from astropy.time import Time
import astropy.units as u
from lksearch import query_catalog, query_EPIC, query_KIC, query_TIC, query_gaia
import pytest

# Tests the region around TIC 228760807 which should return a catalog containing 4 objects.
c = SkyCoord(194.10141041659, -27.3905828803397, unit="deg")
epoch = Time(1569.4424277786259 + 2457000, scale="tdb", format="jd")


def test_tic():
    catalog = query_catalog(
        coord=c,
        epoch=epoch,
        catalog="tic",
        radius=u.Quantity(80, "arcsecond"),
        magnitude_limit=18,
    )
    assert len(catalog) == 4

    # Checks that an astropy Table is returned
    assert isinstance(catalog, pd.DataFrame)

    # Test that the proper motion works

    assert np.isclose(catalog.iloc[0]["RA"], 194.10075230969787, atol=1e-6)
    assert np.isclose(catalog.iloc[0]["Dec"], -27.390340343480744, atol=1e-6)

    # Test different epochs
    catalog_new = query_catalog(
        coord=c,
        epoch=Time(2461041.500, scale="tt", format="jd"),
        catalog="tic",
        radius=80,
        magnitude_limit=18,
    )

    assert np.isclose(catalog_new.iloc[0]["RA"], 194.10052070792756, atol=1e-6)
    assert np.isclose(catalog_new.iloc[0]["Dec"], -27.390254988629433, atol=1e-6)

    # Test different epochs
    catalog_new = query_TIC(
        coord=c,
        epoch=Time(2461041.500, scale="tt", format="jd"),
        radius=4 * u.pixel,
        magnitude_limit=18,
    )


def test_bad_catalog():
    # test the catalog type i.e., simbad is not included in our catalog list.
    # Look at other tests to see if this is correct syntax
    with pytest.raises(ValueError, match="Can not parse catalog name 'simbad'"):
        query_catalog(
            coord=c, epoch=epoch, catalog="simbad", radius=80, magnitude_limit=18
        )


def test_gaia():
    catalog_gaia = query_catalog(
        coord=c,
        epoch=Time(1569.4424277786259 + 2457000, scale="tdb", format="jd"),
        catalog="gaiadr3",
        radius=80,
        magnitude_limit=18,
    )

    assert len(catalog_gaia) == 2

    catalog_gaia = query_gaia(
        coord=c,
        epoch=Time(1569.4424277786259 + 2457000, scale="tdb", format="jd"),
        radius=80,
        magnitude_limit=18,
    )


def test_kic():
    catalog_kepler = query_catalog(
        coord=SkyCoord(285.679391, 50.2413, unit="deg"),
        epoch=Time(120.5391465105713 + 2454833, scale="tdb", format="jd"),
        catalog="kic",
        radius=20,
        magnitude_limit=18,
    )
    assert len(catalog_kepler) == 5
    catalog_kepler = query_KIC(
        coord=SkyCoord(285.679391, 50.2413, unit="deg"),
        epoch=Time(120.5391465105713 + 2454833, scale="tdb", format="jd"),
        radius=1 * u.pixel,
        magnitude_limit=18,
    )


def test_epic():
    catalog_k2 = query_catalog(
        coord=SkyCoord(172.560465, 7.588391, unit="deg"),
        epoch=Time(1975.1781333280233 + 2454833, scale="tdb", format="jd"),
        catalog="epic",
        radius=20,
        magnitude_limit=18,
    )
    assert len(catalog_k2) == 1
    catalog_k2 = query_EPIC(
        coord=SkyCoord(172.560465, 7.588391, unit="deg"),
        epoch=Time(1975.1781333280233 + 2454833, scale="tdb", format="jd"),
        radius=1 * u.pixel,
        magnitude_limit=18,
    )


def test_empty():
    catalog = query_catalog(
        SkyCoord.from_name("Kepler-10"),
        Time.now(),
        "epic",
        radius=20 * u.arcsecond,
        magnitude_limit=18,
    )
    assert isinstance(catalog, pd.DataFrame)
    assert len(catalog) == 0
