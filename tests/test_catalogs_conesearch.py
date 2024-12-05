"""Tests catalog querying"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from lksearch.catalogsearch import query_region
import pytest

# Tests the region around TIC 228760807 which should return a catalog containing 4 objects.
c = SkyCoord(194.10141041659, -27.3905828803397, unit="deg")
epoch = Time(1569.4424277786259 + 2457000, scale="tdb", format="jd")


def test_tic():
    catalog = query_region(
        c,
        output_epoch=epoch,
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
    catalog_new = query_region(
        c,
        output_epoch=Time(2461041.500, scale="tt", format="jd"),
        catalog="tic",
        radius=80,
        magnitude_limit=18,
    )

    assert np.isclose(catalog_new.iloc[0]["RA"], 194.10052070792756, atol=1e-6)
    assert np.isclose(catalog_new.iloc[0]["Dec"], -27.390254988629433, atol=1e-6)


def test_bad_catalog():
    # test the catalog type i.e., simbad is not included in our catalog list.
    # Look at other tests to see if this is correct syntax
    with pytest.raises(ValueError, match="Can not parse catalog name 'simbad'"):
        query_region(
            c, output_epoch=epoch, catalog="simbad", radius=80, magnitude_limit=18
        )


def test_gaia_position():
    catalog_gaia = query_region(
        c,
        output_epoch=Time(1569.4424277786259 + 2457000, scale="tdb", format="jd"),
        catalog="gaiadr3",
        radius=80,
        magnitude_limit=18,
    )

    assert len(catalog_gaia) == 2

    catalog_gaia = query_region(
        c,
        output_epoch=Time(1569.4424277786259 + 2457000, scale="tdb", format="jd"),
        radius=80,
        magnitude_limit=18,
        catalog="gaiadr3",
    )


def test_kic():
    catalog_kepler = query_region(
        SkyCoord(285.679391, 50.2413, unit="deg"),
        output_epoch=Time(120.5391465105713 + 2454833, scale="tdb", format="jd"),
        catalog="kic",
        radius=20,
        magnitude_limit=18,
    )
    assert len(catalog_kepler) == 5


def test_epic():
    catalog_k2 = query_region(
        SkyCoord(172.560465, 7.588391, unit="deg"),
        output_epoch=Time(1975.1781333280233 + 2454833, scale="tdb", format="jd"),
        catalog="epic",
        radius=20,
        magnitude_limit=18,
    )
    assert len(catalog_k2) == 1


def test_empty():
    catalog = query_region(
        SkyCoord.from_name("Kepler-10"),
        Time.now(),
        catalog="epic",
        radius=20 * u.arcsecond,
        magnitude_limit=18,
    )
    assert isinstance(catalog, pd.DataFrame)
    assert len(catalog) == 0


def test_resolving():
    catalog = query_region("Kepler 10", catalog="tic")
    assert np.isclose(catalog["RA"][0], 285.679422)
    assert np.isclose(catalog["Dec"][0], 50.241306)

    catalog = query_region("19h02m43.03s +50d14m29.34s", catalog="tic")
    assert np.isclose(catalog["RA"][0], 285.679422)
    assert np.isclose(catalog["Dec"][0], 50.241306)

    catalog = query_region("285.679422 50.241306", catalog="tic")
    assert np.isclose(catalog["RA"][0], 285.679422)
    assert np.isclose(catalog["Dec"][0], 50.241306)

    catalog = query_region((285.679422, 50.241306), catalog="tic")
    assert np.isclose(catalog["RA"][0], 285.679422)
    assert np.isclose(catalog["Dec"][0], 50.241306)


def test_skycoord():
    sc = query_region(
        c,
        output_epoch=epoch,
        catalog="tic",
        radius=u.Quantity(80, "arcsecond"),
        magnitude_limit=18,
        return_skycoord=True,
    )
    assert isinstance(sc, SkyCoord)
