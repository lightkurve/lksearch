"""Tests catalog id querying"""

import numpy as np
from astropy.coordinates import SkyCoord
import pandas as pd
from astropy.time import Time
import astropy.units as u
from lksearch import CatalogSearch
import pytest


def test_id_query():
    tic = 299096513
    tic_result = CatalogSearch.QueryID(tic, catalog="tic")
    assert len(tic_result == 1)
    assert tic_result.TIC.values == tic

    kic = 12644769
    kic_result = CatalogSearch.QueryID(kic, catalog="kic")
    assert len(kic_result == 1)
    assert kic_result.KIC.values == kic

    epic = 201563164
    epic_result = CatalogSearch.QueryID(epic, catalog="epic")
    assert len(epic_result == 1)
    assert epic_result.ID.values == epic

    gaia = 2133452475178900736
    gaia_result = CatalogSearch.QueryID(gaia, catalog="gaiadr3")
    assert len(gaia_result == 1)
    assert gaia_result.Source.values == gaia


def name_disambiguation(string, key, target):
    result = CatalogSearch.QueryID(string)
    return len(result == 1) and result[key].values[0] == target


def test_name_disambiguation():
    tic = 902906874
    assert name_disambiguation(f"TIC {tic}", "TIC", tic)
    assert name_disambiguation(f"TIC{tic}", "TIC", tic)
    assert name_disambiguation(f"tess{tic}", "TIC", tic)
    assert name_disambiguation(f"tess {tic}", "TIC", tic)

    kic = 12644769
    assert name_disambiguation(f"KIC {kic}", "KIC", kic)
    assert name_disambiguation(f"KIC{kic}", "KIC", kic)
    assert name_disambiguation(f"kplr{kic}", "KIC", kic)
    assert name_disambiguation(f"kplr {kic}", "KIC", kic)

    epic = 201563164
    assert name_disambiguation(f"EPIC {epic}", "ID", epic)
    assert name_disambiguation(f"EPIC{epic}", "ID", epic)
    assert name_disambiguation(f"ktwo{epic}", "ID", epic)
    assert name_disambiguation(f"ktwo {epic}", "ID", epic)

    gaiadr3 = 2133452475178900736
    assert name_disambiguation(f"gaiadr3 {gaiadr3 }", "Source", gaiadr3)
    assert name_disambiguation(f"gaiadr3{gaiadr3}", "Source", gaiadr3)
    assert name_disambiguation(f"GAIA{gaiadr3 }", "Source", gaiadr3)
    assert name_disambiguation(f"GAIA {gaiadr3 }", "Source", gaiadr3)


def test_lists():
    sources = [2133452475178900736, 3201680999981276544]
    assert len(CatalogSearch.QueryID(sources, catalog="gaiadr3")) == 2


def test_crossmatch():
    result = CatalogSearch.QueryID("GAIA 2133452475178900736", catalog="tic")
    assert len(result) == 1
    assert result["TIC"].values == 299096513

    sources = [2133452475178900736, 3201680999981276544]
    result = CatalogSearch.QueryID(sources, catalog="tic", input_catalog="gaiadr3")
    assert len(result) == 2
    assert 299096513 in result["TIC"].values
    assert 299096514 in result["TIC"].values

    result = CatalogSearch.QueryID("TIC 299096513", catalog="gaiadr3")
    assert len(result) == 1
    assert result["Source"].values == 2133452475178900736

    result = CatalogSearch.QueryID(f"KIC 12644769", catalog="tic")
    assert len(result) == 1
    assert result["TIC"].values == 299096355
