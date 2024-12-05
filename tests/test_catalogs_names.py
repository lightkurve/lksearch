"""Tests alternate name lookups and matching"""

from lksearch.catalogsearch import query_names, match_names_catalogs


def test_query_names():
    res = query_names("Gaia DR3 3796414192429498880")
    assert "EPIC 201563164" in res.id.values

    list_res = query_names(["Kepler 10", "Kepler 16"])
    assert len(list_res) == 2


def test_match_names_catalogs():
    res = match_names_catalogs(["Kepler 10", "Kepler 16"], match=["tic", "kic", "epic"])

    assert len(res.columns) == 4
    assert len(res["search"]) == 2
    assert res["epic"][0] == ""
