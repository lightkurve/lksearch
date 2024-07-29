"""Test features of search that interact with the data archive at MAST."""

import os
import pytest

from numpy.testing import assert_almost_equal, assert_array_equal
import numpy as np


from astropy.coordinates import SkyCoord
import astropy.units as u

import pandas as pd


from lksearch.utils import SearchError, SearchWarning

from lksearch import MASTSearch, TESSSearch, KeplerSearch, K2Search
from lksearch import conf


def test_search_cubedata():
    # EPIC 210634047 was observed twice in long cadence
    assert len(K2Search("EPIC 210634047").cubedata.table) == 2
    # ...including Campaign 4
    assert len(K2Search("EPIC 210634047", campaign=4).cubedata.table) == 1
    # KIC 11904151 (Kepler-10) was observed in LC in 15 Quarters
    assert len(KeplerSearch("KIC 11904151", exptime="long").cubedata.table) == 15
    # ...including quarter 11 but not 12:
    assert (
        len(KeplerSearch("KIC 11904151", exptime="long", quarter=11).cubedata.table)
        == 1
    )
    assert (
        len(KeplerSearch("KIC 11904151", exptime="long", quarter=12).cubedata.table)
        == 0
    )

    # with mission='TESS', it should return TESS observations
    tic = "TIC 273985862"  # Has been observed in multiple sectors including 1
    assert len(MASTSearch(tic, mission="TESS").table) > 1
    assert (
        len(
            TESSSearch(
                tic, pipeline="SPOC", sector=1, search_radius=100
            ).timeseries.table
        )
        == 2
    )

    manifest = TESSSearch(tic, pipeline="SPOC", sector=1).download()
    assert len(manifest) == len(TESSSearch(tic, pipeline="SPOC", sector=1))
    assert len(TESSSearch("pi Mensae", sector=1, pipeline="SPOC").cubedata.table) == 1
    # Issue #445: indexing with -1 should return the last index of the search result
    assert len(TESSSearch("pi Mensae").cubedata[-1].table) == 1


def test_search_split_campaigns():
    """Searches should should work for split campaigns.

    K2 Campaigns 9, 10, and 11 were split into two halves for various technical
    reasons (C91=C9a, C92=C9b, C101=C10a, C102=C10b, C111=C11a, C112=C11b).
    We expect most targets from those campaigns to return two TPFs.
    """
    campaigns = [9, 10, 11]
    ids = ["EPIC 228162462", "EPIC 228726301", "EPIC 202975993"]
    for c, idx in zip(campaigns, ids):
        sr = K2Search(idx, campaign=c, exptime="long").cubedata.table
        assert len(sr) == 2


def test_search_timeseries(caplog):
    # We should also be able to resolve targets by name instead of KIC ID
    # The name Kepler-10 somehow no longer works on MAST. So we use 2MASS instead:
    #   https://simbad.cds.unistra.fr/simbad/sim-id?Ident=%405506010&Name=Kepler-10
    assert (
        len(
            KeplerSearch(
                "2MASS J19024305+5014286", pipeline="Kepler", exptime="long"
            ).timeseries.table
        )
        == 15
    )

    # Check there is a NameResolveError if putting in nonsense
    # with pytest.raises(SearchError, match="Unable to find"):
    #    MASTSearch("DOES_NOT_EXIST (UNIT TEST)").timeseries

    # If we ask for all cadence types, there should be four Kepler files given
    assert (
        len(
            KeplerSearch(
                "KIC 4914423", quarter=6, exptime="any", pipeline="Kepler"
            ).timeseries.table
        )
        == 4
    )

    # ...and only one should have long cadence
    assert (
        len(
            KeplerSearch(
                "KIC 4914423", quarter=6, exptime="long", pipeline="Kepler"
            ).timeseries.table
        )
        == 1
    )
    # Should be able to resolve an ra/dec
    assert (
        len(
            KeplerSearch(
                "297.5835, 40.98339", quarter=6, pipeline="Kepler"
            ).timeseries.table
        )
        == 1
    )
    # Should be able to resolve a SkyCoord
    c = SkyCoord("297.5835 40.98339", unit=(u.deg, u.deg))
    search = KeplerSearch(c, quarter=6, pipeline="Kepler").timeseries
    assert len(search.table) == 1
    assert len(search) == 1

    # We should be able to download a light curve
    manifest = search.download()
    assert len(manifest) == 1

    # with mission='TESS', it should return TESS observations
    tic = "TIC 273985862"
    assert len(TESSSearch(tic).timeseries.table) > 1
    assert (
        len(
            TESSSearch(
                tic, pipeline="spoc", sector=1, search_radius=100
            ).timeseries.table
        )
        == 2
    )
    assert len(TESSSearch("pi Mensae", pipeline="SPOC", sector=1).timeseries.table) == 1


def test_search_with_skycoord():
    """Can we pass both names, SkyCoord objects, and coordinate strings?"""
    sr_name = KeplerSearch("KIC 11904151", exptime="long").cubedata
    assert (
        len(sr_name) == 15
    )  # Kepler-10 as observed during 15 quarters in long cadence
    # Can we search using a SkyCoord objects?
    sr_skycoord = KeplerSearch(
        SkyCoord.from_name("KIC 11904151"), exptime="long"
    ).cubedata
    assert len(sr_skycoord) == 15
    assert_array_equal(
        sr_name.table["productFilename"], sr_skycoord.table["productFilename"]
    )
    # Can we search using a string of "ra dec" decimals?
    sr_decimal = KeplerSearch("285.67942179 +50.24130576", exptime="long").cubedata

    assert_array_equal(
        sr_name.table["productFilename"], sr_decimal.table["productFilename"]
    )
    # Can we search using a sexagesimal string?
    sr_sexagesimal = KeplerSearch("19:02:43.1 +50:14:28.7", exptime="long").cubedata
    assert_array_equal(
        sr_name.table["productFilename"], sr_sexagesimal.table["productFilename"]
    )

    sr_kic = KeplerSearch("KIC 11904151", exptime="long").cubedata
    assert_array_equal(
        sr_name.table["productFilename"], sr_kic.table["productFilename"]
    )


def test_searchresult():
    sr = KeplerSearch("KIC 11904151").timeseries
    assert len(sr) == len(sr.table)  # Tests SearchResult.__len__
    assert len(sr[2:7]) == 5  # Tests SearchResult.__get__
    assert len(sr[2]) == 1
    assert "kplr" in sr.__repr__()
    assert "kplr" in sr._repr_html_()


def test_month():
    # In short cadence, if we specify both quarter and month
    sr = KeplerSearch("KIC 11904151", quarter=11, month=1, exptime="short").cubedata
    assert len(sr) == 1
    sr = KeplerSearch(
        "KIC 11904151", quarter=11, month=[1, 3], exptime="short"
    ).cubedata
    assert len(sr) == 2


def test_collections():
    assert len(K2Search("EPIC 205998445", search_radius=900).cubedata.table) == 4
    assert (
        len(
            MASTSearch(
                "EPIC 205998445", mission="K2", search_radius=900, pipeline="K2"
            ).filter_table(limit=3)
        )
        == 3
    )
    # if fewer targets are found than targetlimit, should still download all available
    assert (
        len(
            K2Search("EPIC 205998445", search_radius=900, pipeline="K2")
            .cubedata.filter_table(limit=6)
            .table
        )
        == 4
    )

    assert isinstance(
        MASTSearch(
            "EPIC 205998445", mission="K2", search_radius=900, pipeline="K2"
        ).cubedata.download(),
        pd.DataFrame,
    )


def test_properties():
    c = SkyCoord("297.5835 40.98339", unit=(u.deg, u.deg))
    assert_almost_equal(KeplerSearch(c, quarter=6).cubedata.ra[0], 297.5835)
    assert_almost_equal(KeplerSearch(c, quarter=6).cubedata.dec[0], 40.98339)
    assert len(KeplerSearch(c, quarter=6).cubedata.target_name) == 1


def test_source_confusion():
    # Regression test for issue #148.
    # When obtaining the TPF for target 6507433, @benmontet noticed that
    # a target 4 arcsec away was returned instead.
    # See https://github.com/lightkurve/lightkurve/issues/148
    desired_target = "KIC 6507433"
    tpf = KeplerSearch(desired_target, quarter=8).cubedata
    assert "6507433" in tpf.target_name[0]


def test_empty_searchresult():
    """Does an empty SearchResult behave gracefully?"""
    sr = MASTSearch(table=pd.DataFrame())
    assert len(sr) == 0
    str(sr)
    with pytest.warns(SearchWarning, match="Cannot download"):
        sr.download()


def test_issue_472():
    """Regression test for https://github.com/lightkurve/lightkurve/issues/472"""
    # The line below previously threw an exception because the target was not
    # observed in Sector 2; we're always expecting a SearchResult object (empty
    # or not) rather than an exception.
    # Whether or not this SearchResult is empty has changed over the years,
    # because the target is only ~15 pixels beyond the FFI edge and the accuracy
    # of the FFI footprint polygons at the MAST portal have changed at times.
    with pytest.raises(SearchError, match="No data"):
        TESSSearch("TIC41336498", sector=2).tesscut


""" This test used in OG Lightkurve used the fact that we were failing when we
tried to read the file into memmory after a download call.  In this package we
are never reading a file into memmory so we cannot trivially replicate this test.  
A corrupted file will just exist on disk, but can be over-written with a keyword.  
Is there a good way to replicate this behaviour?  
"""
# def test_corrupt_download_handling_case_empty():
#    """When a corrupt file exists in the cache, make sure the user receives
#    a helpful error message.
#
#    This is a regression test for #511 and #1184.
#
#    For case the file is truncated, see test_read.py::test_file_corrupted
#    It cannot be done easily here because on Windows,
#    a similar test would result in PermissionError when `tempfile`
#    tries to do cleanup.
#    Some low level codes (probably astropy.fits) still hold a file handle
#    of the corrupted FIS file.
#    """
#    with tempfile.TemporaryDirectory() as tmpdirname:
#        # Pretend a corrupt file exists at the expected cache location
#        expected_dir = os.path.join(
#            tmpdirname, "mastDownload", "Kepler", "kplr011904151_lc_Q111111110111011101"
#        )
#        expected_fn = os.path.join(
#            expected_dir, "kplr011904151-2010009091648_lpd-targ.fits.gz"
#        )
#        os.makedirs(expected_dir)
#        open(expected_fn, "w").close()  # create "corrupt" i.e. empty file
#        with pytest.raises(SearchWarning):
#            KeplerSearch("KIC 11904151", quarter=4, exptime="long").cubedata.download(
#                download_dir=tmpdirname
#            )
#        #assert "may be corrupt" in err.value.args[0]
#        #assert expected_fn in err.value.args[0]

# Couldn't get the below to work - I think because we're missing a setattr in MASTSearch.  Look into this.
# def test_mast_http_error_handling(monkeypatch):
#    """Regression test for #1211; ensure downloads yields an warning when MAST download result in an error.
#    This is an intentional change from lightkurve search behaviour - since we are returning the file manifest
#    we now throw a warning not an error"""
#
#    from astroquery.mast import Observations
#
#    result = TESSSearch("TIC 273985862").timeseries
#    remote_url = result.table.loc[0,"dataURI"].values
#
#    def mock_http_error_response(*args, **kwargs):
#        """Mock the `download_product()` response to simulate MAST returns HTTP error"""
#        return Table(data={
#            "Local Path": ["./mastDownload/acme_lc.fits"],
#            "Status": ["ERROR"],
#            "Message": ["HTTP Error 500: Internal Server Error"],
#            "URL": [remote_url],
#            })
#
#    monkeypatch.setattr(Observations, "download_products", mock_http_error_response)
#
#    with tempfile.TemporaryDirectory() as tmpdirname:
#        # ensure the we don't hit cache so that it'll always download from MAST
#        with pytest.raises(SearchWarning) as excinfo:
#            result[0].download(download_dir=tmpdirname)
#        assert "HTTP Error 500" in str(excinfo.value)
#        assert remote_url in str(excinfo.value)


def test_indexerror_631():
    """Regression test for #631; avoid IndexError."""
    # This previously triggered an exception:
    result = TESSSearch(
        "KIC 8462852", sector=15, search_radius=1, pipeline="spoc"
    ).timeseries
    assert len(result) == 1


def test_overlapping_targets_718():
    """Regression test for #718."""
    # Searching for the following targets without radius should only return
    # the requested targets, not their overlapping neighbors.
    targets = ["KIC 5112705", "KIC 10058374", "KIC 5385723"]
    for target in targets:
        search = KeplerSearch(target, quarter=11, pipeline="Kepler").timeseries
        assert len(search) == 1
        assert search.target_name[0] == f"kplr{target[4:].zfill(9)}"

    # When using `radius=1` we should also retrieve the overlapping targets
    search = KeplerSearch(
        "KIC 5112705", quarter=11, pipeline="Kepler", search_radius=1 * u.arcsec
    ).timeseries
    assert len(search) > 1


def test_tesscut_795():
    """Regression test for #795: make sure the __repr__.of a TESSCut
    SearchResult works."""
    str(TESSSearch("KIC 8462852"))


def test_exptime_filtering():
    """Can we pass "fast", "short", exposure time to the exptime argument?"""

    res = TESSSearch("AU Mic", sector=27, exptime="fast").timeseries
    assert len(res) == 1
    assert res.exptime[0] == 20.0

    res = TESSSearch("AU Mic", sector=27, exptime="short").timeseries
    assert len(res) == 1
    assert res.table["exptime"][0] == 120.0

    res = TESSSearch("AU Mic", sector=27, exptime=20).timeseries
    assert len(res) == 1
    assert res.table["exptime"][0] == 20.0
    assert "fast" in res.table["productFilename"][0]


def test_search_slicing_regression():
    # Regression test: slicing after calling __repr__ failed.
    res = TESSSearch("AU Mic", exptime=20).timeseries
    res.__repr__()
    res[res.exptime[0] < 100]


def test_ffi_hlsp():
    """Can SPOC, QLP (FFI), and TESS-SPOC (FFI) light curves be accessed?"""
    search = TESSSearch("TrES-2b", sector=26).timeseries  # aka TOI 2140.01
    assert search.table["pipeline"].str.contains("QLP").any()
    assert search.table["pipeline"].str.contains("TESS-SPOC").any()
    assert search.table["pipeline"].str.contains("SPOC").any()
    # tess-spoc also produces tpfs
    search = TESSSearch("TrES-2b", sector=26).cubedata
    assert search.table["pipeline"].str.contains("TESS-SPOC").any()
    assert search.table["pipeline"].str.contains("SPOC").any()


def test_qlp_ffi_lightcurve():
    """Can we search and download an MIT QLP FFI light curve?"""
    search = TESSSearch("TrES-2b", sector=26, pipeline="qlp").timeseries
    assert len(search) == 1
    assert search.pipeline[0] == "QLP"
    assert search.exptime[0] == 1800  # * u.second  # Sector 26 had 30-minute FFIs


def test_spoc_ffi_lightcurve():
    """Can we search and download a SPOC FFI light curve?"""
    search = TESSSearch("TrES-2b", sector=26, pipeline="tess-spoc").timeseries
    assert len(search) == 1
    assert search.pipeline[0] == "TESS-SPOC"
    assert search.exptime[0] == 1800  # * u.second  # Sector 26 had 30-minute FFIs


def test_split_k2_campaigns():
    """Do split K2 campaign sections appear separately in search results?"""
    # Campaign 9
    search_c09 = K2Search("EPIC 228162462", exptime="long", campaign=9).cubedata
    assert search_c09.table["campaign"][0] == "09a"
    assert search_c09.table["campaign"][1] == "09b"
    # Campaign 10

    search_c10 = K2Search("EPIC 228725972", exptime="long", campaign=10).cubedata
    assert search_c10.table["campaign"][0] == "10a"
    assert search_c10.table["campaign"][1] == "10b"
    # Campaign 11
    search_c11 = K2Search("EPIC 203830112", exptime="long", campaign=11).cubedata
    assert search_c11.table["campaign"][0] == "11a"
    assert search_c11.table["campaign"][1] == "11b"


def test_FFI_retrieval():
    """Can we find TESS individual FFI's"""
    assert len(TESSSearch("Kepler 16b").search_sector_ffis(14)) == 1241


def test_tesscut():
    """Can we find and download TESS tesscut tpfs"""
    results = TESSSearch("Kepler 16b", hlsp=False, sector=14)
    assert len(results) == 9
    assert len(results.cubedata) == 2
    manifest = results.cubedata[1].download()
    assert len(manifest) == 1


class TestMASTSearchFilter:
    results = MASTSearch("Kepler 16b")

    @pytest.mark.parametrize(
        "target_name",
        (
            0,
            299096355,
            "kplr012644769",
            [299096355, "kplr012644769"],
            [299096355, "299096355"],
        ),
    )
    def test_target_name(self, target_name):
        self.results.filter_table(target_name=target_name)

    @pytest.mark.parametrize("limit", (0, 10, 1000))
    def test_limit(self, limit):
        self.results.filter_table(limit=limit)

    @pytest.mark.parametrize("filetype", (0, "lightcurve"))
    def test_filetype(self, filetype):
        self.results.filter_table(filetype=filetype)

    @pytest.mark.parametrize(
        "exptime",
        (
            0,
            20,
            20.0,
            [0, 20],
            [20, 60.0],
            (0, 100),
            "fast",
            "short",
            "long",
            "shortest",
            "longest",
        ),
    )
    def test_exptime(self, exptime):
        self.results.filter_table(exptime=exptime)

    @pytest.mark.parametrize("distance", (0, 0.2, (0.2, 0.4)))
    def test_distance(self, distance):
        self.results.filter_table(distance=distance)

    @pytest.mark.parametrize("year", (0, 2013, (2000, 2020), [2013, 2019]))
    def test_year(self, year):
        self.results.filter_table(year=year)

    @pytest.mark.parametrize(
        "description", (0, "data", ["TPS", "report"], ("TPS", "report"))
    )
    def test_description(self, description):
        self.results.filter_table(description=description)

    @pytest.mark.parametrize("pipeline", (0, "Kepler", "spoc", ["kepler", "spoc"]))
    def test_pipeline(self, pipeline):
        self.results.filter_table(pipeline=pipeline)

    @pytest.mark.parametrize("sequence", (0, 14, [14, 15]))
    def test_sequence(self, sequence):
        self.results.filter_table(sequence=sequence)

    @pytest.mark.parametrize("mission", (0, "Kepler", "Tess", ["Kepler", "Tess"]))
    def test_mission(self, mission):
        self.results.filter_table(mission=mission)

    def test_combination(
        self,
    ):
        filter_results = self.results.filter_table(
            target_name=299096355,
            pipeline="SPOC",
            mission="TESS",
            exptime=120,
            distance=0.1,
            year=2022,
            description="light curves",
            filetype="lightcurve",
            sequence=55,
            limit=10,
        )
        assert (
            filter_results.table.obs_id.values[0]
            == "tess2022217014003-s0055-0000000299096355-0242-s"
        )


def test_filter():
    """Can we properly filter the data"""

    results = TESSSearch("Kepler 16b")
    filtered = results.filter_table(sector=14)
    queried = results.query_table("sector == 14")
    assert len(filtered) == len(queried)


def test_tess_clouduris():
    """regression test - do tesscut/nan's in dataURI column break cloud uri fetching"""
    toi = TESSSearch("TOI 1161", sector=14)
    # 17 products should be returned
    assert len(toi.cloud_uris) == 17
    # 5 of them should have cloud uris
    assert (
        np.sum(
            ([cloud_uri is not None for cloud_uri in toi.cloud_uris.values]).astype(int)
        )
        == 5
    )


def test_tess_return_clouduri_not_download():
    """Test to see if we return a S3 bucket instead of downloading if
    `~conf.DOWNLOAD_CLOUD` = False
    """
    # reload the config, set download_cloud = False
    conf.reload()
    conf.DOWNLOAD_CLOUD = False
    # Try to download a file without a S3 bucket, and one with
    # Search for TESS data only. This by default includes both HLSPs and FFI cutouts.
    toi = TESSSearch("TOI 1161", sector=14)
    uris = toi.dvreports.cloud_uris
    not_cloud = pd.isna(uris)
    # A DV Report is not on the cloud - this should still get downloaded locally
    dvr = toi.dvreports[not_cloud]
    dvr_man = dvr[0].download()
    assert os.path.isfile(dvr_man["Local Path"][0])
    # A SPOC TPF is on the cloud, this should return a S3 bucket
    mask = toi.timeseries.pipeline == "SPOC"
    lc_man = toi.timeseries[mask].download()
    assert lc_man["Local Path"][0][0:5] == "s3://"
