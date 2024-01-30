"""Defines tools to retrieve 1D- and 2D- time series data from the archive at MAST."""

from __future__ import division

import glob
import logging
import os
import re
import warnings

from typing import Union
import numpy.typing as npt

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
#from astropy.io import ascii
from astropy.table import Row, Table, join
from astropy.time import Time
from astropy.utils import deprecated
from memoization import cached
from requests import HTTPError
import pandas as pd


from . import PACKAGEDIR, conf, config
from lightkurve.collections import LightCurveCollection, TargetPixelFileCollection
from lightkurve.io import read
from lightkurve.targetpixelfile import TargetPixelFile
from lightkurve.utils import  (
    LightkurveDeprecationWarning,
    LightkurveError,
    LightkurveWarning,
    suppress_stdout,
)

log = logging.getLogger(__name__)


__all__ = [
    "search_targetpixelfile",
    "search_lightcurve",
    "search_lightcurvefile",
    "search_tesscut",
    "SearchResult",
]


# Which external links should we display in the SearchResult repr?
# Do we need need to maintain this?  Probably Yes.  Can we automate this for HLSPs in a MAST standard? 

AUTHOR_LINKS = {
    "Kepler": "https://archive.stsci.edu/kepler/data_products.html",
    "K2": "https://archive.stsci.edu/k2/data_products.html",
    "SPOC": "https://heasarc.gsfc.nasa.gov/docs/tess/pipeline.html",
    "TESS-SPOC": "https://archive.stsci.edu/hlsp/tess-spoc",
    "QLP": "https://archive.stsci.edu/hlsp/qlp",
    "TASOC": "https://archive.stsci.edu/hlsp/tasoc",
    "PATHOS": "https://archive.stsci.edu/hlsp/pathos",
    "CDIPS": "https://archive.stsci.edu/hlsp/cdips",
    "K2SFF": "https://archive.stsci.edu/hlsp/k2sff",
    "EVEREST": "https://archive.stsci.edu/hlsp/everest",
    "TESScut": "https://mast.stsci.edu/tesscut/",
    "GSFC-ELEANOR-LITE": "https://archive.stsci.edu/hlsp/gsfc-eleanor-lite",
    "TGLC": "https://archive.stsci.edu/hlsp/tglc",
    "KBONUS-BKG":"https://archive.stsci.edu/hlsp/kbonus-bkg",
}

REPR_COLUMNS_BASE = [
    "#",
    "mission",
    "year",
    "author",
    "exptime",
    "target_name",
    "distance",
]


class SearchError(Exception):
    pass


class SearchResult(object):
    """Container for the results returned by the search functions.

    The purpose of this class is to provide a convenient way to inspect and
    download products that have been identified using one of the data search
    functions.

    Parameters
    ----------
    table : `~astropy.table.Table` object
        Astropy table returned by a join of the astroquery `Observations.query_criteria()`
        and `Observations.get_product_list()` methods.
    """

    table = None
    """`~astropy.table.Table` containing the full search results returned by the MAST API."""

    display_extra_columns = []
    """A list of extra columns to be included in the default display of the search result.
    It can be configured in a few different ways.

    For example, to include ``proposal_id`` in the default display, users can set it:

    1. in the user's ``lightkurve.cfg`` file::

        [search]
        # The extra comma at the end is needed for a single extra column
        search_result_display_extra_columns = proposal_id,

    2. at run time::

        import lightkurve as lk
        lk.conf.search_result_display_extra_columns = ['proposal_id']

    3. for a specific `SearchResult` object instance::

        result.display_extra_columns = ['proposal_id']

    See :ref:`configuration <api.config>` for more information.
    """

    def __init__(self, table=None):
        if table is None:
            self.table = pd.DataFrame()

        else:
            if isinstance(table, Table):
                log.warning("Search Result Now Expects a pandas dataframe but an astropy Table was given; converting astropy.table to pandas")
                table = table.to_pandas()
            self.table = table
            if len(table) > 0:
                self._add_columns()
                self._sort_table()
        self.display_extra_columns = conf.search_result_display_extra_columns

    def _sort_table(self):
        """Sort the table of search results by distance, author, and filename.

        The reason we include "author" in the sort criteria is that Lightkurve v1 only
        showed data products created by the official pipelines (i.e. author equal to
        "Kepler", "K2", or "SPOC"). To maintain backwards compatibility, we want to
        show products from these authors at the top, so that `search.download()`
        operations tend to download the same product in Lightkurve v1 vs v2.
        This ordering is not a judgement on the quality of one product vs another,
        because we love all pipelines!
        """
        sort_priority = {"Kepler": 1, 
                         "K2": 1, 
                         "SPOC": 1, 
                         "TESS-SPOC": 2, 
                         "QLP": 3}
        self.table["sort_order"] = [
            sort_priority.get(author, 9) for author in self.table["author"]
        ]
        self.table.sort_values(by=["distance", "project", "sort_order", "start_time", "exptime"])

    def _add_columns(self):
        """Adds a user-friendly index (``#``) column and adds column unit
        and display format information.
        """
        '''
        Will eliminate this, depending on how we do units
        if "#" not in self.table.columns:
            self.table["#"] = None
        self.table["exptime"].unit = "s"
        self.table["exptime"].format = ".0f"
        self.table["distance"].unit = "arcsec"'''

        # Add the year column from `t_min` or `productFilename`
        self.table['year'] = self.table['start_time'].dt.year #Assumes 'start_time' is a pandas datetime object
  
        # `t_min` is incorrect for Kepler products, so we extract year from the filename for those =(
        for idx in np.where(self.table["author"] == "Kepler")[0]:
            self.table["year"][idx] = re.findall(
                r"\d+.(\d{4})\d+", self.table["productFilename"][idx]
            )[0]

    def __repr__(self, html=False):
        def to_tess_gi_url(proposal_id):
            if re.match("^G0[12].+", proposal_id) is not None:
                return f"https://heasarc.gsfc.nasa.gov/docs/tess/approved-programs-primary.html#:~:text={proposal_id}"
            elif re.match("^G0[34].+", proposal_id) is not None:
                return f"https://heasarc.gsfc.nasa.gov/docs/tess/approved-programs-em1.html#:~:text={proposal_id}"
            else:
                return f"https://heasarc.gsfc.nasa.gov/docs/tess/approved-programs.html#:~:text={proposal_id}"

        out = "SearchResult containing {} data products.".format(len(self.table))
        if len(self.table) == 0:
            return out
        columns = REPR_COLUMNS_BASE
        if self.display_extra_columns is not None:
            columns = REPR_COLUMNS_BASE + self.display_extra_columns
        # search_tesscut() has fewer columns, ensure we don't try to display columns that do not exist
        columns = [c for c in columns if c in self.table.colnames]

        self.table["#"] = [idx for idx in range(len(self.table))]
        out += "\n\n" + "\n".join(self.table[columns].pformat(max_width=300, html=html))
        # Make sure author names show up as clickable links
        if html:
            for author, url in AUTHOR_LINKS.items():
                out = out.replace(f">{author}<", f"><a href='{url}'>{author}</a><")
            # special HTML formating for TESS proposal_id
            tess_table = self.table[self.table["project"] == "TESS"]
            if "proposal_id" in tess_table.colnames:
                proposal_id_col = np.unique(tess_table["proposal_id"])
            else:
                proposal_id_col = []
            for p_ids in proposal_id_col:
                # for CDIPS products, proposal_id is a np MaskedConstant, not a string
                if p_ids == "N/A" or (not isinstance(p_ids, str)):
                    continue
                # e.g., handle cases with multiple proposals, e.g.,  G12345_G67890
                p_id_links = [
                    f"""\
<a href='{to_tess_gi_url(p_id)}'>{p_id}</a>\
"""
                    for p_id in p_ids.split("_")
                ]
                out = out.replace(f">{p_ids}<", f">{' , '.join(p_id_links)}<")
        return out

    def _repr_html_(self):
        return self.__repr__(html=True)

    def __getitem__(self, key):
        """Implements indexing and slicing."""
        selection = self.table.loc[key]

        return SearchResult(table=selection)

    def __len__(self):
        """Returns the number of products in the SearchResult table."""
        return len(self.table)

    @property
    def unique_targets(self):
        """Returns a table of targets and their RA & dec values produced by search"""
        mask = ["target_name", "s_ra", "s_dec"]
        return self.table[mask].drop_duplicates("target_name").reset_index(drop=True)
        
    @property
    def obsid(self):
        """MAST observation ID for each data product found."""
        return np.asarray(np.unique(self.table["obsid"]), dtype="int64")

    @property
    def ra(self):
        """Right Ascension coordinate for each data product found."""
        return self.table["s_ra"].values

    @property
    def dec(self):
        """Declination coordinate for each data product found."""
        return self.table["s_dec"].values

    @property
    def mission(self):
        """Kepler quarter or TESS sector names for each data product found."""
        return self.table["mission"].values

    @property
    def year(self):
        """Year the observation was made."""
        return self.table["year"].values

    @property
    def author(self):
        """Pipeline name for each data product found."""
        return self.table["author"].values

    @property
    def target_name(self):
        """Target name for each data product found."""
        return self.table["target_name"].values

    @property
    def exptime(self):
        """Exposure time for each data product found."""
        # TODO: return with quantities
        return self.table["exptime"].values 

    @property
    def distance(self):
        """Distance from the search position for each data product found."""
        # TODO: return with quantities
        return self.table["distance"].values

    # @magiccachedecorator
    def _download_one(
        self, 
        table,
        quality_bitmask, 
        download_dir, 
        cutout_size, 
        **kwargs
    ):
        """Private method used by `download()` and `download_all()` to download
        exactly one file from the MAST archive.

        Always returns a `TargetPixelFile` or `LightCurve` object.
        """
        # Make sure astroquery uses the same level of verbosity
        logging.getLogger("astropy").setLevel(log.getEffectiveLevel())

        if download_dir is None:
            download_dir = self._default_download_dir()

        # if the SearchResult row is a TESScut entry, then download cutout
        if "FFI Cutout" in table[0]["description"]:
            try:
                log.debug(
                    "Started downloading TESSCut for '{}' sector {}."
                    "".format(table[0]["target_name"], table[0]["sequence_number"])
                )
                path = self._fetch_tesscut_path(
                    table.loc[0]["target_name"],
                    table.loc[0]["sequence_number"],
                    download_dir,
                    cutout_size,
                )
            except Exception as exc:
                msg = str(exc)
                if "504" in msg:
                    # TESSCut will occasionally return a "504 Gateway Timeout
                    # error" when it is overloaded.
                    raise HTTPError(
                        "The TESS FFI cutout service at MAST appears "
                        "to be temporarily unavailable. It returned "
                        "the following error: {}".format(exc)
                    )
                else:
                    raise SearchError(
                        "Unable to download FFI cutout. Desired target "
                        "coordinates may be too near the edge of the FFI."
                        "Error: {}".format(exc)
                    )

            return read(
                path, quality_bitmask=quality_bitmask, targetid=table.loc[0]["targetid"]
            )

        else:
            if cutout_size is not None:
                warnings.warn(
                    "`cutout_size` can only be specified for TESS "
                    "Full Frame Image cutouts.",
                    LightkurveWarning,
                )
            # Whenever `astroquery.mast.Observations.download_products` is called,
            # a HTTP request will be sent to determine the length of the file
            # prior to checking if the file already exists in the local cache.
            # For performance, we skip this HTTP request and immediately try to
            # find the file in the cache.  The path we check here is consistent
            # with the one hard-coded inside `astroquery.mast.Observations._download_files()`
            # in Astroquery v0.4.1.  It would be good to submit a PR to astroquery
            # so we can avoid having to use this hard-coded hack.
            path = os.path.join(
                download_dir.rstrip("/"),
                "mastDownload",
                table["obs_collection"][0],
                table["obs_id"][0],
                table["productFilename"][0],
            )
            if os.path.exists(path):
                log.debug("File found in local cache.")
            else:
                from astroquery.mast import Observations

                download_url = table[:1]["dataURI"][0]
                log.debug("Started downloading {}.".format(download_url))
                download_response = Observations.download_products(
                    table[:1], mrp_only=False, download_dir=download_dir
                )[0]
                if download_response["Status"] != "COMPLETE":
                    raise LightkurveError(
                        f"Download of {download_url} failed. "
                        f"MAST returns {download_response['Status']}: {download_response['Message']}"
                    )
                path = download_response["Local Path"]
                log.debug("Finished downloading.")
            return read(path, quality_bitmask=quality_bitmask, **kwargs)

    @suppress_stdout
    def download(
        self, 
        quality_bitmask: Union[str, int]="default", 
        download_dir: str =  None, 
        cutout_size: Union[int, tuple[int]]=None, 
        **kwargs
    ):
        """Download and open the first data product in the search result.

        If multiple files are present in `SearchResult.table`, only the first
        will be downloaded.

        Parameters
        ----------
        quality_bitmask : str or int, optional
            Bitmask (integer) which identifies the quality flag bitmask that should
            be used to mask out bad cadences. If a string is passed, it has the
            following meaning:

                * "none": no cadences will be ignored
                * "default": cadences with severe quality issues will be ignored
                * "hard": more conservative choice of flags to ignore
                  This is known to remove good data.
                * "hardest": removes all data that has been flagged
                  This mask is not recommended.

            See the :class:`KeplerQualityFlags <lightkurve.utils.KeplerQualityFlags>` or :class:`TessQualityFlags <lightkurve.utils.TessQualityFlags>` class for details on the bitmasks.
        download_dir : str, optional
            Location where the data files will be stored.
            If `None` is passed, the value from `cache_dir` configuration parameter is used,
            with "~/.lightkurve/cache" as the default.

            See `~lightkurve.config.get_cache_dir()` for details.
        cutout_size : int, float or tuple, optional
            Side length of cutout in pixels. Tuples should have dimensions (y, x).
            Default size is (5, 5)
        flux_column : str, optional
            The column in the FITS file to be read as `flux`. Defaults to 'pdcsap_flux'.
            Typically 'pdcsap_flux' or 'sap_flux'.
        kwargs : dict, optional
            Extra keyword arguments passed on to the file format reader function.

        Returns
        -------
        data : `TargetPixelFile` or `LightCurve` object
            The first entry in the products table.

        Raises
        ------
        HTTPError
            If the TESSCut service times out (i.e. returns HTTP status 504).
        SearchError
            If any other error occurs.

        """
        if len(self.table) == 0:
            warnings.warn(
                "Cannot download from an empty search result.", LightkurveWarning
            )
            return None
        if len(self.table) != 1:
            warnings.warn(
                "Warning: {} files available to download. "
                "Only the first file has been downloaded. "
                "Please use `download_all()` or specify additional "
                "criteria (e.g. quarter, campaign, or sector) "
                "to limit your search.".format(len(self.table)),
                LightkurveWarning,
            )

        return self._download_one(
            table=self.table[:1],
            quality_bitmask=quality_bitmask,
            download_dir=download_dir,
            cutout_size=cutout_size,
            **kwargs,
        )

    @suppress_stdout
    def download_all(
        self, quality_bitmask="default", download_dir=None, cutout_size=None, **kwargs
    ):
        """Download and open all data products in the search result.

        This method will return a `~lightkurve.TargetPixelFileCollection` or
        `~lightkurve.LightCurveCollection`.

        Parameters
        ----------
        quality_bitmask : str or int, optional
            Bitmask (integer) which identifies the quality flag bitmask that should
            be used to mask out bad cadences. If a string is passed, it has the
            following meaning:

                * "none": no cadences will be ignored
                * "default": cadences with severe quality issues will be ignored
                * "hard": more conservative choice of flags to ignore
                  This is known to remove good data.
                * "hardest": removes all data that has been flagged
                  This mask is not recommended.

            See the :class:`KeplerQualityFlags <lightkurve.utils.KeplerQualityFlags>` or :class:`TessQualityFlags <lightkurve.utils.TessQualityFlags>` class for details on the bitmasks.
        download_dir : str, optional
            Location where the data files will be stored.
            If `None` is passed, the value from `cache_dir` configuration parameter is used,
            with "~/.lightkurve/cache" as the default.

            See `~lightkurve.config.get_cache_dir()` for details.
        cutout_size : int, float or tuple, optional
            Side length of cutout in pixels. Tuples should have dimensions (y, x).
            Default size is (5, 5)
        flux_column : str, optional
            The column in the FITS file to be read as `flux`. Defaults to 'pdcsap_flux'.
            Typically 'pdcsap_flux' or 'sap_flux'.
        kwargs : dict, optional
            Extra keyword arguments passed on to the file format reader function.

        Returns
        -------
        collection : `~lightkurve.collections.Collection` object
            Returns a `~lightkurve.LightCurveCollection` or
            `~lightkurve.TargetPixelFileCollection`,
            containing all entries in the products table

        Raises
        ------
        HTTPError
            If the TESSCut service times out (i.e. returns HTTP status 504).
        SearchError
            If any other error occurs.
        """
        if len(self.table) == 0:
            warnings.warn(
                "Cannot download from an empty search result.", LightkurveWarning
            )
            return None
        log.debug("{} files will be downloaded.".format(len(self.table)))

        products = []
        for idx in range(len(self.table)):
            products.append(
                self._download_one(
                    table=self.table[idx : idx + 1],
                    quality_bitmask=quality_bitmask,
                    download_dir=download_dir,
                    cutout_size=cutout_size,
                    **kwargs,
                )
            )
        if isinstance(products[0], TargetPixelFile):
            return TargetPixelFileCollection(products)
        else:
            return LightCurveCollection(products)

    def _default_download_dir(self):
        return config.get_cache_dir()

    def _fetch_tesscut_path(self, target, sector, download_dir, cutout_size):
        """Downloads TESS FFI cutout and returns path to local file.

        Parameters
        ----------
        download_dir : str
            Path to location of `.lightkurve-cache` directory where downloaded
            cutouts are stored
        cutout_size : int, float or tuple
            Side length of cutout in pixels. Tuples should have dimensions (y, x).
            Default size is (5, 5)

        Returns
        -------
        path : str
            Path to locally downloaded cutout file
        """
        from astroquery.mast import TesscutClass

        coords = _resolve_object(target)

        # Set cutout_size defaults
        if cutout_size is None:
            cutout_size = 5

        # Check existence of `~/.lightkurve-cache/tesscut`
        tesscut_dir = os.path.join(download_dir, "tesscut")
        if not os.path.isdir(tesscut_dir):
            # if it doesn't exist, make a new cache directory
            try:
                os.mkdir(tesscut_dir)
            # downloads into default cache if OSError occurs
            except OSError:
                tesscut_dir = download_dir

        # Resolve SkyCoord of given target
        coords = _resolve_object(target)

        # build path string name and check if it exists
        # this is necessary to ensure cutouts are not downloaded multiple times
        sec = TesscutClass().get_sectors(coordinates=coords)
        sector_name = sec[sec["sector"] == sector]["sectorName"][0]
        if isinstance(cutout_size, int):
            size_str = str(int(cutout_size)) + "x" + str(int(cutout_size))
        elif isinstance(cutout_size, tuple) or isinstance(cutout_size, list):
            size_str = str(int(cutout_size[1])) + "x" + str(int(cutout_size[0]))

        # search cache for file with matching ra, dec, and cutout size
        # ra and dec are searched within 0.001 degrees of input target
        ra_string = str(coords.ra.value)
        dec_string = str(coords.dec.value)
        matchstring = r"{}_{}*_{}*_{}_astrocut.fits".format(
            sector_name,
            ra_string[: ra_string.find(".") + 4],
            dec_string[: dec_string.find(".") + 4],
            size_str,
        )
        cached_files = glob.glob(os.path.join(tesscut_dir, matchstring))

        # if any files exist, return the path to them instead of downloading
        if len(cached_files) > 0:
            path = cached_files[0]
            log.debug("Cached file found.")
        # otherwise the file will be downloaded
        else:
            cutout_path = TesscutClass().download_cutouts(
                coordinates=coords, size=cutout_size, sector=sector, path=tesscut_dir
            )
            path = cutout_path[0][0]  # the cutoutpath already contains testcut_dir
            log.debug("Finished downloading.")
        return path


@deprecated(
    "2.0", alternative="search_cubedata()", warning_type=LightkurveDeprecationWarning
)
def search_targetpixelfile(*args, **kwargs):
    product="Target Pixel"
    return search_cubedata(*args,product=product, **kwargs)

@deprecated(
    "2.0", alternative="search_cubedata()", warning_type=LightkurveDeprecationWarning
)
def search_tesscut(*args, **kwargs):
    product="ffi"
    mission="TESS"
    return search_cubedata(*args,product=product,mission=mission, **kwargs)


@cached
def search_cubedata(
    target:  Union[str, int, SkyCoord],
    radius:  Union[float, u.Quantity] = None,
    exptime:  Union[str, int, tuple] = None,
    cadence: Union[str, int, tuple] = None,
    mission: Union[str, tuple] = ("Kepler", "K2", "TESS"),
    product: Union[str, tuple[str]] = ("Target Pixel","ffi"),
    author:  Union[str, tuple] = None,
    quarter:  Union[int, list[int]] = None,
    month:    Union[int, list[int]] = None,
    campaign: Union[int, list[int]] = None,
    sector:   Union[int, list[int]] = None,
    limit:    int = None,
):
    
    try:
        return _search_products(
            target,
            radius=radius,
            filetype=product,
            exptime=exptime or cadence,
            mission=mission,
            provenance_name=author,
            quarter=quarter,
            month=month,
            campaign=campaign,
            sector=sector,
            limit=limit,
        )
    except SearchError as exc:
        log.error(exc)
        return SearchResult(None)

@deprecated(
    "2.0", alternative="search_timeseries()", warning_type=LightkurveDeprecationWarning
)
def search_lightcurvefile(*args, **kwargs):
    return search_lightcurve(*args, **kwargs)

@deprecated(
    "2.0", alternative="search_timeseries()", warning_type=LightkurveDeprecationWarning
)
def search_lightcurve(*args, **kwargs):
    return search_timeseries(*args, **kwargs)

@cached
def search_timeseries(
    target:  Union[str, int, SkyCoord],
    radius:  Union[float, u.Quantity] = None,
    exptime:  Union[str, int, tuple] = None,
    cadence: Union[str, int, tuple] = None,
    mission: Union[str, tuple] = ("Kepler", "K2", "TESS"),
    author:  Union[str, tuple] = None,
    quarter:  Union[int, list[int]] = None,
    month:    Union[int, list[int]] = None,
    campaign: Union[int, list[int]] = None,
    sector:   Union[int, list[int]] = None,
    limit:    int = None,
):
    """Search the `MAST data archive <https://archive.stsci.edu>`_ for light curves.

    This function fetches a data table that lists the Light Curve Files
    that fall within a region of sky centered around the position of `target`
    and within a cone of a given `radius`. If no value is provided for `radius`,
    only a single target will be returned.

    Parameters
    ----------
    target : str, int, or `astropy.coordinates.SkyCoord` object
        Target around which to search. Valid inputs include:

            * The name of the object as a string, e.g. "Kepler-10".
            * The KIC or EPIC identifier as an integer, e.g. 11904151.
            * A coordinate string in decimal format, e.g. "285.67942179 +50.24130576".
            * A coordinate string in sexagesimal format, e.g. "19:02:43.1 +50:14:28.7".
            * An `astropy.coordinates.SkyCoord` object.
    radius : float or `astropy.units.Quantity` object
        Conesearch radius.  If a float is given it will be assumed to be in
        units of arcseconds.  If `None` then we default to 0.0001 arcsec.
    exptime : 'long', 'short', 'fast', or float
        'long' selects 10-min and 30-min cadence products;
        'short' selects 1-min and 2-min products;
        'fast' selects 20-sec products.
        Alternatively, you can pass the exact exposure time in seconds as
        an int or a float, e.g., ``exptime=600`` selects 10-minute cadence.
        By default, all cadence modes are returned.
    cadence : 'long', 'short', 'fast', or float
        Synonym for `exptime`. This keyword will likely be deprecated in the future.
    mission : str, tuple of str
        'Kepler', 'K2', or 'TESS'. By default, all will be returned.
    author : str, tuple of str, or "any"
        Author of the data product (`provenance_name` in the MAST API).
        Official Kepler, K2, and TESS pipeline products have author names
        'Kepler', 'K2', and 'SPOC'.
        Community-provided products that are supported include 'K2SFF', 'EVEREST'.
        By default, all light curves are returned regardless of the author.
    quarter, campaign, sector : int, list of ints
        Kepler Quarter, K2 Campaign, or TESS Sector number.
        By default all quarters/campaigns/sectors will be returned.
    month : 1, 2, 3, 4 or list of int
        For Kepler's prime mission, there are three short-cadence
        TargetPixelFiles for each quarter, each covering one month.
        Hence, if ``exptime='short'`` you can specify month=1, 2, 3, or 4.
        By default all months will be returned.
    limit : int
        Maximum number of products to return.

    Returns
    -------
    result : :class:`SearchResult` object
        Object detailing the data products found.

    Examples
    --------
    This example demonstrates how to use the `search_lightcurve()` function to
    query and download data. Before instantiating a `LightCurve` object or
    downloading any science products, we can identify potential desired targets with
    `search_lightcurve`::

        >>> from lightkurve import search_lightcurve  # doctest: +SKIP
        >>> search_result = search_lightcurve("Kepler-10")  # doctest: +SKIP
        >>> print(search_result)  # doctest: +SKIP

    The above code will query mast for lightcurve files available for the known
    planet system Kepler-10, and display a table containing the available
    data products. Because Kepler-10 was observed in multiple quarters and sectors
    by both Kepler and TESS, the search will return many dozen results.
    If we want to narrow down the search to only return Kepler light curves
    in long cadence, we can use::

        >>> search_result = search_lightcurve("Kepler-10", author="Kepler", exptime=1800)   # doctest: +SKIP
        >>> print(search_result)  # doctest: +SKIP

    That is better, we now see 15 light curves corresponding to 15 Kepler quarters.
    If we want to download a `~lightkurve.collections.LightCurveCollection` object containing all
    15 observations, use::

        >>> search_result.download_all()  # doctest: +SKIP

    or we can specify the downloaded products by selecting a specific row using
    rectangular brackets, for example::

        >>> lc = search_result[2].download()  # doctest: +SKIP

    The above line of code will only search and download Quarter 2 data and
    create a `LightCurve` object called lc.

    We can also pass a radius into `search_lightcurve` to perform a cone search::

        >>> search_lightcurve('Kepler-10', radius=100, quarter=4, exptime=1800)  # doctest: +SKIP

    This will display a table containing all targets within 100 arcseconds of
    Kepler-10 and in Quarter 4.  We can then download a
    `~lightkurve.collections.LightCurveFile` containing all these
    light curves using::

        >>> search_lightcurve('Kepler-10', radius=100, quarter=4, exptime=1800).download_all()  # doctest: +SKIP
    """
    try:
        return _search_products(
            target,
            radius=radius,
            filetype="Lightcurve",
            exptime=exptime or cadence,
            mission=mission,
            provenance_name=author,
            quarter=quarter,
            month=month,
            campaign=campaign,
            sector=sector,
            limit=limit,
        )
    except SearchError as exc:
        log.error(exc)
        return SearchResult(None)


def _search_products(
    target : Union[str, SkyCoord],
    radius : Union[float, u.Quantity] = None,
    filetype : Union[str, list[str]] = "Lightcurve",
    mission : Union[str, list[str]] = ("Kepler", "K2", "TESS"),
    provenance_name : Union[str, list[str]] = None,
    exptime : Union[float, tuple[float]]=(0, 9999),
    quarter : Union[int, list[int]] = None,
    month : Union[int, list[int]] = None,
    campaign : Union[int, list[int]] = None,
    sector : Union[int, list[int]] = None,
    limit : int = None,
    **extra_query_criteria,
):
    """Helper function which returns a SearchResult object containing MAST
    products that match several criteria.

    Parameters
    ----------
    target : string or `astropy.coordinates.SkyCoord` object
        See docstrings above.
    radius : float or `astropy.units.Quantity` object
        Conesearch radius.  If a float is given it will be assumed to be in
        units of arcseconds.  If `None` then we default to 0.0001 arcsec.
    filetype : {'Target pixel', 'Lightcurve', 'FFI'}
        Type of files queried at MAST.
    exptime : 'long', 'short', 'fast', or float
        'long' selects 10-min and 30-min cadence products;
        'short' selects 1-min and 2-min products;
        'fast' selects 20-sec products.
        Alternatively, you can pass the exact exposure time in seconds as
        an int or a float, e.g., ``exptime=600`` selects 10-minute cadence.
        By default, all cadence modes are returned.
    mission : str, list of str
        'Kepler', 'K2', or 'TESS'. By default, all will be returned.
    provenance_name : str, list of str
        Provenance of the data product. Defaults to official products, i.e.
        ('Kepler', 'K2', 'SPOC').  Community-provided products such as 'K2SFF'
        are supported as well.
    quarter, campaign, sector : int, list of ints
        Kepler Quarter, K2 Campaign, or TESS Sector number.
        By default all quarters/campaigns/sectors will be returned.
        Only quarter OR campain OR sector can be provided for a given search
    month : 1, 2, 3, 4 or list of int
        For Kepler's prime mission, there are three short-cadence
        TargetPixelFiles for each quarter, each covering one month.
        Hence, if ``exptime='short'`` you can specify month=1, 2, 3, or 4.
        By default all months will be returned.
    limit : int
        Maximum number of products to return

    Returns
    -------
    SearchResult : :class:`SearchResult` object.
    """
    if isinstance(target, int):
        raise TypeError("Target must be a target name string or astropy coordinate object")
    # Specifying quarter, campaign, or quarter should constrain the mission
    # Can we specify multiple?   No, throw an error if so

    if [bool(quarter), 
        bool(campaign),
        bool(sector)].count(True) > 1:
       raise LightkurveError("Ambiguity Error; multiple quarter/campaign/sector specified accross different missions."
                    "If searching for specific data across different missions, perform separate searches")

    # if a quarter/campaign/sector is specified, search only that mission
    if quarter is not None:
        mission = "Kepler"
    if campaign is not None:
        mission = "K2"
    if sector is not None:
        mission = "TESS"

    sequence = quarter or campaign or sector

    # Ensure mission is a list
    mission = np.atleast_1d(mission).tolist()

    # Avoid filtering on `provenance_name` if `author` equals "any" or "all"
    if provenance_name in ("any", "all") or provenance_name is None:
        provenance_name = None
    else:
        provenance_name = np.atleast_1d(provenance_name).tolist()

    # Speed up by restricting the MAST query if we don't want FFI image data
    extra_query_criteria = {}
    if not any(['ffi' in ftype.lower() for ftype in filetype]):
        # At MAST, non-FFI Kepler pipeline products are known as "cube" products,
        # and non-FFI TESS pipeline products are listed as "timeseries".
        extra_query_criteria["dataproduct_type"] = ["cube", "timeseries"]

    # Query Mast to get a list of observations
    observations = _query_mast(
        target,
        radius=radius,
        project=mission,
        provenance_name=provenance_name,
        exptime=exptime,
        sequence_number=sequence,
        **extra_query_criteria,
    )
    log.debug(
        f"MAST found {len(observations)} observations. "
        "Now querying MAST for the corresponding data products."
    )
    if len(observations) == 0:
        raise SearchError(f"No data found for target {target}.")

    #First, define empty dataframes to simplify our join between ffi & TPF later
    masked_result = pd.DataFrame()
    ffi_result = pd.DataFrame()

    # Light curves and target pixel files have similar query structure
    if  set(["Lightcurve", "Target Pixel"]) & set(np.atleast_1d(filetype)):
        from astroquery.mast import Observations

        products = Observations.get_product_list(observations)
        joint_table = join(
            observations,
            products,
            keys="obs_id",
            join_type="right",
            uniq_col_name="{col_name}{table_name}",
            table_names=["", "_products"],
        )
        return joint_table

        joint_table = joint_table.to_pandas()

        # Add the user-friendly 'author' column (synonym for 'provenance_name')
        joint_table["author"] = joint_table["provenance_name"]
        # Add the user-friendly 'mission' column
        joint_table["smission"] = None
        obs_prefix = {"Kepler": "Quarter", "K2": "Campaign", "TESS": "Sector"}

        #Carry over all sequence numbers where they exist
        seq_num = joint_table["sequence_number"].values.astype(str)

        mask = joint_table["sequence_number"].notna()
        seq_num[mask] = [f"{item[0]:02d}" if item else "" for item in joint_table.loc[mask,["sequence_number"]].values]

        # Kepler sequence_number values were not populated at the time of
        # writing this code, so we parse them from the description field.
        mask = ((joint_table["project"] == "Kepler") &
                joint_table["sequence_number"].isna())
        re_expr = r".*Q(\d+)"
        seq_num[mask] = [re.findall(re_expr, item[0])[0] if re.findall(re_expr, item[0]) else "" for item in joint_table.loc[mask,["description"]].values]

        # K2 campaigns 9, 10, and 11 were split into two sections, which are
        # listed separately in the table with suffixes "a" and "b"        
        mask = ((joint_table["project"] == "K2") & 
                (joint_table["sequence_number"].values in [9, 10, 11]))
        for item in joint_table.loc[mask,:]:
            for half, letter in zip([1,2],["a","b"]):
                if f"c{item['sequence_number']}{half}" in joint_table["productFilename"][idx]:
                    seq_num[item.index.values] = f"{int(item['sequence_number']):02d}{letter}"

        joint_table["mission"] = [f" {proj} {pref} {seq}" 
                                  for proj, pref, seq in zip(
                                      joint_table['project'], 
                                      obs_prefix.get(joint_table['project'], ''),
                                      seq_num)]
        
        masked_result = _filter_products(
            joint_table,
            filetype=filetype,
            campaign=campaign,
            quarter=quarter,
            exptime=exptime,
            project=mission,
            provenance_name=provenance_name,
            month=month,
            sector=sector,
            limit=limit,
        )
        log.debug(f"MAST found {len(masked_result)} matching data products.")
        masked_result["distance"].info.format = ".1f"  # display <0.1 arcsec

    # Full Frame Images - build this from the querry table
    if any(['ffi' in ftype.lower() for ftype in filetype]):
    # Make sure `search_tesscut` always performs a cone search (i.e. always
    # passed a radius value), because strict target name search does not apply.
        if radius is None:
            radius = 0.0001 * u.arcsec

        cutouts = []
        for idx in np.where(["TESS FFI" in t for t in observations["target_name"]])[0]:
            # if target passed in is a SkyCoord object, convert to RA, dec pair
            if isinstance(target, SkyCoord):
                target = "{}, {}".format(target.ra.deg, target.dec.deg)
            # pull sector numbers
            s = observations["sequence_number"][idx]
            # if the desired sector is available, add a row
            if s in np.atleast_1d(sector) or sector is None:
                cutouts.append(
                    {
                        "description": f"TESS FFI Cutout (sector {s})",
                        "mission": f"TESS Sector {s:02d}",
                        "target_name": str(target),
                        "targetid": str(target),
                        "t_min": observations["t_min"][idx],
                        "exptime": observations["exptime"][idx],
                        "productFilename": "TESScut",
                        "provenance_name": "TESScut",
                        "author": "TESScut",
                        "distance": 0.0,
                        "sequence_number": s,
                        "project": "TESS",
                        "obs_collection": "TESS",
                    }
                )
        if len(cutouts) > 0:
            log.debug("Found {} matching cutouts.".format(len(cutouts)))
            ffi_result = Table(cutouts)
            ffi_result = ffi_result.to_pandas()

    query_result = pd.concat([masked_result,
                              ffi_result]).sort_values(["distance", 
                                                        "obsid", 
                                                        "sequence_number"])
    return(SearchResult(query_result))

def _query_mast(
    target: Union[str, SkyCoord],
    radius: Union[float, u.Quantity, None] = None,
    project: Union[str, list[str]] = ["Kepler", "K2", "TESS"],
    provenance_name: Union[str, list[str], None] = None,
    exptime: Union[int, float, tuple] = (0, 9999),
    sequence_number: Union[int, list[int], None] = None,
    **extra_query_criteria,

) -> Table:
    """Helper function which wraps `astroquery.mast.Observations.query_criteria()`
    to return a table of all project [Kepler/K2/TESS] observations of a given target.

    By default only the official data products are returned, but this can be
    adjusted by adding alternative data product names into `provenance_name`.

    Parameters
    ----------
    target : str, or `astropy.coordinates.SkyCoord` object
        See docstrings above.
    radius : float or `astropy.units.Quantity` object
        Conesearch radius.  If a float is given it will be assumed to be in
        units of arcseconds.  If `None` then we default to 0.0001 arcsec.
    project : str, list of str
        Mission name.  Typically 'Kepler', 'K2', or 'TESS'.
        This parameter is case-insensitive.
    provenance_name : str, list of str
        Provenance of the observation.  Common options include 'Kepler', 'K2',
        'SPOC', 'K2SFF', 'EVEREST', 'KEPSEISMIC'.
        This parameter is case-insensitive.
    exptime : (float, float) tuple or float
        if tuple, Exposure time range in seconds. Common values include `(59, 61)`
        for Kepler short cadence and `(1799, 1801)` for Kepler long cadence.
        If float, search for exact exposure time
    sequence_number : int, list of int
        Quarter (Kepler) Campaign (K2), or Sector (TESS) number.
    **extra_query_criteria : kwargs
        Extra criteria to be passed to `astroquery.mast.Observations.query_criteria`.
        See https://mast.stsci.edu/api/v0/_c_a_o_mfields.html

    Returns
    -------
    obs : astropy.Table
        Table detailing the available observations on MAST.
    """
    # Local astroquery import because the package is not used elsewhere
    from astroquery.exceptions import NoResultsWarning, ResolverError
    from astroquery.mast import Observations

    # If passed a SkyCoord, convert it to an "ra, dec" string for MAST
    if isinstance(target, SkyCoord):
        target = "{}, {}".format(target.ra.deg, target.dec.deg)

    # We pass the following `query_criteria` to MAST regardless of whether
    # we search by position or target name:
    query_criteria = {"project": project, **extra_query_criteria}
    if provenance_name is not None:
        query_criteria["provenance_name"] = provenance_name
    if sequence_number is not None:
        query_criteria["sequence_number"] = sequence_number
    if exptime is not None:
        query_criteria["t_exptime"] = exptime

    # If an exact KIC ID is passed, we will search by the exact `target_name`
    # under which MAST will know the object to prevent source confusion.
    # For discussion, see e.g. GitHub issues #148, #718.
    exact_target_name = None
    target_lower = str(target).lower()
    # Was a Kepler target ID passed?
    kplr_match = re.match(r"^(kplr|kic) ?(\d+)$", target_lower)
    if kplr_match:
        exact_target_name = f"kplr{kplr_match.group(2).zfill(9)}"
    # Was a K2 target ID passed?
    ktwo_match = re.match(r"^(ktwo|epic) ?(\d+)$", target_lower)
    if ktwo_match:
        exact_target_name = f"ktwo{ktwo_match.group(2).zfill(9)}"
    # Was a TESS target ID passed?
    tess_match = re.match(r"^(tess|tic) ?(\d+)$", target_lower)
    if tess_match:
        exact_target_name = f"{tess_match.group(2).zfill(9)}"

    if exact_target_name and radius is None:
        log.debug(
            "Started querying MAST for observations with the exact "
            f"target_name='{exact_target_name}'."
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NoResultsWarning)
            warnings.filterwarnings("ignore", message="t_exptime is continuous")
            obs = Observations.query_criteria(
                target_name=exact_target_name, **query_criteria
            )
        if len(obs) > 0:
            # We use `exptime` as an alias for `t_exptime`
            obs["exptime"] = obs["t_exptime"]
            # astroquery does not report distance when querying by `target_name`;
            # we add it here so that the table returned always has this column.
            obs["distance"] = 0.0
            return obs
        else:
            log.debug(f"No observations found. Now performing a cone search instead.")

    # If the above did not return a result, then do a cone search using the MAST name resolver
    # `radius` defaults to 0.0001 and unit arcsecond
    if radius is None:
        radius = 0.0001 * u.arcsec
    elif not isinstance(radius, u.quantity.Quantity):
        radius = radius * u.arcsec
    query_criteria["radius"] = str(radius.to(u.deg))

    try:
        log.debug(
            "Started querying MAST for observations within "
            f"{radius.to(u.arcsec)} arcsec of objectname='{target}'."
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NoResultsWarning)
            warnings.filterwarnings("ignore", message="t_exptime is continuous")
            obs = Observations.query_criteria(objectname=target, **query_criteria)
        obs.sort("distance")
        # We use `exptime` as an alias for `t_exptime`
        obs["exptime"] = obs["t_exptime"]
        return obs
    except ResolverError as exc:
        # MAST failed to resolve the object name to sky coordinates
        raise SearchError(exc) from exc


def _filter_products(
    products,
    campaign: Union[int, list[int]] = None,
    quarter: Union[int, list[int]] = None,
    #month=None,
    sector: Union[int, list[int]] = None,
    exptime: Union[str, int, tuple[int]] = None,
    limit: int =None,
    project=("Kepler", "K2", "TESS"),
    provenance_name=None,
    filetype="Target Pixel",
) -> pd.DataFrame:
    """Helper function which filters a SearchResult's products table by one or
    more criteria.

    Parameters
    ----------
    products : `pandas.DataFrame` object
        Pandas dataframe containing data products returned by MAST
    campaign : int or list
        Desired campaign of observation for data products
    quarter : int or list
        Desired quarter of observation for data products
    month : int or list
        Desired month of observation for data products
    exptime : 'long', 'short', 'fast', or float
        'long' selects 10-min and 30-min cadence products;
        'short' selects 1-min and 2-min products;
        'fast' selects 20-sec products.
        Alternatively, you can pass the exact exposure time or range of exposure times
        in seconds 
            e.g., ``exptime=600`` selects 10-minute cadence.
            exptime= (120,600) selects anything between 2 and 10 minutes
        By default, all cadence modes are returned.
    filetype : str
        Type of files queried at MAST (`Target Pixel` or `Lightcurve`).
    limit: int
        Max number of data products to return 

    Returns
    -------
    products : pandas dataframe object
        Masked astropy table containing desired data products
    """
    if provenance_name is None:  # apply all filters
        provenance_lower = ("kepler", "k2", "spoc")
    else:
        provenance_lower = [p.lower() for p in np.atleast_1d(provenance_name)]

    mask = np.ones(len(products), dtype=bool)

    # Kepler data needs a special filter for quarter and month
    mask &= ~np.array(
        [prov.lower() == "kepler" for prov in products["provenance_name"]]
    )
    if "kepler" in provenance_lower and campaign is None and sector is None:
        mask |= _mask_kepler_products(products, quarter=quarter, month=month)

    # HLSP products need to be filtered by extension
    if filetype.lower() == "lightcurve":
        mask &= np.array(
            [uri.lower().endswith("lc.fits") for uri in products["productFilename"]]
        )
    elif filetype.lower() == "target pixel":
        mask &= np.array(
            [
                uri.lower().endswith(("tp.fits", "targ.fits.gz"))
                for uri in products["productFilename"]
            ]
        )
    elif filetype.lower() == "ffi":
        mask &= np.array(["TESScut" in desc for desc in products["description"]])

    # Allow only fits files
    mask &= np.array(
        [
            uri.lower().endswith("fits") or uri.lower().endswith("fits.gz")
            for uri in products["productFilename"]
        ]
    )

    # Filter by cadence
    mask &= _mask_by_exptime(products, exptime)

    products = products[mask]

    products.sort(["distance", "productFilename"])
    if limit is not None:
        return products[0:limit]
    return products


def _mask_kepler_products(products, quarter=None, month=None):
    """Returns a mask flagging the Kepler products that match the criteria."""
    #mask = np.array([proj.lower() == "kepler" for proj in products["provenance_name"]])
    mask = products['provenance_name'] == 'kepler' 
    if sum(mask) == 0:
        return mask

    # Identify quarter by the description.
    # This is necessary because the `sequence_number` field was not populated
    # for Kepler prime data at the time of writing this function.
    quarter_mask = np.zeros(len(products), dtype=bool)
    if quarter is not None:
        for q in np.atleast_1d(quarter):
            quarter_mask += products['description'].str.endswith(f"Q{q}")
    mask &= quarter_mask

    # For Kepler short cadence data the month can be specified
    if month is not None:
        month = np.atleast_1d(month)
        # Get the short cadence date lookup table.
        table = pd.read_csv(
            os.path.join(PACKAGEDIR, "data", "short_cadence_month_lookup.csv")
        )
        # The following line is needed for systems where the default integer type
        # is int32 (e.g. Windows/Appveyor), the column will then be interpreted
        # as string which makes the test fail.
        table["StartTime"] = table["StartTime"].astype(str)
        # Grab the dates of each of the short cadence files.
        # Make sure every entry has the correct month
        is_shortcadence = mask & products['description'].str.contains("Short")

        for idx in np.where(is_shortcadence)[0]:
            quarter = np.atleast_1d(int(products["description"][idx].split(" - ")[-1].replace("-", "")[1:]))
            date = products['dataURI'][idx].split("/")[-1].split("-")[1].split("_")[0]
            # Check if the observation date matches the specified Quarter/month from the lookup table
            if date not in table["StartTime"][table['Month'].isin(month) & table['Quarter'].isin(quarter)].values:
                mask[idx] = False

    return mask


def _mask_by_exptime(products, exptime):
    """Helper function to filter by exposure time.
       Returns a boolean array """
    mask = np.ones(len(products), dtype=bool)
    if isinstance(exptime, (int, float)):
        mask &= products["exptime"] == exptime
    elif isinstance(exptime, tuple):
        mask &= (products["exptime"] >= min(exptime) & (products["exptime"] <= max(exptime)))
    elif isinstance(exptime, str):
        exptime = exptime.lower()
        if exptime in ["fast"]:
            mask &= products["exptime"] <= 60
        elif exptime in ["short"]:
            mask &= (products["exptime"] > 60) & (products["exptime"] <= 120)
        elif exptime in ["long", "ffi"]:
            mask &= products["exptime"] > 120
    return mask.values


def _resolve_object(target):
    """Ask MAST to resolve an object string to a set of coordinates."""
    from astroquery.mast import MastClass

    # Note: `_resolve_object` was renamed `resolve_object` in astroquery 0.3.10 (2019)
    return MastClass().resolve_object(target)