from astroquery.mast import Observations
import pandas as pd
from typing import Union, Optional
import re
import logging
import warnings
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from tqdm import tqdm

from copy import deepcopy

from .utils import SearchError, SearchWarning, suppress_stdout

from . import PACKAGEDIR, PREFER_CLOUD, DOWNLOAD_CLOUD, conf, config

pd.options.display.max_rows = 10

default_download_dir = config.get_cache_dir()

log = logging.getLogger(__name__)


class MASTSearch(object):
    """
    Generic Search Class for data exploration that queries mast for observations performed by the: Kepler, K2, TESS
    Missions, and returns the results in a convenient table with options to download.
    By default only mission products are returned.

    Parameters
    ----------
    target: Optional[Union[str, tuple[float], SkyCoord]] = None
        The target to search for observations of. Can be provided as a name (string),
        coordinates in decimal degrees (tuple), or astropy `~astropy.coordinates.SkyCoord` Object.
    obs_table:Optional[pd.DataFrame] = None
        Optionally, can provice a Astropy `~astropy.table.Table` Object from
        AstroQuery `~astroquery.mast.Observations.query_criteria` which will be used to construct the observations table
    prod_table:Optional[pd.DataFrame] = None
        Optionally, if you provide an obs_table, you may also provide a products table of assosciated products.  These
        two tables will be concatenated to become the primary joint table of data products.
    table:Optional[pd.DataFrame] = None
        Optionally, may provide an stropy `~astropy.table.Table` Object  that is the already merged joint table of obs_table
        and prod_table.
    search_radius:Optional[Union[float,u.Quantity]] = None
        The radius around the target name/location to search for observations.  Can be provided in arcsonds (float) or as an
        AstroPy `~astropy.units.u.Quantity` Object
    exptime:Optional[Union[str, int, tuple]] = (0,9999)
        Exposure time to filter observation results on.  Can be provided as a mission-specific string,
        an int which forces an exact match to the time in seconds, or a tuple, which provides a range to filter on.
    mission: Optional[Union[str, list[str]]] = ["Kepler", "K2", "TESS"]
        Mission(s) for which to search for data on
    pipeline:  Optional[Union[str, list[str]]] = ["Kepler", "K2", "SPOC"]
        Pipeline(s) which have produced the observed data
    sequence: Optional[int] = None,
        Mission Specific Survey value that corresponds to Sector (TESS), Campaign (K2), or Quarter (Kepler). Only 1 value permitted, will assume the same sequence number for all missions.
    """

    _REPR_COLUMNS = [
        "target_name",
        "pipeline",
        "mission",
        "exptime",
        "distance",
        "year",
        "description",
    ]

    table = None

    def __init__(
        self,
        target: Optional[Union[str, tuple[float], SkyCoord]] = None,
        obs_table: Optional[pd.DataFrame] = None,
        prod_table: Optional[pd.DataFrame] = None,
        table: Optional[pd.DataFrame] = None,
        search_radius: Optional[Union[float, u.Quantity]] = None,
        exptime: Optional[Union[str, int, tuple]] = (0, 9999),
        mission: Optional[Union[str, list[str]]] = ["Kepler", "K2", "TESS"],
        pipeline: Optional[Union[str, list[str]]] = ["Kepler", "K2", "SPOC"],
        sequence: Optional[int] = None,
    ):
        self.search_radius = search_radius
        self.search_exptime = exptime
        self.search_mission = np.atleast_1d(mission).tolist()
        if pipeline is not None:
            pipeline = np.atleast_1d(pipeline).tolist()
        self.search_pipeline = pipeline
        self.search_sequence = sequence

        # Legacy functionality - no longer query kic/tic by integer value only
        if isinstance(target, int):
            raise TypeError(
                "Target must be a target name string, (ra, dec) tuple, "
                "or astropy coordinate object"
            )

        self.target = target

        if isinstance(table, type(None)):
            self._searchtable_from_target(target)
            self.table = self._update_table(self.table)
            self.table = self._fix_table_times(self.table)

        # If MAST search tables are provided, another MAST search is not necessary
        else:
            self._searchtable_from_table(table, obs_table, prod_table)

    def __len__(self):
        """Returns the number of products in the SearchResult table."""
        return len(self.table)

    def __repr__(self):
        if isinstance(self.table, pd.DataFrame):
            if len(self.table) > 0:
                out = f"{self.__class__.__name__} object containing {len(self.table)} data products"
                return out + self.table[self._REPR_COLUMNS].__repr__()
            else:
                return "No results found"
        else:
            return "I am an uninitialized MASTSearch result"

    # Used to call the pandas table html output which is nicer
    def _repr_html_(self):
        if isinstance(self.table, pd.DataFrame):
            if len(self.table) > 0:
                out = f"{self.__class__.__name__} object containing {len(self.table)} data products"
                return out + self.table[self._REPR_COLUMNS]._repr_html_()
            else:
                return "No results found"
        else:
            return "I am an uninitialized MASTSearch result"

    def __getitem__(self, key):
        # TODO: Look into class mixins & pandas for this?
        strlist = False
        intlist = False
        if isinstance(key, list):
            if all(isinstance(n, int) for n in key):
                intlist = True
            if all(isinstance(n, str) for n in key):
                strlist = True

        if isinstance(key, (slice, int)) or (intlist):
            if not intlist:
                mask = np.in1d(
                    np.arange(len(self.table)), np.arange(len(self.table))[key]
                )
            else:
                mask = np.in1d(self.table.index, key)
            return self._mask(mask)
        if isinstance(key, (str, pd.Series)) or strlist:
            # Return a column as a series, or a dataframe of columns
            # Note that we're not returning a Search Object here as
            # we havce additional Requiered columns, etc.
            return self.table[key]
        if hasattr(key, "__iter__"):
            if len(key) == len(self.table):
                return self._mask(key)

    @property
    def ra(self):
        """Right Ascension coordinate for each data product found."""
        return self.table["s_ra"].values

    @property
    def dec(self):
        """Declination coordinate for each data product found."""
        return self.table["s_dec"].values

    @property
    def exptime(self):
        """Exposure times for all returned products"""
        return self.table["exptime"].values

    @property
    def mission(self):
        """Kepler quarter or TESS sector names for each data product found."""
        return self.table["mission"].values

    @property
    def year(self):
        """Year the observation was made."""
        return self.table["year"].values

    @property
    def pipeline(self):
        """Pipeline name for each data product found."""
        return self.table["pipeline"].values

    @property
    def target_name(self):
        """Target name for each data product found."""
        return self.table["target_name"].values

    @property
    def uris(self):
        """Location Information of the products in the table"""
        uris = self.table["dataURI"].values

        if PREFER_CLOUD:
            cloud_uris = self.cloud_uris
            mask = cloud_uris != None
            uris[mask] = cloud_uris[mask]

        return uris

    @property
    def cloud_uris(self):
        """Returns the cloud uris for products in self.table.
        Returns
        -------
        ~numpy.array of URI's from ~astroquery.mast
            an array where each element is the cloud-URI of a product in self.table
        """
        Observations.enable_cloud_dataset()
        return np.asarray(
            Observations.get_cloud_uris(Table.from_pandas(self.table), full_url=True)
        )

    @property
    def timeseries(self):
        """return a MASTSearch object with self.table only containing products that are a time-series measurement"""
        mask = self._mask_product_type("lightcurve")
        return self._mask(mask)

    @property
    def cubedata(self):
        """return a MASTSearch object with self.table only containing products that are image cubes"""
        mask = self._mask_product_type("target pixel")

        # return self._cubedata()
        return self._mask(mask)

    @property
    def dvreports(self):
        """return a MASTSearch object with self.table only containing products that are data validation pdf files"""
        mask = self._mask_product_type(filetype="dvreport")
        return self._mask(mask)

    # @cached
    def _searchtable_from_target(self, target: Union[str, tuple[float], SkyCoord]):
        """
        takes a target to search and
         - parses that target to search
         - searches mast to create a table from merged Obs/Prod Tables
         - filters the joint table

        Parameters
        ----------
        target : Union[str, tuple[float], SkyCoord]
            the target to search for, either a name (str) or coordinate - (tupe[float]/SkyCoord)

        Returns
        -------
        None - sets self.table equal to the masked/filtered joint table
        """
        self._parse_input(target)
        seq = self.search_sequence
        if isinstance(seq, list):
            seq = None
        self.table = self._search(
            search_radius=self.search_radius,
            exptime=self.search_exptime,
            mission=self.search_mission,
            pipeline=self.search_pipeline,
            sequence=seq,
        )
        mask = self._filter(
            exptime=self.search_exptime,
            mission=self.search_mission,
            pipeline=self.search_pipeline,
            sequence=self.search_sequence,
        )  # setting provenance_name=None will return HLSPs

        self.table = self.table[mask]

    def _searchtable_from_table(
        self,
        table: Optional[pd.DataFrame] = None,
        obs_table: Optional[pd.DataFrame] = None,
        prod_table: Optional[pd.DataFrame] = None,
    ):
        """creates a unified table from either:
            - a passed joint-table
                - this is just passed through
            - an obs_table from astroquery.mast.query_critera.to_pandas()
                - this uses obs_table to create a prod_table and merges tables
            - an obs_table AND a prod_table from from astroquery.mast.get_products.to_pandas()
                - this meges obs_table and prod_table

            self.table is then set from this table
        Parameters
        ----------
        table : Optional[pd.DataFrame], optional
            pre-merged obs_table, prod_table by default None
        obs_table : Optional[pd.DataFrame], optional
            table from astroquery.mast.query_critera.to_pandas(), by default None
        prod_table : Optional[pd.DataFrame], optional
            table from astroquery.mast.get_products.to_pandas(), by default None
        """
        # see if function was passed a joint table
        if isinstance(table, pd.DataFrame):
            self.table = table

        # If we don't have a name or a joint table,
        # check to see if tables were passed
        elif isinstance(obs_table, pd.DataFrame):
            # If we have an obs table and no name, use it
            self.obs_table = obs_table
            if isinstance(prod_table, type(None)):
                # get the prod table if we don't have it
                prod_table = self._search_products(self)
            self.prod_table = prod_table
            self.table = self._join_tables()
        else:
            raise (ValueError("No Target or object table supplied"))

    def _mask(self, mask):
        """Masks down the product and observation tables given an input mask, then returns them as a new Search object.
        deepcopy is used to preserve the class metadata stored in class variables"""
        new_MASTSearch = deepcopy(self)
        new_MASTSearch.table = self.table[mask].reset_index()

        return new_MASTSearch

    def _update_table(self, joint_table: pd.DataFrame):
        """updates self.table for a handfull of convenient column renames and streamlining choices

        Parameters
        ----------
        joint_table : ~pandas.DataFrame
            a dataframe from a merged astroquery.mast.query_criteria & astroquery.mast.get_products tables

        Returns
        -------
        joint_table : ~pandas.DataFrame
            the updated & re-formatted data-frame
        """

        # Modifies a MAST table with user-friendly columns
        if "t_exptime" in joint_table.columns:
            joint_table = joint_table.rename(columns={"t_exptime": "exptime"})
        if "provenance_name" in joint_table.columns:
            joint_table["pipeline"] = joint_table["provenance_name"].copy()
        if "obs_collection_obs" in joint_table.columns:
            joint_table["mission"] = joint_table["obs_collection_obs"].copy()

        # rename identical columns
        joint_table.rename(
            columns={
                "obs_collection_prod": "obs_collection",
                "project_prod": "project",
                "dataproduct_type_prod": "dataproduct_type",
                "proposal_id_prod": "proposal_id",
                "dataRights_prod": "dataRights",
            },
            inplace=True,
        )

        return joint_table

    def _fix_table_times(self, joint_table: pd.DataFrame):
        """fixes incorrect times and adds convenient columns to the search table
        MAST returns the min and max time for each observation in the table.
        Turn this value into the standard JD format
        Also extract the observation year for sorting purposes

        Parameters
        ----------
        joint_table : ~pandas.DataFrame
            the meged obs, prod search table

        Returns
        -------
        ~pandas.DataFrame
            the input table with updated values and additional columns
        """
        if isinstance(joint_table.index, pd.MultiIndex):
            # Multi-Index leading to issues, re-index?
            joint_table = joint_table.reset_index()

        year = np.floor(Time(joint_table["t_min"], format="mjd").decimalyear)
        # `t_min` is incorrect for Kepler pipeline products, so we extract year from the filename for those
        for idx, row in joint_table.iterrows():
            if (row["pipeline"] == "Kepler") & (
                "Data Validation" not in row["description"]
            ):
                year[idx] = re.findall(r"\d+.(\d{4})\d+", row["productFilename"])[0]
        joint_table["year"] = year.astype(int)

        joint_table["start_time"] = Time(self.table["t_min"], format="mjd").iso
        joint_table["end_time"] = Time(self.table["t_max"], format="mjd").iso

        return joint_table

    def _search(
        self,
        search_radius: Union[float, u.Quantity] = None,
        exptime: Union[str, int, tuple] = (0, 9999),
        mission: Union[str, list[str]] = ["Kepler", "K2", "TESS"],
        pipeline: Union[str, list[str]] = None,
        sequence: int = None,
    ):
        """from a parsed target input in self.target/self.SkyCoord -
            creates an obs_table & prod_table from self._search_obs() & self._search_prod(,
            and performs an outer joint to merge them

        Parameters
        ----------
        search_radius : Union[float, u.Quantity], optional
            radius (in arcseconds if units not specified)
            around target/coordinates to search, by default None
        exptime : Union[str, int, tuple], optional
            exposure time of products to search for by default (0, 9999)
        pipeline : Union[str, list[str]], optional
            pipeline to search for products from, by default None
        sequence : int, optional
            sequence number (e.g. Cadence/Sector/Campaign) to search for
            products from, by default None

        Returns
        -------
        ~pandas.DataFrame
            A DataFrame resulting from an outer join on a table from
            ~astroquery.mast.Observations.query_criteria and its assosciated
            ~astroquery.mast.Observations.get_product_list
        """
        self.obs_table = self._search_obs(
            search_radius=search_radius,
            exptime=exptime,
            mission=mission,
            pipeline=pipeline,
            sequence=sequence,
            filetype=["lightcurve", "target pixel", "dv"],
        )
        self.prod_table = self._search_prod()
        joint_table = self._join_tables()

        return joint_table

    def _parse_input(self, search_input: Union[str, tuple[float], SkyCoord]):
        """Parses the provided target input search information based on input type
            - If SkyCoord, sets self.Skycoord &
                creates a search string based on coordinates
            - If tuple, assumes RA dec - sets search string based on coordinates &
                creates self.SkyCoord
            - if str, assumes target name - sets search string to input string &
                creates self.SkyCoord from SkyCoord.from_name

        Parameters
        ----------
        search_input : Union[str, tuple, SkyCoord]
            The provided user target search input

        Raises
        ------
        TypeError
            Raise an error if we don't recognise what type of data was passed in
        """
        # If passed a SkyCoord, convert it to an "ra, dec" string for MAST
        self.exact_target = False

        if isinstance(search_input, SkyCoord):
            self.target_search_string = f"{search_input.ra.deg}, {search_input.dec.deg}"
            self.SkyCoord = search_input

        elif isinstance(search_input, tuple):
            self.target_search_string = f"{search_input[0]}, {search_input[1]}"
            self.SkyCoord = SkyCoord(search_input, frame="icrs", unit="deg")

        elif isinstance(search_input, str):
            self.target_search_string = search_input
            self.SkyCoord = SkyCoord.from_name(search_input, frame="icrs")

            target_lower = str(search_input).lower()
            target_str = self._check_exact(target_lower)
            if target_str:
                self.exact_target_name = self._target_to_exact_name(target_str)
                self.exact_target = True

        else:
            raise TypeError(
                "Target must be a target name string or astropy coordinate object"
            )

    def _check_exact(self, target):
        """We dont check exact target name for the generic MAST search -
        TESS/Kepler/K2 have different exact names for identical objects"""
        return False

    def _target_to_exact_name(self, target):
        """We dont check exact target name for the generic MAST search -
        TESS/Kepler/K2 have different exact names for identical objects"""
        return NotImplementedError("Use mission appropriate search for exact targets")

    def _add_s3_url_column(self, joint_table: pd.DataFrame) -> pd.DataFrame:
        """
            self.table will updated to have an extra column of s3 URLS if possible

        Parameters
        ----------
        joint_table : ~pandas.DataFrame
            Dataframe of merged ~astroquery.mast.Observations observations table and product table

        Returns
        -------
        ~pandas.DataFrame
            input dataframe with a column added which countaings the cloud uris of assosciated producs
        """

        Observations.enable_cloud_dataset()
        cloud_uris = Observations.get_cloud_uris(
            Table.from_pandas(joint_table), full_url=True
        )
        joint_table["cloud_uri"] = cloud_uris
        return joint_table

    def _search_obs(
        self,
        search_radius: Union[float, u.Quantity, None] = None,
        filetype: Union[str, tuple[str]] = ["lightcurve", "target pixel", "dv"],
        mission=["Kepler", "K2", "TESS"],
        pipeline: Union[str, tuple[str], type[None]] = None,
        exptime: Union[int, tuple[int]] = (0, 9999),
        sequence: Union[int, list[int], type[None]] = None,
    ):
        """Assuming we alreads have a pased target search input,
        parse optional inputs and then query mast using
        ~astroquery.mast.Observations.query_criteria

        Parameters
        ----------
        search_radius : Union[float, u.Quantity, None], optional
            radius to search around the target, by default None
        filetype : Union[str, tuple, optional
            type of files to search for, by default ["lightcurve", "target pixel", "dv"]
        mission : list, optional
            mission to search for data from, by default ["Kepler", "K2", "TESS"]
        pipeline : Union[str, tuple, optional
            pipeline to search for data from, by default None
        exptime : Union[int, tuple[int]], optional
            exposure time of data products to search for, by default (0, 9999)
        sequence : Union[int, list[int], type, optional
            mission dependent sequence (e.g. segment/campaign/sector) to search for
            data from, by default None

        Returns
        -------
        ~pandas.DataFrame
            observations table from ~astroquery.mast.Observations.query_criteria().to_pandas()

        Raises
        ------
        SearchError
             - If a ffi filetype is searched, use TESSSearch instead
        SearchError
            If no data is found
        """

        if filetype == "ffi":
            raise SearchError(
                f"FFI search not implemented in MASTSearch. Please use TESSSearch."
            )

        # Ensure mission is a list
        mission = np.atleast_1d(mission).tolist()
        if pipeline is not None:
            pipeline = np.atleast_1d(pipeline).tolist()
            # If pipeline "TESS" is used, we assume it is SPOC
            pipeline = np.unique(
                [p if p.lower() != "tess" else "SPOC" for p in pipeline]
            )

        # Speed up by restricting the MAST query if we don't want FFI image data
        # At MAST, non-FFI Kepler pipeline products are known as "cube" products,
        # and non-FFI TESS pipeline products are listed as "timeseries"
        extra_query_criteria = {}
        filetype_query_criteria = {"lightcurve": "timeseries", "target pixel": "cube"}

        extra_query_criteria["dataproduct_type"] = [
            filetype_query_criteria[file.lower()]
            for file in filetype
            if (file.lower() in filetype_query_criteria.keys())
        ]

        observations = self._query_mast(
            search_radius=search_radius,
            project=mission,
            pipeline=pipeline,
            exptime=exptime,
            sequence_number=sequence,
            **extra_query_criteria,
        )
        log.debug(
            f"MAST found {len(observations)} observations. "
            "Now querying MAST for the corresponding data products."
        )
        if len(observations) == 0:
            raise SearchError(f"No data found for target {self.target}.")

        return observations

    def _query_mast(
        self,
        search_radius: Union[float, u.Quantity, None] = None,
        project: Union[str, list[str]] = ["Kepler", "K2", "TESS"],
        pipeline: Union[str, list[str], None] = None,
        exptime: Union[int, float, tuple, type(None)] = (0, 9999),  # None,
        sequence_number: Union[int, list[int], None] = None,
        **extra_query_criteria,
    ):
        """Attempts to find mast observations using ~astroquery.mast.Observations.query_criteria

        Parameters
        ----------
        search_radius : Union[float, u.Quantity, None], optional
            radius around target/location to search, by default None
        project : Union[str, list[str]], optional
            project (mission) to search for data from, by default ["Kepler", "K2", "TESS"]
        pipeline : Union[str, list[str], None], optional
            providence(pipeline) of processed data, by default None
        exptime : Union[int, float, tuple, type, optional
            exposure time of data products to search, by default (0, 9999)
        sequence_number : Union[int, list[int], None], optional
            mission dependent identifier (e.g. Cadence/Campaign/Sector) to search for data products from, by default None

        Returns
        -------
        ~pandas.DataFrame
            an observations_table from query_criteria in pandas DataFrame format

        Raises
        ------
        SearchError
            When unable to resolve search target
        """
        from astroquery.exceptions import NoResultsWarning, ResolverError

        # Constructs the appropriate query for mast
        log.debug(f"Searching for {self.target} with {exptime} on project {project}")

        # We pass the following `query_criteria` to MAST regardless of whether
        # we search by position or target name:
        query_criteria = {"project": project, **extra_query_criteria}
        if pipeline is not None:
            query_criteria["provenance_name"] = pipeline
        if sequence_number is not None:
            query_criteria["sequence_number"] = sequence_number
        if exptime is not None:
            query_criteria["t_exptime"] = exptime

        if self.exact_target and (search_radius is None):
            log.debug(
                f"Started querying MAST for observations with exact name {self.exact_target_name}"
            )

            # do an exact name search with target_name=
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=NoResultsWarning)
                warnings.filterwarnings("ignore", message="t_exptime is continuous")
                obs = Observations.query_criteria(
                    target_name=self.exact_target_name, **query_criteria
                )

            if len(obs) > 0:
                # astroquery does not report distance when querying by `target_name`;
                # we add it here so that the table returned always has this column.
                obs["distance"] = 0.0
                return obs.to_pandas()
        else:
            if search_radius is None:
                search_radius = 0.0001 * u.arcsec

            elif not isinstance(search_radius, u.quantity.Quantity):
                log.warning(
                    f"Search radius {search_radius} units not specified, assuming arcsec"
                )
                search_radius = search_radius * u.arcsec

            query_criteria["radius"] = str(search_radius.to(u.deg))

            try:
                log.debug(
                    "Started querying MAST for observations within "
                    f"{search_radius.to(u.arcsec)} arcsec of objectname='{self.target}'."
                    f"Via {self.target_search_string} search string and query_criteria: "
                    f"{query_criteria}"
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=NoResultsWarning)
                    warnings.filterwarnings("ignore", message="t_exptime is continuous")
                    obs = Observations.query_criteria(
                        objectname=self.target_search_string, **query_criteria
                    )
                obs.sort("distance")
                return obs.to_pandas()
            except ResolverError as exc:
                # MAST failed to resolve the object name to sky coordinates
                raise SearchError(exc) from exc

        return obs.to_pandas()

    def _search_prod(self):
        """uses ~astroquery.mast.Observations.get_product_list to get data products assosicated
        with self.obs_table

        Returns
        -------
        ~pandas.DataFrame
            product table from ~astroquery.mast.Observations.get_product_list.to_pandas
        """
        # Use the search result to get a product list
        products = Observations.get_product_list(Table.from_pandas(self.obs_table))
        return products.to_pandas()

    def _join_tables(self):
        """perform an outer join on self.obs_table on obs_id,
        and self.prod_table and return joined table

        Returns
        -------
        ~pandas.DataFrame
            joined table
        """
        joint_table = pd.merge(
            self.obs_table.reset_index().rename({"index": "obs_index"}, axis="columns"),
            self.prod_table.reset_index().rename(
                {"index": "prod_index"}, axis="columns"
            ),
            on="obs_id",
            how="left",
            suffixes=("_obs", "_prod"),
        ).set_index(["obs_index", "prod_index"])

        log.debug(f"MAST found {len(joint_table)} matching data products.")
        return joint_table

    def _add_kepler_sequence_num(self):
        """adds sequence number to kepler data in the self.table["sequence_number"]
        column since these are not populated in mast
        """
        seq_num = self.table["sequence_number"].values.astype(str)

        # Kepler sequence_number values were not populated at the time of
        # writing this code, so we parse them from the description field.
        mask = (self.table["project"] == "Kepler") & self.table[
            "sequence_number"
        ].isna()
        re_expr = r".*Q(\d+)"
        seq_num[mask] = [
            re.findall(re_expr, item[0])[0] if re.findall(re_expr, item[0]) else ""
            for item in self.table.loc[mask, ["description"]].str.values
        ]

    def _mask_product_type(
        self,
        filetype: str,
    ):
        """convenience function to filter the productFilename column in self.table
        using the pandas .endswith function

        Parameters
        ----------
        filetype : str
            the filetype to filter for

        Returns
        -------
        numpy boolean array
            boolean mask for the column/table
        """

        mask = np.zeros(len(self.table), dtype=bool)
        # This is the dictionary of what files end with that correspond to each allowed file type
        ftype_suffix = {
            "lightcurve": ["lc.fits"],
            "target pixel": ["tp.fits", "targ.fits.gz"],
            "dvreport": ["dvr.pdf", "dvm.pdf", "dvs.pdf"],
        }

        for value in ftype_suffix[filetype]:
            mask |= self.table.productFilename.str.endswith(value)
        return mask

    def _filter(
        self,
        exptime: Union[str, int, tuple[int], type(None)] = (0, 9999),
        mission: Union[str, list[str]] = ["Kepler", "K2", "TESS"],
        pipeline: Union[str, list[str]] = ["kepler", "k2", "spoc"],
        filetype: Union[str, list[str]] = [
            "target pixel",
            "lightcurve",
            "dvreport",
        ],
        sequence: Union[int, list[int]] = None,
    ) -> pd.DataFrame:
        """filter self.table based on your product search preferences

        Parameters
        ----------
        exptime : Union[str, int, tuple[int], type, optional
            exposure time of data products to search, by default (0, 9999)
        project : Union[str, list[str]], optional
            mission project to search, by default ["Kepler", "K2", "TESS"]
        pipeline : Union[str, list[str]], optional
            pipeline provinence to search for data from, by default ["kepler", "k2", "spoc"]
        filetype : Union[str, list[str]], optional
            file types to search for, by default [ "target pixel", "lightcurve", "dvreport", ]
        sequence : Union[int, list[int]], optional
            sequence number to filter by. Corresponds to sector for TESS, campaign for K2, and quarter for Kepler
        Returns
        -------
        mask
            cumulative boolean mask for self.table based off of
            the product of individual filter properties
        """

        """ Modify this so that it can choose what types of products to keep?
        Since this will be used by mission specific search we want this to filter:
        Filetype
        ExposureTime/cadence
        Pipe(Provenance)/Project - e.g. (SPOC/TESSSpoc)
        <Segment/Quarter/Sector/Campaign> this will be in mission specific search
        """

        self.search_exptime = exptime
        mask = np.zeros(len(self.table), dtype=bool)

        # First filter on filetype
        file_mask = mask.copy()

        # This is the list of allowed filetypes we can interact with
        allowed_ftype = ["lightcurve", "target pixel", "dvreport"]

        filter_ftype = [
            file.lower() for file in filetype if file.lower() in allowed_ftype
        ]
        # First filter on filetype

        if len(filter_ftype) == 0:
            filter_ftype = allowed_ftype
            log.warning("Invalid filetype filtered. Returning all data.")

        file_mask = mask.copy()
        for ftype in filter_ftype:
            file_mask |= self._mask_product_type(ftype)

        # Next Filter on mission
        mission_mask = mask.copy()
        if not isinstance(mission_mask, type(None)):
            for m in mission:
                mission_mask |= self.table.project_obs.values == m
        else:
            mission_mask = np.logical_not(mission_mask)

        # Next Filter on pipeline (provenance_name in mast table)
        provenance_mask = mask.copy()
        if not isinstance(pipeline, type(None)):
            for a in np.atleast_1d(pipeline).tolist():
                provenance_mask |= self.table.provenance_name.str.lower() == a.lower()
        else:
            provenance_mask = np.logical_not(provenance_mask)

        # Filter by cadence
        if not isinstance(exptime, type(None)):
            exptime_mask = self._mask_by_exptime(exptime)
        else:
            exptime_mask = not mask

        # Filter by sequence
        sequence_mask = mask.copy()
        if not isinstance(sequence, type(None)):
            for s in np.atleast_1d(sequence).tolist():
                sequence_mask |= self.table.sequence_number == s
        else:
            sequence_mask = np.logical_not(sequence_mask)

        mask = file_mask & mission_mask & provenance_mask & exptime_mask & sequence_mask
        return mask

    def _mask_by_exptime(self, exptime: Union[int, tuple[float]]):
        """Helper function to filter by exposure time.
        Returns a boolean array"""
        if "t_exptime" in self.table.columns:
            exposures = self.table["t_exptime"]
        else:
            exposures = self.table["exptime"]

        if isinstance(exptime, (int, float)):
            mask = exposures == exptime
        elif isinstance(exptime, tuple):
            mask = (exposures >= min(exptime)) & (exposures <= max(exptime))
        elif isinstance(exptime, str):
            exptime = exptime.lower()
            if exptime in ["fast"]:
                mask = exposures < 60
            elif exptime in ["short"]:
                mask = (exposures >= 60) & (exposures <= 120)
            elif exptime in ["long", "ffi"]:
                mask = exposures > 120
            elif exptime in ["shortest"]:
                mask = exposures == min(exposures)
            elif exptime in ["longest"]:
                mask = exposures == max(exposures)
            else:
                mask = np.ones(len(exposures), dtype=bool)
                log.debug("invalid string input. No exptime filter applied")
        return mask

    @suppress_stdout
    def _download_one(
        self,
        row: pd.Series,
        cloud_only: bool = False,
        cache: bool = True,
        download_dir: str = ".",
    ) -> pd.DataFrame:
        """Helper function that downloads an individual row.
        This may be more efficient if we are caching, but we can sent a full table
        to download_products to get multiple items.
        """

        # Make sure astroquery uses the same level of verbosity
        print(log.getEffectiveLevel())
        logging.getLogger("astropy").setLevel(log.getEffectiveLevel())
        logging.getLogger("astroquery").setLevel(log.getEffectiveLevel())

        manifest = Observations.download_products(
            Table().from_pandas(row.to_frame(name=" ").transpose()),
            download_dir=download_dir,
            cache=cache,
            cloud_only=cloud_only,
        )
        return manifest.to_pandas()

    def filter_table(
        self,
        # Filter the table by keywords
        limit: int = None,
        exptime: Union[int, float, tuple, type(None)] = None,
        pipeline: Union[str, list[str]] = None,
    ):
        mask = np.ones(len(self.table), dtype=bool)

        if exptime is not None:
            mask = mask & self._mask_by_exptime(exptime)
        if pipeline is not None:
            mask = mask & self.table["pipeline"].isin(np.atleast_1d(pipeline))
        if limit is not None:
            cusu = np.cumsum(mask)
            if max(cusu) > limit:
                mask = mask & (cusu <= limit)
        return self._mask(mask)

    def download(
        self,
        cloud: bool = True,
        cache: bool = True,
        cloud_only: bool = False,
        download_dir: str = default_download_dir,
        remove_incomplete: str = True,
    ) -> pd.DataFrame:
        """downloads products in self.table to the local hard-drive

        Parameters
        ----------
        cloud : bool, optional
            enable cloud (as oposed to MAST) downloading, by default True
        cache : bool, optional
            enable astroquery_caching of the downloaded files, by default True
            if True, will not overwrite the file to be downloaded if it is found to exist
        cloud_only : bool, optional
            download only products availaible in the cloud, by default False
        download_dir : str, optional
            directory where the products should be downloaded to,
             by default default_download_dir
        remove_incomplete: str, optional
            remove files with a status not "COMPLETE" in the manifest, by default True
        Returns
        -------
        ~pandas.DataFrame
            table where each row is an ~astroquery.mast.Observations.download_products()
            manifest
        """

        if len(self.table) == 0:
            warnings.warn("Cannot download from an empty search result.", SearchWarning)
            return None

        if cloud:
            logging.getLogger("astroquery").setLevel(log.getEffectiveLevel())
            Observations.enable_cloud_dataset()

        manifest = [
            self._download_one(row, cloud_only, cache, download_dir)
            for _, row in tqdm(
                self.table.iterrows(),
                total=self.table.shape[0],
                desc="pipeline products",
            )
            # for _, row in self.table.iterrows()
        ]

        manifest = pd.concat(manifest)
        status = manifest["Status"] != "COMPLETE"
        if np.any(status):
            warnings.warn(
                "Not All Files Downloaded Successfully, Check Returned Manifest.",
                SearchWarning,
            )
            if remove_incomplete:
                for file in manifest.loc[status]["Local Path"].values:
                    if os.path.isfile(file):
                        os.remove(file)
                        warnings.warn(f"Removed {file}", SearchWarning)
                    else:
                        warnings.warn(f"Not a file: {file}", SearchWarning)

        return manifest
