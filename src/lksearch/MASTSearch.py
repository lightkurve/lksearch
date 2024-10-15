from astroquery.mast import Observations
from astroquery.mast import MastClass
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

from . import conf, config, log

pd.options.display.max_rows = 10


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
    sequence: Optional[Union[int, list[int]]] = None,
        Mission Specific Survey value that corresponds to Sector (TESS) AND Campaign (K2). Not valid for Kepler.
        Setting sequence is not recommented for MASTSearch.
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
        sequence: Optional[Union[int, list[int]]] = None,
    ):
        self.search_radius = search_radius
        self.search_exptime = exptime
        self.search_mission = np.atleast_1d(mission).tolist()
        if pipeline is not None:
            pipeline = np.atleast_1d(pipeline).tolist()
        self.search_pipeline = pipeline

        if ("kepler" in (m.lower() for m in mission)) & (sequence is not None):
            log.warning(
                "Sequence not valid when searching for Kepler data. Setting sequence to None"
            )
            sequence = None

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
            self.table = self._fix_table_times(self.table)

        # If MAST search tables are provided, another MAST search is not necessary
        else:
            self._searchtable_from_table(table, obs_table, prod_table)

        for col in self.table.columns:
            if not hasattr(self, col):
                setattr(self, col, self.table[col])

    def __len__(self):
        """Returns the number of products in the SearchResult table."""
        return len(self.table)

    def __repr__(self):
        if isinstance(self.table, pd.DataFrame):
            if len(self.table) > 0:
                out = f"{self.__class__.__name__} object containing {len(self.table)} data products \n"
                return out + self.table[self._REPR_COLUMNS].__repr__()
            else:
                return "No results found"
        else:
            return "I am an uninitialized MASTSearch result"

    # Used to call the pandas table html output which is nicer
    def _repr_html_(self):
        if isinstance(self.table, pd.DataFrame):
            if len(self.table) > 0:
                out = f"{self.__class__.__name__} object containing {len(self.table)} data products \n"
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

        if hasattr(key, "__iter__") or isinstance(key, pd.Series):
            if len(key) == len(self.table):
                return self._mask(key)

        if isinstance(key, (slice, int)) or (intlist):
            if not intlist:
                mask = np.in1d(
                    np.arange(len(self.table)), np.arange(len(self.table))[key]
                )
            else:
                mask = np.in1d(self.table.index, key)
            return self._mask(mask)
        if isinstance(key, str) or strlist:
            # Return a column as a series, or a dataframe of columns
            # Note that we're not returning a Search Object here as
            # we havce additional Requiered columns, etc.
            return self.table[key]

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

        if conf.PREFER_CLOUD:
            cloud_uris = self.cloud_uris
            mask = cloud_uris is not None
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
        if "cloud_uri" not in self.table.columns:
            self.table = self._add_s3_url_column(self.table)

        return self.table["cloud_uri"]

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

        self.table = self._search(
            search_radius=self.search_radius,
            exptime=self.search_exptime,
            mission=self.search_mission,
            pipeline=self.search_pipeline,
            sequence=self.search_sequence,
        )
        self.table = self._update_table(self.table)

        filetype = [
            "target pixel",
            "lightcurve",
            "dvreport",
        ]
        mask = self._filter(
            exptime=self.search_exptime,
            mission=self.search_mission,
            pipeline=self.search_pipeline,
            sequence_number=self.search_sequence,
            filetype=filetype,
        )
        self.table = self.table[mask].reset_index(drop=True)

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

    # def _downsize_table(self, ds_table):
    def _mask(self, mask):
        """Masks down the product and observation tables given an input mask, then returns them as a new Search object.
        deepcopy is used to preserve the class metadata stored in class variables"""

        new_MASTSearch = deepcopy(self)
        new_MASTSearch.table = new_MASTSearch.table[mask].reset_index(drop=True)

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
            joint_table = joint_table.reset_index(drop=True)

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
                creates self.SkyCoord from `MastClass().resolve_object()`

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
            self.SkyCoord = SkyCoord(*search_input, frame="icrs", unit="deg")

        elif isinstance(search_input, str):
            self.SkyCoord = MastClass().resolve_object(search_input)
            self.target_search_string = (
                f"{self.SkyCoord.ra.deg}, {self.SkyCoord.dec.deg}"
            )

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

        logging.getLogger("astroquery").setLevel(log.getEffectiveLevel())

        Observations.enable_cloud_dataset()
        cloud_uris = Observations.get_cloud_uris(
            Table.from_pandas(joint_table.loc[pd.notna(joint_table["dataURI"])]),
            full_url=True,
        )
        joint_table.loc[pd.notna(joint_table["dataURI"]), "cloud_uri"] = cloud_uris
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
                "FFI search not implemented in MASTSearch. Please use TESSSearch."
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
        elif isinstance(exptime, list):
            mask = np.zeros(len(exposures), dtype=bool)
            for et in exptime:
                mask = mask | (exposures == et)

        return mask

    def query_table(
        self,
        criteria: str,
        inplace: bool = False,
    ):
        """Filter the Search Result table using pandas query
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html

        Parameters
        ----------
        criteria : str
            string containing criteria to filter table. Can handle multiple criteria,
            e.g. "exptime>=100 and exptime<=500".
        inplace : bool
            if True, modify table in MASTSearch object directly. If False, returns a
            new MASTSearch object with the resulting table

        Returns
        -------
        MASTSearch : MASTSearch object
            Only returned if inplace = False
        """
        filtered_table = self.table.query(criteria).reset_index()
        if inplace:
            self.table = filtered_table
        else:
            new_MASTSearch = deepcopy(self)
            new_MASTSearch.table = filtered_table
            return new_MASTSearch

    def _filter(
        self,
        target_name: Union[str, list[str]] = None,
        pipeline: Union[str, list[str]] = None,
        mission: Union[str, list[str]] = None,
        exptime: Union[int, float, tuple[float]] = None,
        distance: Union[float, tuple[float]] = None,
        year: Union[int, list[int], tuple[int]] = None,
        description: Union[str, list[str], tuple[str]] = None,
        filetype: Union[str, list[str]] = None,
        sequence_number: Union[int, list[int]] = None,
    ):
        """filter self.table based on your product search preferences

        Parameters
        ----------
        target_name : str, optional
            Name of targets.
        pipeline : str or list[str]], optional
            Pipeline provenance to search for data from.
        mission : str or list[str], optional
            Mission projects to search, options are "Kepler", "K2", and "TESS".
        exptime : int or float, tuple[float]], optional
            Exposure time of data products to search.
            An int, float, or list will look for exact matches,
            a tuple will look for a range of times,
            and string values "fast", "short", "long", "shortest", and "longest"
            will match the appropriate exposure times.
        distance : float or tuple[float]], optional
            Distance from given search target to get data products.
            A float searches for products within the given distance,
            a tuple searches between the given values.
        year : int or list[int], tuple[int], optional
            Year of creation.
            A list will look for all years given,
            a tuple will look in the range of years given.
        description : str or list[str], optional
            Key words to search for in the description of the data product.
            A list will look for descriptions containing any keywords given,
            a tuple will look for descriptions containing all the keywords.
        filetype : str or list[str], optional
            file types to search for, options are "target pixel", "lightcurve", "dvreport".
        sequence_number : int or list[int]], optional
            Sequence number to filter by. Corresponds to sector for TESS, campaign for K2, and quarter for Kepler.

        Returns
        -------
        mask : np.ndarray
            cumulative boolean mask for self.table based off of
            the product of individual filter properties
        """
        mask = np.ones(len(self.table), dtype=bool)
        if target_name is not None:
            target_name = np.atleast_1d(target_name).astype(str)
            mask = mask & self.table["target_name"].isin(target_name)

        if not isinstance(filetype, type(None)):
            allowed_ftype = ["lightcurve", "target pixel", "dvreport"]
            filter_ftype = [
                file.lower()
                for file in np.atleast_1d(filetype).astype(str)
                if file.lower() in allowed_ftype
            ]
            if len(filter_ftype) == 0:
                filter_ftype = allowed_ftype
                log.warning("Invalid filetype filtered. Returning all filetypes.")

            file_mask = np.zeros(len(self.table), dtype=bool)
            for ftype in filter_ftype:
                file_mask |= self._mask_product_type(ftype)
            mask = mask & file_mask

        if exptime is not None:
            mask = mask & self._mask_by_exptime(exptime)

        if distance is not None:
            if isinstance(distance, float):
                mask = mask & self.table.eval("distance <= @distance")
            elif isinstance(distance, tuple):
                mask = mask & self.table.eval(
                    "(distance >= @distance[0]) & (distance <= @distance[1])"
                )
            else:
                log.warning(
                    "Invalid input for `distance`, allowed inputs are float and tuple. Ignoring `distance` search parameter."
                )

        if year is not None:
            if isinstance(year, str):
                year = int(year)
            if hasattr(year, "__iter__"):
                year_type = type(year)
                year = [int(y) for y in year]
                year = year_type(year)
            if (
                isinstance(year, np.int_)
                or isinstance(year, int)
                or isinstance(year, list)
            ):
                mask = mask & self.table["year"].isin(np.atleast_1d(year))
            elif isinstance(year, tuple):
                mask = mask & self.table.eval("year>=@year[0] & year<=@year[1]")
            else:
                log.warning(
                    "Invalid input for `year`, allowed inputs are str, int, and tuple. Ignoring `year` search parameter."
                )

        if pipeline is not None:
            pipeline = list(map(str.lower, np.atleast_1d(pipeline).astype(str)))
            mask = mask & self.table["pipeline"].str.lower().isin(pipeline)

        if description is not None:
            if isinstance(description, str):
                mask = mask & self.table["description"].str.lower().str.contains(
                    description.lower()
                )
            elif isinstance(description, tuple):
                # Looks for descriptions which contain *all* of the given keywords
                for word in description:
                    mask = mask & self.table["description"].str.lower().str.contains(
                        word.lower()
                    )
            elif hasattr(description, "__iter__"):
                # Looks for descriptions which contain *any* of the given keywords
                desc_mask = np.zeros(len(self.table), dtype=bool)
                for word in description:
                    desc_mask = desc_mask | self.table[
                        "description"
                    ].str.lower().str.contains(word.lower())
                mask = mask & desc_mask
            else:
                log.warning(
                    "Invalid input for `description`, allowed inputs are str and list[str]. Ignoring `description` search parameter."
                )

        if sequence_number is not None:
            mask = mask & self.table.sequence_number.isin(
                np.atleast_1d(sequence_number)
            )

        if mission is not None:
            mission = list(map(str.lower, np.atleast_1d(mission).astype(str)))
            mask = mask & self.table["mission"].str.lower().isin(mission)

        return mask

    def filter_table(
        self,
        target_name: Union[str, list[str]] = None,
        pipeline: Union[str, list[str]] = None,
        mission: Union[str, list[str]] = None,
        exptime: Union[int, float, tuple[float]] = None,
        distance: Union[float, tuple[float]] = None,
        year: Union[int, list[int], tuple[int]] = None,
        description: Union[str, list[str]] = None,
        filetype: Union[str, list[str]] = None,
        sequence: Union[str, list[str]] = None,
        limit: int = None,
        inplace=False,
    ):
        """Filter the search by keywords

        Parameters
        ----------
        target_name : str, optional
            Name of targets. A list will look for multiple target names.
        pipeline : str or list[str]], optional
            Data pipeline.  A list will look for multiple pipelines.
        mission : str or list[str]], optional
            Mission. A list will look for muliple missions.
        exptime : int or float, tuple[float]], optional
            Exposure Time. A tuple will look for a range of times.
        distance : float or tuple[float]], optional
            Distance. A float searches for products with a distance less than the value given,
            a tuple will search between the given values.
        year : int or list[int], tuple[int]], optional
            Year. A list will look for multiple years, a tuple will look in the range of years.
        description : str or list[str]], optional
            Description of product. A list will look for descriptions containing any keywords given,
            a tuple will look for descriptions containing all the keywords.
        filetype : str or list[str]], optional
            Type of product. A list will look for multiple filetypes.
        sequence : int or list[int]], optional
            Sequence number refers to "quarter" for Kepler, "campaign" for K2, and "sector" for TESS.
        limit : int, optional
            _description_, by default None
        inplace : bool, optional
            _description_, by default False

        Returns
        -------
        MASTSearch or None
            Returns a filtered MASTSearch object or None if `inplace=True`
        """

        mask = self._filter(
            target_name=target_name,
            pipeline=pipeline,
            mission=mission,
            exptime=exptime,
            distance=distance,
            year=year,
            description=description,
            filetype=filetype,
            sequence_number=sequence,
        )

        if limit is not None:
            cusu = np.cumsum(mask)
            if max(cusu) > limit:
                mask = mask & (cusu <= limit)

        if inplace:
            self.table = self.table[mask].reset_index()
        else:
            return self._mask(mask)

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
        logging.getLogger("astropy").setLevel(log.getEffectiveLevel())
        logging.getLogger("astroquery").setLevel(log.getEffectiveLevel())

        # We don't want to query cloud_uri if we don't have to
        # First check to see if we're not downloading on a cloud platform
        # If not - cloud_uris should have already been queried - in that case
        # check to see if a cloud_uri exists, if so we just pass that

        download = True
        if not conf.DOWNLOAD_CLOUD:
            if pd.notna(row["cloud_uri"]):
                download = False
        if conf.DOWNLOAD_CLOUD or download:
            manifest = Observations.download_products(
                Table().from_pandas(row.to_frame(name=" ").transpose()),
                download_dir=download_dir,
                cache=cache,
                cloud_only=cloud_only,
            )
            manifest = manifest.to_pandas()
        else:
            manifest = pd.DataFrame(
                {
                    "Local Path": [row["cloud_uri"]],
                    "Status": ["COMPLETE"],
                    "Message": ["Link to S3 bucket for remote read"],
                    "URL": [None],
                }
            )

        return manifest

    def download(
        self,
        cloud: bool = conf.PREFER_CLOUD,
        cache: bool = True,
        cloud_only: bool = conf.CLOUD_ONLY,
        download_dir: str = config.get_cache_dir(),
        remove_incomplete: str = True,
    ) -> pd.DataFrame:
        """downloads products in self.table to the local hard-drive

        Parameters
        ----------
        cloud : bool, optional
            enable cloud (as opposed to MAST) downloading, by default True
        cloud_only : bool, optional
            download only products availaible in the cloud, by default False
        download_dir : str, optional
            directory where the products should be downloaded to,
            by default default_download_dir
            cache : bool, optional
        passed to `~astroquery.mast.Observations.download_products`, by default True
            if False, will overwrite the file to be downloaded (for example to replace a corrrupted file)
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

        if (not conf.DOWNLOAD_CLOUD) and ("cloud_uri" not in self.table.columns):
            self.table = self._add_s3_url_column(self.table)

        manifest = [
            self._download_one(row, cloud_only, cache, download_dir)
            for _, row in tqdm(
                self.table.iterrows(),
                total=self.table.shape[0],
                desc="Downloading products",
            )
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
        manifest = manifest.reset_index(drop=True)
        return manifest
