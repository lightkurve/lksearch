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
from .MASTSearch import MASTSearch
from . import PACKAGEDIR, PREFER_CLOUD, DOWNLOAD_CLOUD, conf, config

pd.options.display.max_rows = 10

default_download_dir = config.get_cache_dir()

log = logging.getLogger(__name__)


class TESSSearch(MASTSearch):
    """
    Search Class that queries mast for observations performed by the TESS
    Mission, and returns the results in a convenient table with options to download.
    By default mission products and HLSPs are returned.

    Parameters
    ----------
    target: Optional[Union[str, tuple[float], SkyCoord]] = None
        The target to search for observations of. Can be provided as a name (string),
        coordinates in decimal degrees (tuple), or Astropy `~astropy.coordinates.SkyCoord` Object.
    obs_table:Optional[pd.DataFrame] = None
        Optionally, can provice a Astropy `~astropy.table.Table` Object from
        AstroQuery `astroquery.mast.Observations.query_criteria` which will be used to construct the observations table
    prod_table:Optional[pd.DataFrame] = None
        Optionally, if you provide an obs_table, you may also provide a products table of assosciated products.  These
        two tables will be concatenated to become the primary joint table of data products.
    table:Optional[pd.DataFrame] = None
        Optionally, may provide an astropy `~astropy.table.Table` Object  that is the already merged joint table of obs_table
        and prod_table.
    search_radius:Optional[Union[float,u.Quantity]] = None
        The radius around the target name/location to search for observations.  Can be provided in arcseconds (float) or as an
        astropy `~astropy.units.Quantity` Object
    exptime:Optional[Union[str, int, tuple]] = (0,9999)
        Exposure time to filter observation results on.  Can be provided as a mission-specific string,
        an int which forces an exact match to the time in seconds, or a tuple, which provides a range to filter on.
    mission: Optional[Union[str, list[str]]] = ["Kepler", "K2", "TESS"]
        Mission(s) for which to search for data on
    pipeline:  Optional[Union[str, list[str]]] = ["Kepler", "K2", "SPOC"]
        Pipeline(s) which have produced the observed data
    sector: Optional[int] = None,
        TESS Observing Sector for which to search for data. In the initial search, only a single sector can be used. However, you can later use search_result.filter_table(sector=[1,2]) to access a specific subset of sectors.
    """

    _REPR_COLUMNS = [
        "target_name",
        "pipeline",
        "mission",
        "sector",
        "exptime",
        "distance",
        "year",
        "description",
    ]

    def __init__(
        self,
        target: Optional[Union[str, tuple[float], SkyCoord]] = None,
        obs_table: Optional[pd.DataFrame] = None,
        prod_table: Optional[pd.DataFrame] = None,
        table: Optional[pd.DataFrame] = None,
        search_radius: Optional[Union[float, u.Quantity]] = None,
        exptime: Optional[Union[str, int, tuple]] = (0, 9999),
        pipeline: Optional[Union[str, list[str]]] = None,
        sector: Optional[int] = None,
        hlsp: bool = True,
    ):
        if hlsp is False:
            pipeline = ["SPOC", "TESS-SPOC", "TESScut"]

        super().__init__(
            target=target,
            mission=["TESS"],
            obs_table=obs_table,
            prod_table=prod_table,
            table=table,
            search_radius=search_radius,
            exptime=exptime,
            pipeline=pipeline,
            sequence=sector,
        )
        if table is None:
            if ("TESScut" in np.atleast_1d(pipeline)) or (type(pipeline) is type(None)):
                self._add_tesscut_products(sector)
            self._add_TESS_mission_product()
            self._sort_TESS()

    @property
    def HLSPs(self):
        """return a MASTSearch object with self.table only containing High Level Science Products"""
        mask = self.table["mission_product"]
        return self._mask(~mask)

    @property
    def mission_products(self):
        """return a MASTSearch object with self.table only containing Mission Products"""
        mask = self.table["mission_product"]
        return self._mask(mask)

    @property
    def tesscut(self):
        """return the TESScut only data"""
        mask = self.table["pipeline"] == "TESScut"
        return self._mask(mask)

    @property
    def cubedata(self):
        """return a MASTSearch object with self.table only containing products that are image cubes"""
        mask = self._mask_product_type("target pixel") | (
            self.table["pipeline"] == "TESScut"
        )

        # return self._cubedata()
        return self._mask(mask)

    def _check_exact(self, target):
        """Was a TESS target ID passed?"""
        return re.match(r"^(tess|tic) ?(\d+)$", target)

    def _target_to_exact_name(self, target):
        "parse TESS TIC to exact target name"
        return f"{target.group(2)}"

    def _add_TESS_mission_product(self):
        """Determine whick products are HLSPs and which are mission products"""
        mission_product = np.zeros(len(self.table), dtype=bool)
        mission_product[self.table["pipeline"] == "SPOC"] = True
        self.table["mission_product"] = mission_product
        self.table["sector"] = self.table["sequence_number"]

    def _add_tesscut_products(self, sector_list: Union[int, list[int]]):
        """Add tesscut product information to the search table

        Parameters
        ----------
        sector_list : int, list[int]
            list of sectors to search for tesscut observations of
        """

        # get the ffi info for the targets
        tesscut_info = self._get_tesscut_info(sector_list)
        # add the ffi info to the table
        self.table = pd.concat([self.table, tesscut_info])

    # FFIs only available when using TESSSearch.
    # Use TESS WCS to just return a table of sectors and dates?
    # Then download_ffi requires a sector and time range?

    def _get_tesscut_info(self, sector_list: Union[int, list[int]]):
        """Get the tesscut (TESS FFI) obsering information for self.target
        for a particular sector(s)

        Parameters
        ----------
        sector_list : Union[int, list[int]]
            Sector(s) to search for TESS ffi observations

        Returns
        -------
        tesscut_results: pd.DataFrame
            table containing information on sectors in which TESS FFI data is available
        """
        from tesswcs import pointings
        from tesswcs import WCS

        log.debug("Checking tesswcs for TESSCut cutouts")
        tesscut_desc = []
        tesscut_mission = []
        tesscut_tmin = []
        tesscut_tmax = []
        tesscut_exptime = []
        tesscut_seqnum = []
        tesscut_year = []

        # Check each sector / camera / ccd for observability
        # Submit a tesswcs PR for to convert table to pandas
        pointings = pointings.to_pandas()

        if sector_list is None:
            sector_list = pointings["Sector"].values

        for _, row in pointings.iterrows():
            tess_ra = row["RA"]
            tess_dec = row["Dec"]
            tess_roll = row["Roll"]
            sector = row["Sector"].astype(int)

            if sector in np.atleast_1d(sector_list):
                AddSector = False
                for camera in np.arange(1, 5):
                    for ccd in np.arange(1, 5):
                        # predict the WCS
                        wcs = WCS.predict(
                            tess_ra, tess_dec, tess_roll, camera=camera, ccd=ccd
                        )
                        # check if the target falls inside the CCD
                        if wcs.footprint_contains(self.SkyCoord):
                            AddSector = True

                if AddSector:
                    log.debug(
                        f"Target Observable in Sector {sector}, Camera {camera}, CCD {ccd}"
                    )
                    tesscut_desc.append(f"TESS FFI Cutout (sector {sector})")
                    tesscut_mission.append(f"TESS Sector {sector:02d}")
                    tesscut_tmin.append(
                        row["Start"] - 2400000.5
                    )  # Time(row[5], format="jd").iso)
                    tesscut_tmax.append(
                        row["End"] - 2400000.5
                    )  # Time(row[6], format="jd").iso)
                    tesscut_exptime.append(self._sector2ffiexptime(sector))
                    tesscut_seqnum.append(sector)
                    tesscut_year.append(
                        int(np.floor(Time(row["Start"], format="jd").decimalyear))
                    )

        # Build the FFI dataframe from the observability
        n_results = len(tesscut_seqnum)
        tesscut_result = pd.DataFrame(
            {
                "description": tesscut_desc,
                "mission": tesscut_mission,
                "target_name": [self.target_search_string] * n_results,
                "targetid": [self.target_search_string] * n_results,
                "t_min": tesscut_tmin,
                "t_max": tesscut_tmax,
                "exptime": tesscut_exptime,
                "productFilename": ["TESScut"] * n_results,
                "provenance_name": ["TESScut"] * n_results,
                "pipeline": ["TESScut"] * n_results,
                "distance": [0] * n_results,
                "sequence_number": tesscut_seqnum,
                "project": ["TESS"] * n_results,
                "obs_collection": ["TESS"] * n_results,
                "year": tesscut_year,
            }
        )

        if len(tesscut_result) > 0:
            log.debug(f"Found {n_results} matching cutouts.")
        else:
            log.debug("Found no matching cutouts.")

        return tesscut_result

    def _sector2ffiexptime(self, sector: Union[int, list[int]]):
        """lookup table for exposure time based off of sector number

        Parameters
        ----------
        sector : Union[int, list[int]]
            sector(s) to get exposure times for

        Returns
        -------
        int
            exposure time in seconds
        """

        # Determine what the FFI cadence was based on sector
        if sector < 27:
            return 1800
        elif (sector >= 27) & (sector <= 55):
            return 600
        elif sector >= 56:
            return 200

    def _sort_TESS(self):
        """Sort Priority for TESS Observations"""
        # Sort TESS results so that SPOC products appear at the top
        sort_priority = {
            "SPOC": 1,
            "TESS-SPOC": 2,
            "TESScut": 3,
        }

        df = self.table
        df["sort_order"] = df["pipeline"].map(sort_priority).fillna(9)
        df = df.sort_values(
            by=["distance", "sort_order", "sector", "pipeline", "exptime"],
            ignore_index=True,
        )
        self.table = df

    def search_individual_ffi(
        self,
        tmin: Union[float, Time],
        tmax: Union[float, Time],
        search_radius: Union[float, u.Quantity] = 0.0001 * u.arcsec,
        exptime: Union[str, int, tuple] = (0, 9999),
        sector: Union[int, type[None]] = None,
        **extra_query_criteria,
    ):
        """Search for a particular FFI file given a time range, return the product list
        of FFIs for that this target and time range


        Parameters
        ----------
        tmin : Union[float,Time]
            minimum start time of the FFI
        tmax : Union[float,Time]
            maximum end time of the ffi
        search_radius : Union[float, u.Quantity], optional
            radius around target to search for FFIs, by default 0.0001*u.arcsec
        exptime : Union[str, int, tuple], optional
            exposure time of FFI's to search for, by default (0, 9999)
        sector : Union[int, type[None]]
            sector(s) in which to search for FFI files, by default None

        Returns
        -------
        TESSSearch
            TESSSearch object that contains a joint table of FFI info
        """

        query_criteria = {"project": "TESS", **extra_query_criteria}
        query_criteria["provenance_name"] = "SPOC"
        query_criteria["dataproduct_type"] = "image"

        if type(tmin) == type(Time):
            tmin = tmin.mjd

        if type(tmax) == type(Time):
            tmax = tmax.mjd

        query_criteria["t_min"] = (tmin, tmax)
        query_criteria["t_max"] = (tmin, tmax)

        if not isinstance(search_radius, u.quantity.Quantity):
            log.warning(
                f"Search radius {search_radius} units not specified, assuming arcsec"
            )
            search_radius = search_radius * u.arcsec

        if sector is not None:
            query_criteria["sequence_number"] = sector

        if exptime is not None:
            query_criteria["t_exptime"] = exptime

        query_criteria["radius"] = str(search_radius.to(u.deg))

        ffi_obs = Observations.query_criteria(
            objectname=self.target_search_string,
            **query_criteria,
        )

        ffi_products = Observations.get_product_list(ffi_obs)
        # filter out uncalibrated FFIs & theoretical potential HLSP
        prod_mask = ffi_products["calib_level"] == 2
        ffi_products = ffi_products[prod_mask]

        new_table = deepcopy(self)

        # Unlike the other products, ffis don't map cleanly via obs_id as advertised, so we invert and add specific column info
        new_table.obs_table = ffi_products.to_pandas()
        new_table.obs_table["year"] = np.nan

        new_table.prod_table = ffi_obs.to_pandas()
        new_table.table = None

        test_table = new_table._join_tables()
        test_table.reset_index(inplace=True)
        new_table.table = new_table._update_table(test_table)
        new_table.table.reset_index(inplace=True)
        new_table.table["target_name"] = new_table.obs_table["obs_id"]
        new_table.table["obs_collection"] = ["TESS"] * len(new_table.table)
        new_table.table["pipeline"] = [
            new_table.prod_table["provenance_name"].values[0]
        ] * len(new_table.table)
        new_table.table["exptime"] = new_table.table["obs_id"].apply(
            (lambda x: self._sector2ffiexptime(int(x.split("-")[1][1:])))
        )
        new_table.table["year"] = new_table.table["obs_id"].apply(
            (lambda x: int(x.split("-")[0][4:8]))
        )
        new_table.table["sector"] = new_table.table["obs_id"].apply(
            lambda x: int(x.split("-")[1][1:])
        )
        return new_table

    def filter_table(
        self,
        limit: int = None,
        exptime: Union[int, float, tuple, type(None)] = None,
        pipeline: Union[str, list[str]] = None,
        sector: Union[int, list[int]] = None,
    ):
        """
        Filters the search result table by specified parameters

        Parameters
        ----------
        limit : int, optional
            limit to the number of results, by default None
        exptime : Union[int, float, tuple, type, optional
            exposure times to filter by, by default None
        pipeline : Union[str, list[str]], optional
            pipeline used for data reduction, by default None
        sector : Optional[int], optional
            TESS observing sector(s), by default None

        Returns
        -------
        TESSSearch object with updated table
        """
        mask = np.ones(len(self.table), dtype=bool)

        if exptime is not None:
            mask = mask & self._mask_by_exptime(exptime)
        if pipeline is not None:
            mask = mask & self.table["pipeline"].isin(np.atleast_1d(pipeline))
        if sector is not None:
            mask = mask & self.table["sequence_number"].isin(np.atleast_1d(sector))
        if limit is not None:
            cusu = np.cumsum(mask)
            if max(cusu) > limit:
                mask = mask & (cusu <= limit)
        return self._mask(mask)

    def download(
        self,
        cloud: PREFER_CLOUD = True,
        cache: PREFER_CLOUD = True,
        cloud_only: PREFER_CLOUD = False,
        download_dir: PACKAGEDIR = default_download_dir,
        # TESScut_product="SPOC",
        TESScut_size=10,
    ):
        mast_mf = []
        tesscut_mf = []
        manifest = []
        if "TESScut" not in self.table.provenance_name.unique():
            mast_mf = super().download(cloud, cache, cloud_only, download_dir)

        elif "TESScut" in self.table.provenance_name.unique():
            TESSCut_dir = f"{default_download_dir}/mastDownload/TESSCut"
            if not os.path.isdir(TESSCut_dir):
                os.makedirs(TESSCut_dir)
            mask = self.table["provenance_name"] == "TESScut"
            sector_list = self.table.loc[mask]["sequence_number"].values
            if np.any(~mask):
                mast_mf = self._mask(~mask).download()
            from astroquery.mast import Tesscut

            # if cloud:
            #    Tesscut.enable_cloud_dataset()
            tesscut_mf = [
                Tesscut.download_cutouts(
                    coordinates=self.SkyCoord,
                    size=TESScut_size,
                    sector=sector,
                    # Uncomment when astroquery 0.4.8 is released to enable TICA support
                    # product=TESScut_product,
                    # verbose=False
                    path=f"{default_download_dir}/mastDownload/TESSCut",
                    inflate=True,
                    moving_target=False,  # this could be added
                    mt_type=None,
                ).to_pandas()
                # for sector in sector_list
                for sector in tqdm(
                    sector_list, total=len(sector_list), desc="TESScut          "
                )
            ]
        if len(mast_mf) != 0:
            manifest = mast_mf

        if len(tesscut_mf) != 0:
            tesscut_mf = pd.concat(tesscut_mf, ignore_index=True)
            # Check to see if files exist, is so mark complete
            tesscut_mf["Status"] = tesscut_mf["Local Path"].apply(
                lambda x: "COMPLETE" if os.path.isfile(x) else "504"
            )

            if len(manifest) != 0:
                manifest = pd.concat([manifest, tesscut_mf], ignore_index=True)
            else:
                manifest = tesscut_mf

        return manifest
