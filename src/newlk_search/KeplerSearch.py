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

from copy import deepcopy

from .utils import SearchError, SearchWarning, suppress_stdout
from .MASTSearch import MASTSearch
from . import PACKAGEDIR, PREFER_CLOUD, DOWNLOAD_CLOUD, conf, config

pd.options.display.max_rows = 10

default_download_dir = config.get_cache_dir()

log = logging.getLogger(__name__)


class KeplerSearch(MASTSearch):
    """
    Search Class that queries mast for observations performed by the Kepler
    Mission, and returns the results in a convenient table with options to download.
    By default both mission products and HLSPs are returned.

    Parameters
    ----------
    target: Optional[Union[str, tuple[float], SkyCoord]] = None
        The target to search for observations of. Can be provided as a name (string),
        coordinates in decimal degrees (tuple), or Astropy `~astropy.coordinates.SkyCoord` Object.
    obs_table:Optional[pd.DataFrame] = None
        Optionally, can provice a Astropy `~astropy.table.Table` Object from
        AstroQuery `astroquery.mast.Observations.query_criteria' which will be used to construct the observations table
    prod_table:Optional[pd.DataFrame] = None
        Optionally, if you provide an obs_table, you may also provide a products table of assosciated products.  These
        two tables will be concatenated to become the primary joint table of data products.
    table:Optional[pd.DataFrame] = None
        Optionally, may provide an stropy `~astropy.table.Table` Object  that is the already merged joint table of obs_table
        and prod_table.
    search_radius:Optional[Union[float,u.Quantity]] = None
        The radius around the target name/location to search for observations.  Can be provided in arcseconds (float) or as an
        AstroPy `~astropy.units.u.Quantity` Object
    exptime:Optional[Union[str, int, tuple]] = (0,9999)
        Exposure time to filter observation results on.  Can be provided as a mission-specific string,
        an int which forces an exact match to the time in seconds, or a tuple, which provides a range to filter on.
    mission: Optional[Union[str, list[str]]] = ["Kepler", "K2", "TESS"]
        Mission(s) for which to search for data on
    pipeline:  Optional[Union[str, list[str]]] = ["Kepler", "K2", "SPOC"]
        Pipeline(s) which have produced the observed data
    quarter: Optional[int] = None,
        Kepler Observing Quarter for which to search for data 
    month: Optional[int] = None,
        Observation month for Kepler
    """
    _REPR_COLUMNS = [
        "target_name",
        "pipeline",
        "mission",
        "quarter",
        "exptime",
        "distance",
        "year",
        "description",
    ]

    def __init__(
        self,
        target: [Union[str, tuple[float], SkyCoord]],
        obs_table: Optional[pd.DataFrame] = None,
        prod_table: Optional[pd.DataFrame] = None,
        table: Optional[pd.DataFrame] = None,
        search_radius: Optional[Union[float, u.Quantity]] = None,
        exptime: Optional[Union[str, int, tuple]] = (0, 9999),
        pipeline: Optional[Union[str, list[str]]] = None,
        quarter: Optional[int] = None,
        month: Optional[int] = None,
    ):
        super().__init__(
            target=target,
            mission=["Kepler"],
            obs_table=obs_table,
            prod_table=prod_table,
            table=table,
            search_radius=search_radius,
            exptime=exptime,
            pipeline=pipeline,
            sequence=None,
        )
        if table is None:
            self._add_kepler_mission_product()
            self._get_sequence_number()
            self._sort_Kepler()
            # Can't search mast with quarter/month directly, so filter on that after the fact.
            self.table = self.table[self._filter_kepler(quarter, month)]

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
    
    def _check_exact(self,target):
        """ Was a Kepler target ID passed? """
        return re.match(r"^(kplr|kic) ?(\d+)$", target)

    def _target_to_exact_name(self, target):
        "parse Kepler TIC to exact target name"
        return f"kplr{target.group(2).zfill(9)}"
#
    def _handle_kbonus(self):
        # KBONUS times are masked as they are invalid for the quarter data
        # kbonus_mask = self.table["pipeline"] == "KBONUS-BKG"

        # self.table['start_time'][kbonus_mask] = something
        raise NotImplementedError

    def _add_kepler_mission_product(self):
        # Some products are HLSPs and some are mission products
        mission_product = np.zeros(len(self.table), dtype=bool)
        mission_product[self.table["pipeline"] == "Kepler"] = True
        self.table["mission_product"] = mission_product


    def _get_sequence_number(self):
        # Kepler sequence_number values were not populated at the time of
        # writing this code, so we parse them from the description field.

        seq_num = self.table["sequence_number"].values.astype(str)

        mask = (self.table["project"] == "Kepler") & self.table[
            "sequence_number"
        ].isna() & ~self.table["description"].str.contains("Data Validation")
        re_expr = r".*Q(\d+)"
        seq_num[mask] = [
            re.findall(re_expr, item[0])[0] if re.findall(re_expr, item[0]) else ""
            for item in self.table.loc[mask, ["description"]].values
        ]

        self.table["sequence_number"] = seq_num
        seq_num = [int(x) if x != '<NA>' else 99 for x in seq_num]
        # Create a 'Quarter' column
        self.table["quarter"] = seq_num 

        '''self.table["mission"] = [
            f"{proj} - Q{seq}"
            if seq !=  '<NA>' else f"{proj}" for proj, seq in zip(self.table["mission"].values.astype(str), seq_num)  
        ]'''

    def _filter_kepler(
        self,
        quarter: Union[int, list[int]] = None,
        month: Union[int, list[int]] = None,
    ) -> pd.DataFrame:
        import os

        # Filter Kepler product by month/quarter
        # TODO: should this return the mask or replace self.table directly? I'm leaning toward replace directly
        products = self.table

        mask = np.ones(len(products), dtype=bool)

        if sum(mask) == 0:
            return products

        # Identify quarter by the description.
        # This is necessary because the `sequence_number` field was not populated
        # for Kepler prime data at the time of writing this function.
        if quarter is not None:
            quarter_mask = np.zeros(len(products), dtype=bool)
            for q in np.atleast_1d(quarter).tolist():
                quarter_mask += products["description"].str.endswith(f"Q{q}")
            mask &= quarter_mask

        # For Kepler short cadence data the month can be specified
        if month is not None:
            month = np.atleast_1d(month).tolist()
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
            is_shortcadence = mask & products["description"].str.contains("Short")

            for idx in np.where(is_shortcadence)[0]:
                quarter = np.atleast_1d(
                    int(
                        products["description"][idx]
                        .split(" - ")[-1]
                        .replace("-", "")[1:]
                    )
                ).tolist()
                date = (
                    products["dataURI"][idx].split("/")[-1].split("-")[1].split("_")[0]
                )
                # Check if the observation date matches the specified Quarter/month from the lookup table
                if (
                    date
                    not in table["StartTime"][
                        table["Month"].isin(month) & table["Quarter"].isin(quarter)
                    ].values
                ):
                    mask[idx] = False
        return mask

    def _sort_Kepler(self):
        sort_priority = {
            "Kepler": 1,
            "KBONUS-BKG": 2,
        }
        df = self.table
        df["sort_order"] = self.table["pipeline"].map(sort_priority).fillna(9)
        df = df.sort_values(
            by=["distance", "sort_order","quarter", "pipeline", "exptime"], ignore_index=True
        )
        self.table = df

    def filter_table(self, 
            limit: int = None, 
            exptime: Union[int, float, tuple, type(None)] = None,  
            pipeline: Union[str, list[str]] = None,
            quarter: Optional[int] = None,
            month: Optional[int] = None,
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
        quarter : Optional[int], optional
            Kepler observing quarter, by default None
        month : Optional[int], optional
            Kepler observing month, by default None

        Returns
        -------
        KeplerSearch object with updated. 
        """
        mask = np.ones(len(self.table), dtype=bool)

        if exptime is not None:
            mask = mask & self._mask_by_exptime(exptime) 
        if pipeline is not None:
            mask = mask & self.table['pipeline'].isin(np.atleast_1d(pipeline))
        if (quarter is not None) | (month is not None):
            mask = mask & self._filter_kepler(quarter=quarter, month=month)
        if limit is not None:
            cusu = np.cumsum(mask)
            if max(cusu) > limit:
                mask = mask & (cusu <= limit)
        return self._mask(mask)