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


class K2Search(MASTSearch):
    """
    Search Class that queries mast for observations performed by the K2
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
    campaign: Optional[int] = None,
        K2 Observing Campaign for which to search for data 
    """
    _REPR_COLUMNS = [
        "target_name",
        "pipeline",
        "mission",
        "campaign",
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
        campaign: Optional[int] = None,
    ):
        super().__init__(
            target=target,
            mission=["K2"],
            obs_table=obs_table,
            prod_table=prod_table,
            table=table,
            search_radius=search_radius,
            exptime=exptime,
            pipeline=pipeline,
            sequence=campaign,
        )

        if table is None:
            self._add_K2_mission_product()
            self._fix_K2_sequence()
            self._sort_K2()

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
        """ Was a K2 target ID passed? """
        return re.match(r"^(ktwo|epic) ?(\d+)$", target)

    def _target_to_exact_name(self, target):
        "parse K2 TIC to exact target name"
        return f"ktwo{target.group(2).zfill(9)}"

#
    def _add_K2_mission_product(self):
        # Some products are HLSPs and some are mission products
        mission_product = np.zeros(len(self.table), dtype=bool)
        mission_product[self.table["pipeline"] == "K2"] = True
        self.table["mission_product"] = mission_product

    def _fix_K2_sequence(self):
        # K2 campaigns 9, 10, and 11 were split into two sections
        # list these separately in the table with suffixes "a" and "b"
        seq_num = self.table["sequence_number"].values.astype(str)

        mask = self.table["sequence_number"].isin([9, 10, 11])

        for index, row in self.table[mask].iterrows():
            for half, letter in zip([1, 2], ["a", "b"]):
                if f"c{row['sequence_number']}{half}" in row["productFilename"]:
                    seq_num[index] = f"{int(row['sequence_number']):02d}{letter}"

        self.table["campaign"] = seq_num

    def _sort_K2(self):
        # No specific preference for K2 HLSPs
        sort_priority = {
            "K2": 1,
        }
        df = self.table
        df["sort_order"] = self.table["pipeline"].map(sort_priority).fillna(9)
        df = df.sort_values(
            by=["distance", "sort_order", "campaign", "pipeline","exptime"], ignore_index=True
        )
        self.table = df

    def filter_table(self, 
            limit: int = None, 
            exptime: Union[int, float, tuple, type(None)] = None,  
            pipeline: Union[str, list[str]] = None,
            campaign: Union[int, list] = None,
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
        campaign : Optional[int], optional
            K2 observing campaign(s), by default None

        Returns
        -------
        K2Search object with updated table. 
        """
        mask = np.ones(len(self.table), dtype=bool)

        if exptime is not None:
            mask = mask & self._mask_by_exptime(exptime) 
        if pipeline is not None:
            mask = mask & self.table['pipeline'].isin(np.atleast_1d(pipeline))
        if campaign is not None:
            mask = mask & self.table['sequence_number'].isin(np.atleast_1d(campaign))
        if limit is not None:
            cusu = np.cumsum(mask)
            if max(cusu) > limit:
                mask = mask & (cusu <= limit)
        return self._mask(mask)
    