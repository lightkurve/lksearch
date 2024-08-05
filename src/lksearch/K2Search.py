import pandas as pd
from typing import Union, Optional
import re

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from .MASTSearch import MASTSearch

pd.options.display.max_rows = 10


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
        AstroQuery `~astroquery.mast.Observations.query_criteria` which will be used to construct the observations table
    prod_table:Optional[pd.DataFrame] = None
        Optionally, if you provide an obs_table, you may also provide a products table of assosciated products.  These
        two tables will be concatenated to become the primary joint table of data products.
    table:Optional[pd.DataFrame] = None
        Optionally, may provide an astropy `~astropy.table.Table` Object  that is the already merged joint table of obs_table
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
    campaign: Optional[Union[int, list[int]]]  = None,
        K2 Observing Campaign(s) for which to search for data.
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
        campaign: Optional[Union[int, list[int]]] = None,
        hlsp: bool = True,
    ):
        if hlsp is False:
            pipeline = ["K2"]
            self.mission_search = ["K2"]
        else:
            self.mission_search = ["K2", "HLSP"]
        super().__init__(
            target=target,
            mission=self.mission_search,
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

    def _check_exact(self, target):
        """Was a K2 target ID passed?"""
        return re.match(r"^(ktwo|epic) ?(\d+)$", target)

    def _target_to_exact_name(self, target):
        "parse K2 TIC to exact target name"
        return f"ktwo{target.group(2).zfill(9)}"

    #
    def _add_K2_mission_product(self):
        """Determine whick products are HLSPs and which are mission products"""
        mission_product = np.zeros(len(self.table), dtype=bool)
        mission_product[self.table["pipeline"] == "K2"] = True
        self.table["mission_product"] = mission_product

    def _fix_K2_sequence(self):
        """K2 campaigns 9, 10, and 11 were split into two sections
        # list these separately in the table with suffixes 'a' and 'b'"""
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
            by=["distance", "sort_order", "campaign", "pipeline", "exptime"],
            ignore_index=True,
        )
        self.table = df

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
        campaign: Union[int, list] = None,
        limit: int = None,
        inplace=False,
    ):
        """
        Filters the search result table by specified parameters

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
        campaign : Optional[int], optional
            K2 observing campaign, by default None
        limit : int, optional
            how many rows to return, by default None
        inplace : bool, optional
            whether to modify the KeplerSearch inplace, by default False

        Returns
        -------
        K2Search object with updated table or None if `inplace==True`
        """
        mask = self._filter(
            target_name=target_name,
            filetype=filetype,
            exptime=exptime,
            distance=distance,
            year=year,
            description=description,
            pipeline=pipeline,
            mission=mission,
            sequence_number=campaign,
        )
        if limit is not None:
            cusu = np.cumsum(mask)
            if max(cusu) > limit:
                mask = mask & (cusu <= limit)
        if inplace:
            self.table = self.table[mask].reset_index()
        else:
            return self._mask(mask)
