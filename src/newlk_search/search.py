from astroquery.mast import Observations
import pandas as pd
from typing import Union

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

class MASTSearch(object):
    # Shared functions that are used for searches by any mission

    def __init__(self, table = None):
        if table is None:
            self.table = pd.DataFrame()

        else:
            if isinstance(table, Table):
                log.warning("Search Result Now Expects a pandas dataframe but an astropy Table was given; converting astropy.table to pandas")
                table = table.to_pandas()
            self.table = table
            #if len(table) > 0:
                #self._fix_start_and_end_times()
                #self._add_columns()
                #self._sort_table()
        #self.display_extra_columns = conf.search_result_display_extra_columns


    def __repr__(self):
        return f"I'm a class don't index me."

    def _search_timeseries(self, search_name:Union[SkyCoord, str, tuple]) -> pd.DataFrame:
        """All mission search
        Updates self.table, and returns it"""
        self._parse_input(search_name)
        joint_table = self._query_mast(target)
        self.table = self._munge_table(joint_table)          
        return self.table
    
    def _search_cubedata(self,
        target:  Union[str, tuple, SkyCoord],
        radius:  Union[float, u.Quantity] = None,
        exptime:  Union[str, int, tuple] = None,
        cadence: Union[str, int, tuple] = None,
        mission: Union[str, list[str]] = ["Kepler", "K2", "TESS"],
        author:  Union[str, tuple] = None,
        quarter:  Union[int, list[int]] = None,
        month:    Union[int, list[int]] = None,
        campaign: Union[int, list[int]] = None,
        sector:   Union[int, list[int]] = None,
        limit:    int = None,
        ) -> pd.DataFrame:
        """All mission search
        Updates self.table, and returns it
        
        MASTSearch.search_timeseries()
        """
        self._parse_input(search_name)
        joint_table = self._query_mast(search_name)
        joint_table = self._update_table(joint_table)  
        return joint_table

    #@staticmethod
    def search_timeseries(self, target, **kwargs):

        return MASTSearch(self._search_timeseries(*args, **kwargs))

    #@staticmethod
    def search_cubedata(self,*args, **kwargs):
        """docstrings"""
        return self._search_timeseries(*args, **kwargs)
    
    def search_reports():
        raise NotImplementedError("Use Kepler or TESS or whatever")

    def search_FFI():
        raise NotImplementedError("Those don't exist for everything use TESSSearch")
        
        
    def _parse_input(self, search_name):
       """ Prepares target search name based on input type(Is it a skycoord, tuple, or string..."""

        #We used to allow an int to be sent and do some educated-guess parsing

        # If passed a SkyCoord, convert it to an "ra, dec" string for MAST
        if isinstance(search_name, SkyCoord):
            self.target_search_string = f"{search_name.ra.deg}, {search_name.dec.deg}"
            self.SkyCoord = search_name


        elif isinstance(search_name, tuple):
            self.target_search_string = f"{search_name[0]}, {search_name[1]}"

           
        elif isinstance(search_name, str):
            self.target_search_string = search_name

            target_lower = str(target).lower()
            # Was a Kepler target ID passed?
            kplr_match = re.match(r"^(kplr|kic) ?(\d+)$", target_lower)
            if kplr_match:
                self.exact_target_name = f"kplr{kplr_match.group(2).zfill(9)}"
                #self.exact_target = True

            # Was a K2 target ID passed?
            ktwo_match = re.match(r"^(ktwo|epic) ?(\d+)$", target_lower)
            if ktwo_match:
                self.exact_target_name = f"ktwo{ktwo_match.group(2).zfill(9)}"
                #self.exact_target = True

            # Was a TESS target ID passed?
            tess_match = re.match(r"^(tess|tic) ?(\d+)$", target_lower)
            if tess_match:
                self.exact_target_name = f"{tess_match.group(2).zfill(9)}"  
                #self.exact_target = True
            
        else:
           raise TypeError("Target must be a target name string or astropy coordinate object")
        

    def _update_table(self):
        #rename t_exptime to exptime
        self._add_columns("something")
        self._add_urls_to_authors()
        self._add_s3_url_column()      
        self._sort_by_priority()
    
    def _add_s3_url_column():
        # self.table would updated to have an extra column of s3 URLS if possible
        raise NotImplementedError
            
    def _query_mast(target, **kwargs):
        # Constructs the appropriate query
        #first name_search

        # We pass the following `query_criteria` to MAST regardless of whether
        # we search by position or target name:
        query_criteria = {"project": project, }
        if provenance_name is not None:
            query_criteria["provenance_name"] = provenance_name
        if sequence_number is not None:
            query_criteria["sequence_number"] = sequence_number
        if exptime is not None:
            query_criteria["t_exptime"] = exptime

        if hasattr(self, exact_target_name) and (radius is None):
            log.debug(f"Started querying MAST for observations with exact name {self.exact_target_name}")
            #do an exact name search with target_name= 
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=NoResultsWarning)
                warnings.filterwarnings("ignore", message="t_exptime is continuous")
                obs = Observations.query_criteria(target_name = self.exact_target_name, **query_criteria)
       
            if len(obs) > 0:
                # astroquery does not report distance when querying by `target_name`;
                # we add it here so that the table returned always has this column.
                obs["distance"] = 0.0
                return obs
        else:
            if radius is None:
                radius = 0.0001 * u.arcsec
            elif not isinstance(radius, u.quantity.Quantity):
                log.debug('Radius units not specified, assuming arcsec')
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
                
                    obs = Observations.query_criteria(objectname = self.target_search_string, **query_criteria)
                obs.sort("distance")
                return obs
            except ResolverError as exc:
                # MAST failed to resolve the object name to sky coordinates
                raise SearchError(exc) from exc

        #See if we have 
        if(len(obs) == 0): 
            #check for a radius, add one if not?
            obs = Observations.query_criteria

        #else cone_search
        prod = Observations.get_product_list(obs)
        
        joint_table = pd.concat((obs.to_pandas(), prod.to_pandas()))
        return joint_table

    def _sort_by_priority():
        # Basic sort
        raise NotImplementedError("To Do")
    def _add_columns():
        raise NotImplementedError("To Do")
    
    def _add_urls_to_authors():
        raise NotImplementedError("To Do")
    



class TESSSearch(MASTSearch):
     
    #@properties like sector
    def search_cubedata(hlsp=False):
        # _cubedata + _get_ffi
        raise NotImplementedError

    def _get_ffi():
        # Determines what sectors ffi data is in using tesswcs
        raise NotImplementedError

    def sort():
        # base sort + TESS HLSP handling?
        raise NotImplementedError

    def download_ffi():
        raise NotImplementedError

    def filter_hlsp():
        raise NotImplementedError


    
class KeplerSearch(MASTSearch):

    #@properties like quarters

    def search_cubedata(hlsp=False):
        # Regular _search_cubedata + processing
        raise NotImplementedError

    def fix_times():
        # Fixes Kepler times
        raise NotImplementedError

    def handle_kbonus():
        # Deals with kbonus-specific issues
        raise NotImplementedError

    def get_sequence_number():
        raise NotImplementedError


class K2Search(MASTSearch):

    #@properties like campaigns (seasons?)

    def search_cubedata(hlsp=False):
        # Regular _search_cubedata + processing
        raise NotImplementedError

    def parse_split_campaigns():
        raise NotImplementedError


# Potential HLSP reader class in a different .py file?
