from astroquery.mast import Observations
import pandas as pd
from typing import Union

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

class MASTSearch(object):
    # Shared functions that are used for searches by any mission

    def __init__(self):
        
        self.table = None

        # Search Target Info
        # Do we know the *excacty* MAST-DB Target Search Name 
        # for select Kepler/K2/TESS Targets
        self.exact_target = False
        self.exact_target_name = ""
        
        #Whats the Target Skycoord, if one was given/created
        self.SkyCoord = None
        
        #What Search String was used to search mast, if one was created?
        self.target_search_string = ""

    def __repr__(self):
        return f"I'm a class don't index me."

    def _search_timeseries(self, search_name:Union[SkyCoord, str, tuple]) -> pd.DataFrame:
        """All mission search
        Updates self.table, and returns it"""
        self._parse_input(search_name)
        joint_table = self._query_mast(target)
        self.table = self._munge_table(joint_table)          
        return self.table
    
    def _search_cubedata(self, search_name:Union[SkyCoord, str, tuple]) -> pd.DataFrame:
        """All mission search
        Updates self.table, and returns it
        
        MASTSearch.search_timeseries()
        """
        target = self._parse_input(search_name)
        joint_table = self._query_mast(target)
        self.table = self._munge_table(joint_table)  
        return self.table

    #@staticmethod
    def search_timeseries(self, *args, **kwargs):
        """docstrings"""
        return self._search_timeseries(*args, **kwargs)

    #@staticmethod
    def search_cubedata(self,*args, **kwargs):
        """docstrings"""
        return self._search_timeseries(*args, **kwargs)
    
    def search_reports():
        raise NotImplementedError("Use Kepler or TESS or whatever")

    def search_FFI():
        raise NotImplementedError("Those don't exist for everything use TESSSearch")
        
        
    def _parse_input(self, search_name):
        # Is it a skycoord
        # Is it a tuple
        # Is it a string...

        #We used to allow an int to be sent and do some educated-guess parsing
        #This will no longer work
        if isinstance(search_name, int):
            raise TypeError("Target must be a target name string or astropy coordinate object")
    
        # If passed a SkyCoord, convert it to an "ra, dec" string for MAST
        if isinstance(search_name, SkyCoord):
            self.target_search_string = f"{search_name.ra.deg}, {search_name.dec.deg}"
            self.SkyCoord = search_name
    
        if isinstance(search_name, str):
            self.target_search_string = search_name

        target_lower = str(target).lower()
        # Was a Kepler target ID passed?
        kplr_match = re.match(r"^(kplr|kic) ?(\d+)$", target_lower)
        if kplr_match:
            self.exact_target_name = f"kplr{kplr_match.group(2).zfill(9)}"
            self.exact_target = True

        # Was a K2 target ID passed?
        ktwo_match = re.match(r"^(ktwo|epic) ?(\d+)$", target_lower)
        if ktwo_match:
            self.exact_target_name = f"ktwo{ktwo_match.group(2).zfill(9)}"
            self.exact_target = True

        # Was a TESS target ID passed?
        tess_match = re.match(r"^(tess|tic) ?(\d+)$", target_lower)
        if tess_match:
            self.exact_target_name = f"{tess_match.group(2).zfill(9)}"  
            self.exact_target = True

    def _munge_table(self):
        self._add_columns("something")
        self._add_urls_to_authors()
        self._add_s3_url_column()      
        self._sort_by_priority()
    
    def _add_s3_url_column():
        # self.table would updated to have an extra column of s3 URLS if possible
        raise NotImplementedError
            
    def _query_mast(target):
        # Constructs the appropriate query
        #first name_search
        if(self.exact_target):
            #do an exact name search with target_name= 
            obs = Observations.query_criteria(target_name = self.exact_target_name)
        else:
            obs = Observations.query_criteria(objectname = self.target_search_string)
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
