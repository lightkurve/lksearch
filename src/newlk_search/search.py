class MASTSearch(object):
    # Shared functions that are used for searches by any mission

    def __init__(self):
        
        self.table = None

    def __repr__(self):
        return f"I'm a class don't index me."


    def _search_timeseries(self, search_name:Union[SkyCoord, str, tuple]): -> pd.DataFrame
        """All mission search
        Updates self.table, and returns it"""
        
        self._munge_table()
        return self.table
    

    def _search_cubedata(self, search_name:Union[SkyCoord, str, tuple]): -> pd.DataFrame
        """All mission search
        Updates self.table, and returns it
        
        MASTSearch.search_timeseries()
        """
        return self.table

    @staticmethod
    def search_timeseries(*args, **kwargs):
        """docstrings"""
        return self._search_timeseries(*args, **kwargs)
    
    @staticmethod
    def search_cubedata(*args, **kwargs):
        """docstrings"""
        return self._search_timeseries(*args, **kwargs)
    
    def search_reports():
        raise NotImplementedError("Use Kepler or TESS or whatever")

    def search_FFI():
        raise NotImplementedError("Those don't exist for everything use TESSSearch")
        
        
    def _parse_input():
        # Is it a skycoord
        # Is it a tuple
        # Is it a string...
    
    def _munge_table(self):
        self._add_columns("something")
        self._add_urls_to_authors()
        self._add_s3_url_column()      
        self._sort_by_priority()
    
    def _add_s3_url_column():
        # self.table would updated to have an extra column of s3 URLS if possible
        raise NotImplementedError
            
    def _query_mast():
        # Constructs the appropriate query

    def _sort_by_priority():
        # Basic sort
        
    def _add_columns():
    
    def _add_urls_to_authors():
    



class TESSSearch(MASTSearch):
     
     #@properties like sector

     def search_cubedata(hlsp=False):
         # _cubedata + _get_ffi

    def _get_ffi():
        # Determines what sectors ffi data is in using tesswcs

    def sort():
        # base sort + TESS HLSP handling?

    def download_ffi():

    def filter_hlsp():


    
class KeplerSearch(Keplersearch):

    #@properties like quarters

    def search_cubedata(hlsp=False):
        # Regular _search_cubedata + processing
    def fix_times():
        # Fixes Kepler times

    def handle_kbonus():
        # Deals with kbonus-specific issues

    def get_sequence_number():


class K2Search(MASTSearch):

    #@properties like campaigns (seasons?)

    def search_cubedata(hlsp=False):
        # Regular _search_cubedata + processing

    def parse_split_campaigns():


# Potential HLSP reader class in a different .py file?
