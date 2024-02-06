
def search_timeseries(mission:['kepler','k2','tess'], quarter=None, campain=None, sector=None, hlsp=True):
    # Append results from search_TESS, search_Kepler, and search_K2

def search_cubedata(mission:['kepler','k2','tess'], quarter=None, campain=None, sector=None, hlsp=True):


class _search_mission():

    # Shared functions that are used for searches by any mission

    def __init__():
        if 'Kepler' in mission:
            return 

    def __repr__():

    def _search_timeseries():

    def _search_cubedata():

    def _query_mast():
        # Constructs the appropriate query

    def _sort():
        # Basic sort

    def _parse_name():

    def _download_one():

    def download():

    



class search_TESS(search, hlsp=False):

     def search_cubedata():
         _cubedata + _get_ffi

    def _get_ffi():
        # Determines what sectors ffi data is in

    def sort():
        # base sort + TESS HLSP handling?

    def download_ffi():

    def filter_hlsp():


    
class search_Kepler(search, hlsp=False):

    def search_cubedata():
        # Regular _search_cubedata + processing
    def fix_times():
        # Fixes Kepler times

    def handle_kbonus():
        # Deals with kbonus-specific issues

    def get_sequence_number():


class search_K2(search, hlsp=False):

    def search_cubedata():
        # Regular _search_cubedata + processing

    def parse_split_campaigns():



