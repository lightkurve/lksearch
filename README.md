# Search package for finding and retrieving TESS/Kepler/K2 mission data
This package is a stand-alone implementation of the lightkurve search functionalty  

## Changes Include:
  - The class structure has been modified. The base class is MASTSearch. Users are intended to use mission-specific classes (KeplerSearch/K2Search/TESSSearch) to obtain mission-specific results.
  - Result tables are saved as pandas dataframs
  - The TESScut search functionality now uses tesswcs to identify observed sectors
  - Data products are now generalized (timeseries contains lightcurve products, cubedata contains target pixel files and TESSCut, and dvreports contains pdfs contining data validation reports) 
  - 'download' now defaults to the AWS cloud storage. 
  - 'download' downloads files to disk. It no longer returns a lightkurve object. 
 


## Usage
  from newlk_search import search
  ### Get long-cadence target pixel files for Kepler 
  result = search.KeplerSearch("KIC 11904151", cadence="long").cubedata
  ### Search for TESS TPFs by coordinate
  result = search.search_timeseries("297.5835, 40.98339", quarter=6, author="Kepler")
  
  result.download()


## Contact
If you encounter a problem, please open an issue.


