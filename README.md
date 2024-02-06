# Search package for finding and retrieving TESS/Kepler/K2 mission data
This package is a stand-alone implementation of the lightkurve search functionalty modernized for the 2024 data environment.  

## Changes Include:
  - This redevelopment uses pandas dataframs as the back-end for storing search results and mission tables.
  - Its cloud-first, defaulting to retrieve mast products from aws buckets where available, and returning S3 bucket URI's as part of the table results.
  - We've replaced the TESScut search functionality querying MAST FFI products and constructing a sector list) with a tesswcs implementation to identify observed sectors
  - We've unified TESSCut and TargetPixelFile search functionality into search_cubedata,  search_lightcurve has been renamed search_timeseries for thematic resonance.
  - we've deprecated download_all, just use download. it wraps download_one.  
  - TBD - replace astrocut TESScut api query with a URL based API Query?
  - (Not Yet Implemented) The intent is to include additional memmory caching options (not implemented yet) , and potentially cacheless options.


## Usage
  from newlk_search import search
  result = search.search_cubedata("KIC 11904151", mission="Kepler", cadence="long")

  result = search.search_timeseries("297.5835, 40.98339", quarter=6, author="Kepler")
  
  result.download()

  ## Documentation 
  this should probably exist

  ## Contact
  Please Don't

Mermaid Test
```mermaid
graph TD
subgraph sg1[class Search_Mission]
Search_Mission --> Search_Timeseries
Search_Mission --> Search_Cubedata

Search_Timeseries --> Search_Timeseries_MissionProducts
Search_Timeseries --> Search_Timeseries_HLSP

Search_Cubedata --> Search_CubeData_MissionProducts
Search_Cubedata --> Search_CubeData_HLSP
end

subgraph  sg2[class Search_Kepler]
Search_Kepler --> SK_T[Search_Timeseries]
SK_T --> SK_T_M[Search_Timeseries_MissionProducts]
SK_T --> SK_T_H[Search_Timeseries_HLSP]

Search_Kepler --> SK_C[Search_Cubedata]
SK_C --> SK_TP[Search_TargetPixelFile]
SK_C --> SK_TC[Search_TESSCut]
end

subgraph  sg3[class Search_K2]
Search_K2 --> SK2_T[Search_Timeseries]
SK2_T --> SK2_T_M[Search_Timeseries_MissionProducts]
SK2_T --> SK2_T_H[Search_Timeseries_HLSP]

Search_K2 --> SK2_C[Search_Cubedata]
SK2_C --> SK2_TP[Search_TargetPixelFile]
SK2_C --> SK2_TC[Search_TESSCut]
end

subgraph  sg4[class Search_TESS]
Search_TESS --> ST_T[Search_Timeseries]
ST_T --> ST_T_M[Search_Timeseries_MissionProducts]
ST_T --> ST_T_H[Search_Timeseries_HLSP]

Search_TESS --> ST_C[Search_Cubedata]
ST_C --> ST_TP[Search_TargetPixelFile]
ST_C --> ST_TC[Search_TESSCut]

ST_TP --> ST_TP_M[Search_TPF_MissionProducts]
ST_TP --> ST_TP_H[Search_TPF_HLSP]

ST_TC --> ST_TC_M[Search_TESSCut_MissionProducts]
ST_TC --> ST_TC_H[Search_TESSCut_HLSP]
end

sg1 --> sg2
sg1 --> sg3
sg1 --> sg4
```
