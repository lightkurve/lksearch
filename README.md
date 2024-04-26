Timeseries Simple Search Combo (TSSC)
# *PRE-RELEASE*
==========
**Helpful package to search for TESS/Kepler/K2 data**

**TSSC** is a community developed, open source Python package that offers a user-friendly approach to searching the [Barbara A. Mikulski Archive for Space Telescopes (MAST)](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html) web portal for scientific data and mission products from **NASA's TESS, K2, and Kepler missions**.  This package aims to lower the barrier for students, astronomers, and citizen scientists interested in analyzing time-series data from these NASA missions. It does this by providing a set of streamlined classes with simplified inputs and outputs that wraps [Astroquery's](https://astroquery.readthedocs.io/en/latest/#) [MAST.Observations](https://astroquery.readthedocs.io/en/latest/mast/mast_obsquery.html) class with user-friendly post-processing of observation tables and convenient bundled download methods.  .  

Documentation
-------------
Read the documentation [here](https://tylerapritchard.github.io/TSSC/).
Check out the tutorial notebook [here](docs/tutorials/Example_searches.ipynb)


Quickstart and Installation
---------------------------

*This package is not currently available on PyPI, but is intended to be pip-installable when made available to users currently we suggest you clone the github and install with poetry* 

The easiest way to install *TSSC* and all of its dependencies is to use the ``pip`` command,
which is a standard part of all Python distributions. (upon release)

To install *TSSC*, run the following command in a terminal window::

    $ python -m pip install TSSC --upgrade

The ``--upgrade`` flag is optional, but recommended if you already
have *newlk_seach* installed and want to upgrade to the latest version.

Depending on the specific Python environment, you may need to replace ``python``
with the correct Python interpreter, e.g., ``python3``.


# Search package for finding and retrieving TESS/Kepler/K2 mission data
This package is a stand-alone implementation of the lightkurve search functionalty  

## Changes Include:
  - The class structure has been modified. The base class is MASTSearch. Users are intended to use mission-specific classes (KeplerSearch/K2Search/TESSSearch) to obtain mission-specific results.
  - Result tables are saved as pandas dataframs
  - The TESScut search functionality now uses tesswcs to identify observed sectors
  - Data products are now generalized (timeseries contains lightcurve products, cubedata contains target pixel files and TESSCut, and dvreports contains pdfs contining data validation reports) 
  - 'download' now defaults to the AWS cloud storage. 
  - 'download' downloads files to disk. It no longer returns a lightkurve object. 
 


Usage
-----
  from TSSC import MASTSearch, TESSSearch, KeplerSearch, K2Search
  ### Get long-cadence target pixel files for Kepler 
    result = search.KeplerSearch("KIC 11904151", exptime="long").cubedata
  ### Get TESScut cutouts for a particular target and sector
    result = TESSSearch("TOI 2257").tesscut
    result.download()

Contributing
------------
We welcome community contributions!
Please read the  guidelines at [TBD](https://heasarc.gsfc.nasa.gov/docs/tess/). 

Citing
------

If you find *newlk_seach* useful in your research, please cite it and give us a GitHub star!
Please read the citation instructions at [TBD](https://heasarc.gsfc.nasa.gov/docs/tess/)


Contact
-------
*TSSC* is an open source community project created by the [TESS Science Support Center](https://heasarc.gsfc.nasa.gov/docs/tess/). The best way to contact us is to [open an issue](https://github.com/lightkurve/lightkurve/issues/new) or to e-mail tesshelp@bigbang.gsfc.nasa.gov.

Please include a self-contained example that fully demonstrates your problem or question.

alternative name pitches:
  - Search Package for TESS Kepler and K2 Surveys (SPEKKS)
  - (Search)SpaceCubes
  - The Terrible Secret of Space