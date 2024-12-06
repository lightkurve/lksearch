.. image:: https://github.com/lightkurve/lksearch/actions/workflows/python-app.yml/badge.svg
    :target: https://github.com/lightkurve/lksearch/actions/workflows/python-app.yml
    :alt: Test status

.. image:: https://badge.fury.io/py/lksearch.svg
    :target: https://badge.fury.io/py/lksearch
    :alt: PyPI version

.. image:: https://img.shields.io/badge/documentation-live-blue.svg
    :target: https://lightkurve.github.io/lksearch/
    :alt: Documentation

########
lksearch
########

.. <!-- intro content start -->

**Helpful package to search for TESS/Kepler/K2 data**

``lksearch`` is a community developed, open source Python package that offers a user-friendly approach to searching the `Barbara A. Mikulski Archive for Space Telescopes (MAST) <https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html>`_ web portal for scientific data and mission products from **NASA's TESS, K2, and Kepler missions**.  
This package aims to lower the barrier for students, astronomers, and citizen scientists interested in analyzing time-series data from these NASA missions. 
It does this by providing a set of streamlined classes with simplified inputs and outputs that wraps `Astroquery's <https://astroquery.readthedocs.io/en/latest/#>`_ `MAST.Observations <https://astroquery.readthedocs.io/en/latest/mast/mast_obsquery.html>`_ class with user-friendly post-processing of observation tables and convenient bundled download methods.

.. <!-- intro content end -->

.. <!-- quickstart content start -->

Quickstart
==========

The easiest way to install ``lksearch`` and all of its dependencies is to use the ``pip`` command,
which is a standard part of all Python distributions. (upon release)

To install ``lksearch``, run the following command in a terminal window:

.. code-block:: console

  $ python -m pip install lksearch --upgrade

The ``--upgrade`` flag is optional, but recommended if you already
have ``lksearch`` installed and want to upgrade to the latest version.

Depending on the specific Python environment, you may need to replace ``python``
with the correct Python interpreter, e.g., ``python3``.


Search package for finding and retrieving TESS/Kepler/K2 mission data
---------------------------------------------------------------------

This package is a stand-alone implementation of the lightkurve search functionalty. 
While this package shares many common features to the lightkurve.search module, it has many major changes, as described below. 


Usage
-----

.. code-block:: python

  from lksearch import MASTSearch, TESSSearch, KeplerSearch, K2Search
  ### Get long-cadence target pixel files for Kepler 
  res = search.KeplerSearch("KIC 11904151", exptime="long").cubedata
  ### Get TESScut cutouts for a particular target and sector
  res = TESSSearch("TOI 2257").tesscut
  res.download()

.. <!-- quickstart content end -->

.. <!-- Contributing content start -->

Contributing
============

``lksearch``  is an open-source, community driven package. 
We welcome users to contribute and develop new features for lksearch.  

For further information, please see the `Lightkurve Community guidelines <https://docs.lightkurve.org/development/contributing.html>`_.

.. <!-- Contributing content end -->

.. <!-- Citing content start -->

Citing
======

If you find ``lksearch`` useful in your research, please cite it and give us a GitHub star!

If you use Lightkurve for work or research presented in a publication, we request the following acknowledgment or citation:

`This research made use of Lightkurve, a Python package for Kepler and TESS data analysis (Lightkurve Collaboration, 2018).`

See full citation instuctions, including dependencies, in the `Lightkurve documentation <https://docs.lightkurve.org/about/citing.html>`_. 

.. <!-- Citing content end -->

.. <!-- Contact content start -->

Contact
=======
``lksearch`` is an open source community project created by the `TESS Science Support Center`_. 
The best way to contact us is to `open an issue`_ or to e-mail tesshelp@bigbang.gsfc.nasa.gov.
  
  .. _`TESS Science Support Center`: https://heasarc.gsfc.nasa.gov/docs/tess/
  
  .. _`open an issue`: https://github.com/lightkurve/lksearch/issues/new

Please include a self-contained example that fully demonstrates your problem or question.

.. <!-- Contact content end -->

.. <!-- Changelog content start -->

Changelog:
==========
v1.1.0
  - Added ability to query catalogs using the catalogsearch module. This includes:
    - querying vizier for a region for sources using query_region
    - querying vizier for a catlog for rows corresponding to ids using query_id
    - querying simbad for alternative names for a given name using query_names
    - matching alternative names for sources to catalog stings using match_names_catalogs
  - Added documentation for catalogsearch in docs/tutorials/catalog-searches.ipynb
  - Broke previous tutorial doucmentation into multiple files:
    - mission-search focussed data-searches.ipynb
    - cloud-searches and configuration options in cloud-data-searches.ipynb
v1.0.1
  - Now resolving target search strings with MastClass [#27]
v1.0
  - The class structure has been modified. The base class is MASTSearch. Users are intended to use mission-specific classes (KeplerSearch/K2Search/TESSSearch) to obtain mission-specific results.
  - Result tables are saved as pandas dataframs
  - The TESScut search functionality now uses tesswcs to identify observed sectors
  - Data products are now generalized (timeseries contains lightcurve products, cubedata contains target pixel files and TESSCut, and dvreports contains pdfs contining data validation reports) 
  - 'download' now defaults to the AWS cloud storage. 
  - 'download' only downloads files to disk. It no longer returns a lightkurve object. 
 
.. <!-- Changelog content end -->
