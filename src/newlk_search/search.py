from astroquery.mast import Observations
import pandas as pd
from typing import Union
import re
import logging
import warnings
from lightkurve.utils import  (
    LightkurveDeprecationWarning,
    LightkurveError,
    LightkurveWarning,
    suppress_stdout,
)

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, join


log = logging.getLogger(__name__)

class SearchError(Exception):
    pass

class MASTSearch(object):
    # Shared functions that are used for searches by any mission

    def __init__(self, target):
         

        if isinstance(target, int):
            raise TypeError("Target must be a target name string or astropy coordinate object")
        self.target = target
        #if table is None:
        #    self.table = pd.DataFrame()

        #else:
        #    if isinstance(table, Table):
        #        log.warning("Search Result Now Expects a pandas dataframe but an astropy Table was given; converting astropy.table to pandas")
        #        table = table.to_pandas()
        #    self.table = table
            #if len(table) > 0:
                #self._fix_start_and_end_times()
                #self._add_columns()
                #self._sort_table()
        #self.display_extra_columns = conf.search_result_display_extra_columns


    def __repr__(self):
        return f"I'm a MASTSearch class. You can use me to search for timeseries or cubedata."

    def _search_timeseries(self,  
        target,
        radius=None,
        filetype="Lightcurve",
        mission=["Kepler", "K2", "TESS"],
        author=None,
        cadence=None,
        provenance_name=None,
        exptime=(0, 9999),
        #quarter=None,
        #month=None,
        #campaign=None,
        #sector=None,
        limit=None,):
        """All mission search
        Updates self.table, and returns it"""
        print(f"_search_timeseries {mission}")
        joint_table = self._search_products(target, 
                                            radius=radius,
                                            exptime=exptime,
                                            mission=mission,
                                            filetype=filetype,
                                            provenance_name=author,
                                            #quarter=quarter,
                                            #month=month,
                                            #campaign=campaign,
                                            #sector=sector,
                                            limit=limit,)
        
               
        return joint_table
    
    def _search_cubedata(self,
        target :  Union[str, tuple, SkyCoord],
        radius:  Union[float, u.Quantity] = None,
        exptime:  Union[str, int, tuple] = (0, 9999),
        cadence: Union[str, int, tuple] = None,
        mission: Union[str, list[str]] = ["Kepler", "K2", "TESS"],
        filetype: str = 'Target Pixel',
        author:  Union[str, list[str]] = None,
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
        
        joint_table = self._search_products(**kwargs)
        joint_table = self._update_table(joint_table)  
        return joint_table

    # @staticmethod
    def search_timeseries(self,
        target:  Union[str, tuple, SkyCoord],
        radius:  Union[float, u.Quantity] = None,
        exptime:  Union[str, int, tuple] = (0, 9999),
        cadence: Union[str, int, tuple] = None,
        mission: Union[str, list[str]] = ["Kepler", "K2", "TESS"],
        filetype: str = "Lightcurve",
        author:  Union[str, list[str]] = None,
        # Get rid of this in the default call and implement in the mission-specific version?
        #quarter:  Union[int, list[int]] = None,
        #month:    Union[int, list[int]] = None,
        #campaign: Union[int, list[int]] = None,
        #sector:   Union[int, list[int]] = None,
        limit:    int = None,
        ):
        #if isinstance(target, int):
        #    raise TypeError("Target must be a target name string or astropy coordinate object")
        print(f"Search_timeseries {mission}")
        joint_table = self._search_timeseries(self.target, 
                                         radius=radius, 
                                         exptime=exptime, 
                                         cadence=cadence,
                                         mission=mission,
                                         filetype=filetype,
                                         author=author,
                                         #quarter=quarter,
                                         #month=month,
                                         #campaign=campaign,
                                         #sector=sector,
                                         limit=limit,
                                         )
        joint_table = self._update_table(joint_table)
        
        joint_table = self._filter_products(joint_table)
        self.table = joint_table   
        

    #@staticmethod
    def search_cubedata(self,*args, **kwargs):
        
        """docstrings"""
        '''joint_table = self._update_table(joint_table)
        
        joint_table = self._filter_products(joint_table)
        self.table = joint_table   
        return self._search_timeseries(*args, **kwargs)'''
        raise NotImplementedError("Starting with Timeseries")
    
    def search_reports():
        raise NotImplementedError("Use Kepler or TESS or whatever")

    def search_FFI():
        raise NotImplementedError("Those don't exist for everything use TESSSearch")
        
        
    def _parse_input(self, search_name):
        """ Prepares target search name based on input type(Is it a skycoord, tuple, or string..."""
        # We used to allow an int to be sent and do some educated-guess parsing
        # If passed a SkyCoord, convert it to an "ra, dec" string for MAST
        if isinstance(search_name, SkyCoord):
            self.target_search_string = f"{search_name.ra.deg}, {search_name.dec.deg}"
            self.SkyCoord = search_name
            
        elif isinstance(search_name, tuple):
            self.target_search_string = f"{search_name[0]}, {search_name[1]}"

        elif isinstance(search_name, str):
            self.target_search_string = search_name

            target_lower = str(search_name).lower()
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
        

    def _update_table(self, products):
        #rename t_exptime to exptime
        print("UPDATING TABLE")
        products["exptime"] = products["t_exptime"]
        #rename porvenance_name to author
        #self._add_columns("something")
        #self._add_urls_to_authors()
        #self._add_s3_url_column()      
        #self._sort_by_priority()
        return products
    
    def _add_s3_url_column():
        # self.table would updated to have an extra column of s3 URLS if possible
        raise NotImplementedError
    
    def _search_products(self, target, 
        radius=None,
        filetype="Lightcurve",
        mission=["Kepler", "K2", "TESS"],
        provenance_name=None,
        exptime=(0, 9999),
        quarter=None,
        month=None,
        campaign=None,
        sector=None,
        limit=None,):
        # Helper function that returns a Search Result object containing MAST products
        # combines the results of Observations.query_criteria (called via self.query_mast) and Observations.get_product_list

        if [bool(quarter), 
            bool(campaign),
            bool(sector)].count(True) > 1:
                raise LightkurveError("Ambiguity Error; multiple quarter/campaign/sector specified."
                    "If searching for specific data across different missions, perform separate searches by mission.")
        
        # Is this what we want to do/ where we want the error thrown?
        if filetype == 'ffi':
            raise SearchError(f"FFI search not implemented in MASTSearch. Please use TESSSearch.")
        
        # if a quarter/campaign/sector is specified, search only that mission
        if quarter is not None:
            mission = ["Kepler"]
        if campaign is not None:
            mission = ["K2"]
        if sector is not None:
            mission = ["TESS"]    
        # Ensure mission is a list
        mission = np.atleast_1d(mission).tolist()

        if provenance_name is not None:
            provenance_name = np.atleast_1d(provenance_name).tolist()
            # If author "TESS" is used, we assume it is SPOC
            provenance_name = np.unique(
                [p if p.lower() != "tess" else "SPOC" for p in provenance_name]
            )
            
        # Speed up by restricting the MAST query if we don't want FFI image data
        # At MAST, non-FFI Kepler pipeline products are known as "cube" products,
        # and non-FFI TESS pipeline products are listed as "timeseries"
        extra_query_criteria = {}
        if filetype.lower() == 'lightcurve':
            extra_query_criteria["dataproduct_type"] = ["timeseries"]
        elif filetype.lower() == 'target pixel':
            extra_query_criteria["dataproduct_type"] = ["cube"]
        elif filetype.lower() == 'ffi':
            raise SearchError("Please use TESSSearch.search_ffi to search for full frame images")
        else:
            extra_query_criteria["dataproduct_type"] = ["cube", "timeseries"]

        #from astroquery.mast import Observations
        observations = self._query_mast(
            target,
            radius=radius,
            project=mission,
            provenance_name=provenance_name,
            exptime=exptime,
            sequence_number=campaign or sector,
            **extra_query_criteria,
        )
        log.debug(
            "MAST found {} observations. "
            "Now querying MAST for the corresponding data products."
            "".format(len(observations))
        )
        if len(observations) == 0:
            raise SearchError(f"No data found for target {target}.")

            
        # Use the search result to get a product list
        products = Observations.get_product_list(observations)

        joint_table = join(
            observations,
            products,
            keys="obs_id",#in Kepler, K2 this was parent_id at one point not obs_id
            #keys_left="obsid", 
            #keys_right="parent_obsid",
            join_type="right",
            uniq_col_name="{col_name}{table_name}",
            table_names=["", "_products"],
        )

        joint_table = joint_table.to_pandas()
        joint_table = self._update_table(joint_table)
        print("BEFORE FILTER")
        
        masked_result = self._filter_products(
            joint_table,
            exptime=exptime,
            filetype=filetype,
            project=mission,
            provenance_name=provenance_name,
            limit=limit,
        )
        log.debug(f"MAST found {len(masked_result)} matching data products.")

        

        return joint_table


            
    def _query_mast(self, target: Union[str, SkyCoord],
        radius: Union[float, u.Quantity, None] = None,
        project: Union[str, list[str]] = ["Kepler", "K2", "TESS"],
        provenance_name: Union[str, list[str], None] = None,
        exptime: Union[int, float, tuple] = (0, 9999),
        sequence_number: Union[int, list[int], None] = None,
        **extra_query_criteria):

        from astroquery.exceptions import NoResultsWarning, ResolverError

        #**extra_query_criteria,):
        # Constructs the appropriate query for mast
        print(target, exptime)
        print(project)
        self._parse_input(target)

        # We pass the following `query_criteria` to MAST regardless of whether
        # we search by position or target name:
        query_criteria = {"project": project, **extra_query_criteria}
        if provenance_name is not None:
            query_criteria["provenance_name"] = provenance_name
        if sequence_number is not None:
            query_criteria["sequence_number"] = sequence_number
        if exptime is not None:
            query_criteria["t_exptime"] = exptime

        if hasattr(self, 'exact_target_name') and (radius is None):
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
                print(radius, self.target_search_string, query_criteria)
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
    
    def _add_kepler_sequence_num(self):
        seq_num = self.table["sequence_number"].values.astype(str)

         # Kepler sequence_number values were not populated at the time of
        # writing this code, so we parse them from the description field.
        mask = ((self.table["project"] == "Kepler") &
                self.table["sequence_number"].isna())
        re_expr = r".*Q(\d+)"
        seq_num[mask] = [re.findall(re_expr, item[0])[0] if re.findall(re_expr, item[0]) else "" for item in self.table.loc[mask,["description"]].str.values]




    def _filter_products(self,
            products,
            # campaign: Union[int, list[int]] = None,
            # quarter: Union[int, list[int]] = None,
            # month: Union[int, list[int]]= None,
            # sector: Union[int, list[int]] = None,
            exptime: Union[str, int, tuple[int]] = (0, 9999),
            limit: int = None,
            project: Union[str, list[str]] = ["Kepler", "K2", "TESS"],
            provenance_name: Union[str, list[str]] = ["kepler", "k2", "spoc"],
            filetype: Union[str, list[str]] = ["Target Pixel"], #lightcurve, target pixel, ffi, or dv
        ) -> pd.DataFrame:
        # Modify this so that it can choose what types of products to keep
        print(f"EXPTIME: {exptime}")
        print(filetype)
        mask = np.ones(len(products), dtype=bool)

        # I removed the kepler-specific stuff here

        # HLSP products need to be filtered by extension
        if "lightcurve" in [x.lower() for x in np.atleast_1d(filetype).tolist()]:
            mask &= np.array(
                [uri.lower().endswith("lc.fits") for uri in products["productFilename"]]
            )
        # TODO:The elifs only allow for 1 type (target pixel or ffi), is that the behavior we want?
        if "target pixel" in [x.lower() for x in np.atleast_1d(filetype).tolist()]:
            mask &= np.array(
                [
                    uri.lower().endswith(("tp.fits", "targ.fits.gz"))
                    for uri in products["productFilename"]
                ]
            )
        if "ffi" in [x.lower() for x in np.atleast_1d(filetype).tolist()]:
            mask &= np.array(["TESScut" in desc for desc in products["description"]])

        
        
        if "dv" in [x.lower() for x in np.atleast_1d(filetype).tolist()]:
            mask &= np.array(
                [
                    uri.lower().endswith(("dvr.pdf","dvm.pdf","dvs.pdf"))
                    for uri in products["productFilename"]
                ]
            )
        else: # Allow only fits files if you don't want dv products
            mask &= np.array(
                [
                    uri.lower().endswith("fits") or uri.lower().endswith("fits.gz")
                    for uri in products["productFilename"]
                ]
            )

        # Filter by cadence
        mask &= self._mask_by_exptime(products, exptime)

        products = products[mask]

        products.sort_values(by=["distance", "productFilename"])
        if limit is not None:
            return products[0:limit]
        return products

    def _mask_by_exptime(self, products, exptime):
        """Helper function to filter by exposure time.
        Returns a boolean array """
        mask = np.ones(len(products), dtype=bool)
        if isinstance(exptime, (int, float)):
            mask &= products["exptime"] == exptime
        elif isinstance(exptime, tuple):
            mask &= (products["exptime"] >= min(exptime) & (products["exptime"] <= max(exptime)))
        elif isinstance(exptime, str):
            exptime = exptime.lower()
            if exptime in ["fast"]:
                mask &= products["exptime"] < 60
            elif exptime in ["short"]:
                mask &= (products["exptime"] >= 60) & (products["exptime"] <= 120)
            elif exptime in ["long", "ffi"]:
                mask &= products["exptime"] > 120
        return mask


    



class TESSSearch(MASTSearch):
     
    #@properties like sector
    def search_cubedata(hlsp=False):
        # _cubedata + _get_ffi
        raise NotImplementedError

    def _get_ffi():
        from tesswcs import pointings
        from tesswcs import WCS
        log.debug("Checking tesswcs for TESSCut cutouts")
        tesscut_desc=[]
        tesscut_mission=[]
        tesscut_tmin=[]
        tesscut_tmax=[]
        tesscut_exptime=[]
        tesscut_seqnum=[]

        coords = _resolve_object(target)
        # Check each sector / camera / ccd for observability
        for row in pointings.iterrows():
            tess_ra, tess_dec, tess_roll = row[2:5]
            for camera in np.arange(1, 5):
                for ccd in np.arange(1, 5):
                    # predict the WCS
                    wcs = WCS.predict(tess_ra, tess_dec, tess_roll , camera=camera, ccd=ccd)
                    # check if the target falls inside the CCD
                    if wcs.footprint_contains(coords):
                        sector = row[0]
                        log.debug(f"Target Observable in Sector {sector}, Camera {camera}, CCD {ccd}")
                        tesscut_desc.append(f"TESS FFI Cutout (sector {sector})")
                        tesscut_mission.append(f"TESS Sector {sector:02d}")
                        tesscut_tmin.append(row[5]- 2400000.5) # Time(row[5], format="jd").iso)
                        tesscut_tmax.append(row[6]- 2400000.5) # Time(row[6], format="jd").iso)
                        tesscut_exptime.append(_tess_sector2exptime(sector))
                        tesscut_seqnum.append(sector)
        
        # Build the ffi dataframe from the observability
        n_results = len(tesscut_seqnum)
        ffi_result = pd.DataFrame({"description" : tesscut_desc,
                                          "mission": tesscut_mission,
                                          "target_name" : [str(target)] * n_results,
                                          "targetid" : [str(target)] * n_results,
                                          "t_min" : tesscut_tmin,
                                          "t_max" : tesscut_tmax,
                                          "exptime" : tesscut_exptime,
                                          "productFilename" : ["TESScut"] * n_results,
                                          "provenance_name" : ["TESScut"] * n_results,
                                          "author" : ["TESScut"] * n_results,
                                          "distance" : [0] * n_results,
                                          "sequence_number" : tesscut_seqnum,
                                          "project" : ["TESS"] * n_results,
                                          "obs_collection" : ["TESS"] * n_results})
        
        if len(ffi_result) > 0:
            log.debug(f"Found {n_results} matching cutouts.")
        else:
            log.debug("Found no matching cutouts.")

        return ffi_result
    


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
        # INCLUDE:
        #  _filter_kepler 
        #  fix+times
        #  handle_kbonus
        raise NotImplementedError

    def _fix_times():
        # Fixes Kepler times
        raise NotImplementedError

    def _handle_kbonus():
        # Deals with kbonus-specific issues
        raise NotImplementedError

    def get_sequence_number(self):
        # Kepler sequence_number values were not populated at the time of
        # writing this code, so we parse them from the description field.

        seq_num = self.table["sequence_number"].values.astype(str)

        mask = ((self.table["project"] == "Kepler") &
                self.table["sequence_number"].isna())
        re_expr = r".*Q(\d+)"
        seq_num[mask] = [re.findall(re_expr, item[0])[0] if re.findall(re_expr, item[0]) else "" for item in joint_table.loc[mask,["description"]].values]

    def _filter_kepler(
            products,
            quarter: Union[int, list[int]] = None,
            month: Union[int, list[int]]= None,
            limit: int = None,
        ) -> pd.DataFrame:
        # Filter Kepler product by month/quarter


        mask = np.ones(len(products), dtype=bool)

        if sum(mask) == 0:
            return products

        # Identify quarter by the description.
        # This is necessary because the `sequence_number` field was not populated
        # for Kepler prime data at the time of writing this function.
        if quarter is not None:
            quarter_mask = np.zeros(len(products), dtype=bool)
            for q in np.atleast_1d(quarter).tolist():
                quarter_mask += products['description'].str.endswith(f"Q{q}")
            mask &= quarter_mask

        # For Kepler short cadence data the month can be specified
        if month is not None:
            month = np.atleast_1d(month).tolist()
            # Get the short cadence date lookup table.
            table = pd.read_csv(
                os.path.join(PACKAGEDIR, "data", "short_cadence_month_lookup.csv")
            )
            # The following line is needed for systems where the default integer type
            # is int32 (e.g. Windows/Appveyor), the column will then be interpreted
            # as string which makes the test fail.
            table["StartTime"] = table["StartTime"].astype(str)
            # Grab the dates of each of the short cadence files.
            # Make sure every entry has the correct month
            is_shortcadence = mask & products['description'].str.contains("Short")

            for idx in np.where(is_shortcadence)[0]:
                quarter = np.atleast_1d(int(products["description"][idx].split(" - ")[-1].replace("-", "")[1:])).tolist()
                date = products['dataURI'][idx].split("/")[-1].split("-")[1].split("_")[0]
                # Check if the observation date matches the specified Quarter/month from the lookup table
                if date not in table["StartTime"][table['Month'].isin(month) & table['Quarter'].isin(quarter)].values:
                    mask[idx] = False
        products = products[mask]

        products.sort_values(by=["distance", "productFilename"])

        return products

class K2Search(MASTSearch):

    #@properties like campaigns (seasons?)

    def search_cubedata(hlsp=False):
        # Regular _search_cubedata + processing
        raise NotImplementedError

    def parse_split_campaigns():
        raise NotImplementedError


# Potential HLSP reader class in a different .py file?
