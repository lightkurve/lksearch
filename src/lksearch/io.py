"""Tools for reading in fits files into dictionaries."""
from typing import List
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import numpy as np
import warnings

from . import log

__all__ = ['get_data_from_fits_file', 'get_data_from_hdulist']

# Add units that users define that have equivalent units
u.add_enabled_units(
    [
        u.def_unit(["ppt", "parts per thousand"], u.Unit(1e-3)),
        u.def_unit(["ppm", "parts per million"], u.Unit(1e-6)),
    ]
)

# Correct units that have no real equivalencies.
unit_corrections = {
    "BJD - 2454833": u.day,
    "BJD - 2457000, days": u.day,
    "sigma": None,
    "rel": None,
    "pixels":u.pixel,
    "days":u.day,
    "e-/s": u.electron / u.second
}

default_header_keywords = [
    "SIMPLE",
    "EXTEND",
    "NEXTEND",
    "XTENSION",
    "EXTNAME",
    "EXTVER",
    "BITPIX",
    "NAXIS",
    "PCOUNT",
    "GCOUNT",
    "TFIELD",
    "TTYPE",
    "TFORM",
    "TUNIT",
    "TDISP",
    "TDIM",
    "INHERIT",
    "TNULL",
    "WCAX",
    "WCSN",
    "1CTY",
    "2CTY",
    "1CUN",
    "2CUN",
    "1CRV",
    "2CRV",
    "1CDL",
    "2CDL",
    "1CRP",
    "2CRP",
    "1CRPX",
    "2CRPX",
    "1CUNI",
    "2CUNI",
    "1CDLT",
    "2CDLT",
    "11PC",
    "12PC",
    "21PC",
    "22PC",
    "CRPIX",
    "CDELT",
    "CRVAL",
    "CTYPE",
    "CUNIT",
    "PC1",
    "PC2",
    "CHECKSUM",
    "DATASUM",
    "AP_",
    "BP_",
    "A_",
    "B_",
    "CD1",
    "CD2",
    "WCSAXES",
    "COMMENT",
]

numeric_types = (
    float,
    int,
    np.float_,
    np.int_,
    np.float64,
    np.float32,
    np.float16,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.ndarray,
)

# Which extension will we trust in the case of duplicate columns and cards?
trust_ext = 1


def clean_unit(unit: str):
    """Clean up unit if it is not recognized by astropy."""
    if unit in unit_corrections.keys():
        unit = unit_corrections[unit]
    return unit


def get_header_dict(hdr: fits.header.Header):
    """Turn the header into a dictionary, removing common keywords."""
    return {
        card[0]: card[1]
        for card in hdr.cards
        if not np.any([k in card[0] for k in default_header_keywords])
    }


def get_wcs(hdr: fits.header.Header):
    """Get the WCS for a header, if it exists."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FITSFixedWarning)
        wcs = WCS(hdr)
    if sum(wcs.wcs.crpix + wcs.wcs.crval) == 0:
        # Empty WCS
        return None
    else:
        return wcs


def clean_card_list(cards: List):
    """Takes a list of dictionaries which contain cards from headers. Removes the duplicates, preferring the `trust_ext` extension, but raising an error if there are different values for the same key."""
    unq_keys, num_keys = np.unique(
        np.hstack([[*dictionary.keys()] for dictionary in cards]), return_counts=True
    )
    final_dict = {}
    for key, num in zip(unq_keys, num_keys):
        vals = [dictionary[key] for dictionary in cards if key in dictionary.keys()]
        vals = np.hstack(
            [val if not isinstance(val, fits.card.Undefined) else "" for val in vals]
        )
        if num == 1:
            final_dict[key] = vals[0]
        elif len(np.unique(vals, return_counts=True)[1]) == 1:
            final_dict[key] = vals[0]
        else:
            final_dict[key] = vals[trust_ext]
            log.warning(
                f"Card `{key}` appears multiple times with different values, using extention {trust_ext}."
            )
    return final_dict

def get_trusted_column_names(hdulist:fits.hdu.HDUList):
    """Returns a list of the columns that are duplicated in the fits file, that we should take from the trusted extension only. """
    unq_keys, num_keys = np.unique(
        np.hstack(
            [
                [*hdu.columns.names]
                for hdu in hdulist
                if isinstance(hdu, (fits.TableHDU, fits.BinTableHDU))
            ]
        ),
        return_counts=True,
    )
    trust_ext_keys = []
    for key, num in zip(unq_keys, num_keys):
        if num > 1:
            if key in hdulist[trust_ext].columns.names:
                trust_ext_keys.append(key)
    return trust_ext_keys


def get_data_from_hdulist(hdulist:fits.hdu.HDUList, units:bool=True):
    """Returns a dictionary of 1D header cards, and ND data from fits file."""
    cols = {}
    cards = []
    wcss = {}

    trust_ext_keys = get_trusted_column_names(hdulist)
    for idx, hdu in enumerate(hdulist):
        if idx > 0:
            if isinstance(hdu, (fits.hdu.TableHDU, fits.hdu.BinTableHDU)):
                for col in hdu.columns:
                    data = hdu.data[col.name]
                    if len(data) == 0:
                        continue
                    if (col.name in trust_ext_keys) & (idx != trust_ext):
                        log.warning(
                            f"Column `{col.name}` is in extension {trust_ext} and {idx}. Using only data from extension {trust_ext}."
                        )
                        continue
                    if isinstance(data, (list, np.ndarray)):
                        if isinstance(data[0], numeric_types):
                            if units:
                                cols[col.name.lower()] = u.Quantity(data, clean_unit(col.unit))
                            else:
                                cols[col.name.lower()] = data
                        else:
                            cols[col.name.lower()] = np.array([*data])
            elif isinstance(hdu, fits.hdu.ImageHDU):
                cols[hdu.name.lower()] = hdu.data
        wcs = get_wcs(hdu.header)
        if wcs is not None:
            wcss[f"WCS_EXT{idx + 1}"] = wcs
        cards.append(get_header_dict(hdu.header))
    cards = clean_card_list(cards)
    return cards, cols, wcss

def get_data_from_fits_file(path:str, units:bool=True):
    """Returns a dictionary of 1D header cards, and ND data from fits file."""
    with fits.open(path, lazy_load_hdus=False) as hdulist:
        cards, cols, wcss = get_data_from_hdulist(hdulist, units=units)
    return cards, cols, wcss