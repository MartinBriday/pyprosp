
import numpy as np
import pandas
import time
import sys
import os
import warnings

from ..utils import tools

_DEFAULT_FILTER_CONFIG = {"sdss":{_f:"sdss_{}0".format(_f) for _f in ["u", "g", "r", "i", "z"]},
                          "galex":{"FUV":"galex_FUV", "NUV":"galex_NUV"},
                          "ps1":{_f:"ps1_{}".format(_f) for _f in ["g", "r", "i", "z", "y"]}}
                          
def keys_to_filters(keys):
    """
    Convert dataframe keys to filters.
    To be detected, "{filter_name}.err" must appears in the given list of keys.
    Return a list of filters with the format "instrument.band" (e.g. sdss.u, ps1.g, ...).
    
    Parameters
    ----------
    keys : [list(string)]
        List of keys from which to extract the filters.
    
    
    Returns
    -------
    list(string)
    """
    return [k for k in keys if k+".err" in keys]
    
def filters_to_pysed(filters):
    """
    Convert filters with the format "instrument.band" (e.g. sdss.u, ps1.g, ...) to 'pysed' compatible filters.
    
    Parameters
    ----------
    filters : [string or list(string)]
        List of filters to convert in 'pysed' compatible format.
        Must be on the format instrument.band (e.g. sdss.u, ps1.g, ...).
    
    
    Returns
    -------
    list(string)
    """
    return [_DEFAULT_FILTER_CONFIG[_filt.split(".")[0]][_filt.split(".")[1]] for _filt in np.atleast_1d(filters)]

def pysed_to_filters(filters):
    """
    Convert 'pysed' compatible filters to filters with the format "instrument.band" (e.g. sdss.u, ps1.g, ...).
    
    Parameters
    ----------
    filters : [string or list(string)]
        List of 'pysed' compatible filters.
    
    
    Returns
    -------
    list(string)
    """
    return [_inst+"."+_band for _inst, _instv in _DEFAULT_FILTER_CONFIG.items()
                            for _band, _bandv in _instv.items() 
                            for _filt in filters if _filt==_bandv ]
