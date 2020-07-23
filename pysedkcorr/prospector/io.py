
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

def load_phot(phot, unit):
    """
    Return the photometry in a prospector compatible unit and format.
    
    Parameters
    ----------
    phot : [dict or pandas.DataFrame]
        Photometry to convert and load in prospector.
    
    unit : [string]
        Photometry's unit.
        Can be:
            - "Hz": erg/s/cm2/Hz
            - "AA": erg/s/cm2/AA
            - "mgy": maggies
            - "Jy": Jansky
            - "mag": magnitudes
    
    
    Returns
    -------
    dict
    """
    if phot is None:
        return phot
    
    from sedpy.observate import load_filters
    _filters = keys_to_filters(_phot.keys())
    _phot = [float(_phot[_filt]) for _filt in _filters]
    _dphot = [float(_phot[_filt+".err"]) for _filt in _filters]
    _lbda = [_filt.wave_average for _filt in load_filters(_filters)]
    if unit == "mag":
        _phot, _dphot = tools.mag_to_flux(_phot, _dphot, _lbda, inhz=True)
        unit = "Hz"
    _phot, _dphot = tools.convert_flux_unit([_phot, _dphot], unit, "mgy")
    _out = {_filt:_phot[ii] for ii, _filt in enumerate(_filters)}
    _out.update({_filt+".err":_dphot[ii] for ii, _filt in enumerate(_filters)})
    return _out

def load_spec(spec, unit):
    """
    Return the spectrometry in a prospector compatible unit and format.
    
    Parameters
    ----------
    spec : [dict or pandas.DataFrame]
        Spectrometry to convert and load in prospector.
    
    unit : [string]
        Photometry's unit.
        Can be:
            - "Hz": erg/s/cm2/Hz
            - "AA": erg/s/cm2/AA
            - "mgy": maggies
            - "Jy": Jansky
            - "mag": magnitudes
    
    
    Returns
    -------
    dict
    """
    if spec is None:
        return spec
    
    _lbda = np.array(spec["lbda"])
    _spec = np.array(spec["flux"])
    try:
        _dspec = np.array(spec["flux.err"])
    except:
        _dspec = None
    
    if unit == "mag":
        _spec, _dspec = tools.mag_to_flux(_spec, _dspec, _lbda, inhz=True)
        unit = "Hz"
    _spec = tools.convert_flux_unit(_spec, unit, "mgy")
    _out = {"lbda":_lbda, "flux":_spec}
    
    if _dspec is not None:
        _out.update("flux.err":tools.convert_flux_unit(_dspec, unit, "mgy"))
    
    return _out
    
