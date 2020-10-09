
import numpy as np
import pandas
import time
import sys
import os
import warnings

from . import tools

_DEFAULT_FILTER_CONFIG = {"sdss":{_f:f"sdss_{_f}0" for _f in ["u", "g", "r", "i", "z"]},
                          "galex":{"FUV":"galex_FUV", "NUV":"galex_NUV"},
                          "ps1":{_f:f"ps1_{_f}" for _f in ["g", "r", "i", "z", "y"]},
                          "spitzer":{_f:f"spitzer_irac_ch{_f}" for _f in ["1", "2", "3", "4"]},
                          "spitzer":{_f0:f"spitzer_{_f1}" for _f0, _f1 in zip(["irac_1", "irac_2", "irac_3", "irac_4",
                                                                               "irs_blue", "mips_24", "mips_70", "mips_160"],
                                                                              ["irac_ch1", "irac_ch2", "irac_ch3", "irac_ch4",
                                                                               "irs_16", "mips_24", "mips_70", "mips_160"])},
                          "bessell":{_f:f"bessell_{_f}" for _f in ["B", "I", "R", "U", "V"]},
                          "decam":{_f:f"decam_{_f}" for _f in ["u", "g", "r", "i", "z", "Y"]},
                          "gaia":{"gbp":"gaia_bp", "g":"gaia_g", "grp":"gaia_rp"},
                          "herschel":{_f0:f"herschel_{_f1}" for _f0, _f1 in zip(["pacs_blue", "pacs_green", "pacs_red",
                                                                                 "spire_psw", "spire_pmw", "spire_plw",
                                                                                 "spire_psw_ext", "spire_pmw_ext", "spire_plw_ext"],
                                                                                ["pacs_70", "pacs_100", "pacs_160",
                                                                                 "spire_250", "spire_350", "spire_500",
                                                                                 "spire_ext_250", "spire_ext_350", "spire_ext_500"])},
                          "hipparcos":{_f:f"hipparcos_{_f}" for _f in ["B", "H", "V"]},
                          "hsc":{_f:f"hsc_{_f}" for _f in ["g", "r", "i", "z", "y"]},
                          "sofia":{f"hawc_{_f}":f"sofia_hawc_band{_f}" for _f in ["A", "B", "C", "D", "E"]},
                          "stromgren":{_f:f"stromgren_{_f}" for _f in ["b", "u", "v", "y"]},
                          "2mass":{_f:f"twomass_{_f}" for _f in ["H", "J", "Ks"]},
                          "wise":{_f:f"wise_{_f}" for _f in ["w1", "w2", "w3", "w4"]},
                          }
                          
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
    return np.array([k for k in keys if k+".err" in keys])
    
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
    return np.array([_DEFAULT_FILTER_CONFIG[_filt.split(".")[0].lower()][_filt.split(".")[1]] for _filt in np.atleast_1d(filters)])

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
    return np.array([_inst+"."+_band for _filt in filters
                                     for _inst, _instv in _DEFAULT_FILTER_CONFIG.items()
                                     for _band, _bandv in _instv.items()
                                     if _filt==_bandv ])

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
    _data = phot.copy()
    _filters = keys_to_filters(_data.keys())
    _phot = np.array([float(_data[_filt]) for _filt in _filters])
    _dphot = np.array([float(_data[_filt+".err"]) for _filt in _filters])
    _lbda = np.array([_filt.wave_effective for _filt in load_filters(filters_to_pysed(_filters))])
    if unit == "mag":
        _phot, _dphot = tools.mag_to_flux(_phot, _dphot, _lbda, inhz=True)
        unit = "Hz"
    _phot, _dphot = tools.convert_flux_unit([_phot, _dphot], unit, "mgy", _lbda)
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
    
    _data = spec.copy()
    _lbda = np.array(_data["lbda"])
    _spec = np.array(_data["spec"])
    try:
        _dspec = np.array(_data["spec.err"])
    except:
        _dspec = None
    
    if unit == "mag":
        _spec, _dspec = tools.mag_to_flux(_spec, _dspec, _lbda, inhz=True)
        unit = "Hz"
    _spec = tools.convert_flux_unit(_spec, unit, "mgy")
    _out = {"lbda":_lbda, "spec":_spec}
    
    if _dspec is not None:
        _out["spec.err"] = tools.convert_flux_unit(_dspec, unit, "mgy")
    
    return _out
    
