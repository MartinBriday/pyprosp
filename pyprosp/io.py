
import numpy as np
import pandas
import time
import sys
import os
from warnings import warn
from datetime import datetime

from . import tools

_DEFAULT_FILTER_CONFIG = {"sdss":{_f:f"sdss_{_f}0" for _f in ["u", "g", "r", "i", "z"]},
                          "galex":{"FUV":"galex_FUV", "NUV":"galex_NUV"},
                          "ps1":{_f:f"ps1_{_f}" for _f in ["g", "r", "i", "z", "y"]},
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

_DEFAULT_PHYS_PARAMS = {"z":{"prosp_name":"zred",
                             "label":"", 
                             "unit":"",
                             "def":"Redshift"},
                        "mass":{"prosp_name":"mass",
                                "label":"", 
                                "unit":r"M$_\odot$",
                                "def":"Stellar mass"},
                        "log_mass":{"prosp_name":"logmass",
                                    "label":"log(mass)", 
                                    "unit":r"log(M$_\odot$)",
                                    "def":"Log scaled stellar mass"},
                        "total_mass":{"prosp_name":"total_mass",
                                      "label":"", 
                                      "unit":r"M$_\odot$",
                                      "def":"Stellar mass formed"},
                        "sfr":{"prosp_name":"",
                               "label":"SFR", 
                               "unit":"M$_\odot$ / yr",
                               "def":"Star Formation Rate (SFR)"},
                        "log_sfr":{"prosp_name":"",
                                   "label":"log(SFR)", 
                                    "unit":"dex",
                                    "def":"log(SFR)"},
                        "ssfr":{"prosp_name":"",
                                "label":"sSFR", 
                                "unit":"yr$^{-1}$",
                                "def":"Specific Star Formation Rate (sSFR)"},
                        "log_ssfr":{"prosp_name":"",
                                    "label":"log(sSFR)", 
                                    "unit":"dex",
                                    "def":"log(sSFR)"},
                        "log_zsol":{"prosp_name":"logzsol",
                                    "label":"log(Zsol)", 
                                    "unit":r"$\log (Z/Z_\odot)$",
                                    "def":"Metalicity"},
                        "tage":{"prosp_name":"tage",
                                "label":"", 
                                "unit":"Gyr",
                                "def":"Stellar age"},
                        "tau":{"prosp_name":"tau",
                               "label":"", 
                               "unit":"Gyr$^{-1}$",
                               "def":"Exponentially declining Star Formation Histories (SFH) tau exposant"},
                        "dust1":{"prosp_name":"dust1",
                                 "label":"", 
                                 "unit":"",
                                 "def":"Optical depth towards young stars"},
                        "dust2":{"prosp_name":"dust2",
                                 "label":"", 
                                 "unit":"",
                                 "def":"Optical depth at $5500\mathring{A}$"},
                        "dust_ratio":{"prosp_name":"dust_ratio",
                                      "label":"", 
                                      "unit":"",
                                      "def":"Ratio of birth-cloud to diffuse dust"},
                        "duste_gamma":{"prosp_name":"duste_gamma",
                                       "label":"", 
                                       "unit":"",
                                       "def":"Mass fraction of dust in high radiation intensity"},
                        "duste_qpah":{"prosp_name":"duste_qpah",
                                      "label":"", 
                                      "unit":"",
                                      "def":"Percent mass fraction of PAHs in dust"},
                        "duste_umin":{"prosp_name":"duste_umin",
                                      "label":"", 
                                      "unit":"",
                                      "def":"MMP83 local MW intensity"},
                        "gas_logu":{"prosp_name":"gas_logu",
                                    "label":"", 
                                    "unit":"",
                                    "def":r"$\frac{Q_H}{N_H}$"},
                        "agn_tau":{"prosp_name":"agn_tau",
                                   "label":"", 
                                   "unit":"",
                                   "def":"AGN optical depth"},
                        "fagn":{"prosp_name":"fagn",
                                "label":"", 
                                "unit":"",
                                "def":r"$\frac{L_{AGN}}{L_*}$"},
                        "fage_burst":{"prosp_name":"fage_burst",
                                      "label":"", 
                                      "unit":"",
                                      "def":"Time at wich SF burst happens, as a fraction of `tage`"},
                        "fburst":{"prosp_name":"fburst",
                                  "label":"", 
                                  "unit":"",
                                  "def":"Fraction of total mass formed in the SF burst"},
                        "tburst":{"prosp_name":"tburst",
                                  "label":"", 
                                  "unit":"Gyr",
                                  "def":"Time at wich SF burst happens"},
                        }


_DEFAULT_NON_PARAMETRIC_SFH = {"agebins":"agebins", "z_fraction":"z_fraction", "logsfr_ratios":"logsfr_ratios"}

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
    _spec = tools.convert_flux_unit(flux=_spec, unit_in=unit, unit_out="mgy", wavelength=_lbda)
    _out = {"lbda":_lbda, "spec":_spec}
    
    if _dspec is not None:
        _out["spec.err"] = tools.convert_flux_unit(flux=_dspec, unit_in=unit, unit_out="mgy", wavelength=_lbda)
    
    return _out

def get_now(format="%Y%m%d_%H%M%S"):
    """ 
    Return a string filled with the current day and time.
    
    Parameters
    ----------
    format : [string]
        Format with which returning the current time.
        See e.g. https://www.tutorialspoint.com/python/time_strftime.htm for the format.
    
    
    Returns
    -------
    string
    """
    return datetime.now().strftime(format)


    
