
import numpy as np
from astropy import constants
import pandas
from warnings import warn

from pylephare.lephare import format_input_data as LP_format_input_data

# --------------------------- #
# - Conversion Tools        - #
# --------------------------- #
def flux_to_mag(flux, dflux, wavelength=None, zp=None, inhz=False):
    """
    Convert fluxes (erg/s/cm2/A or erg/s/cm2/Hz) into AB or ZP magnitudes
    
    Parameters
    ----------
    flux, dflux: [floats or arrays]
        Flux(es)
    
    dflux : [floats or arrays or None]
        Flux(es) error(s).
        Must be the same size than 'flux'.
        Dafault is None.
    
    wavelength: [float or array] -optional-
        Central wavelength [in AA] of the photometric filter.
        // Ignored if inhz=True //

    zp: [float or array] -optional-
        Zero point of the flux.
        // Ignored if inhz=True //
        // Ignored if wavelength is provided //

    inhz:
        Set to true if the flux (and flux error(s)) are given in erg/s/cm2/Hz.
        False means in erg/s/cm2/AA.
        Default is False.
        
        
    Returns
    -------
    - float or array (if magerr is None)
    - float or array, float or array (if magerr provided)
    
    """
    if inhz:
        zp = -48.598 # instaad of -48.60 such that hz to aa is correct
        wavelength = 1
    else:
        if zp is None and wavelength is None:
            raise ValueError("'zp' or 'wavelength' must be provided.")
        if zp is None:
            zp = -2.406 
        else:
            wavelength=1
            
    mag_ab = -2.5*np.log10(flux*wavelength**2) + zp
    if dflux is None:
        return mag_ab, None
    
    dmag_ab = -2.5/np.log(10) * dflux / flux
    return mag_ab, dmag_ab

def mag_to_flux(mag, dmag=None, wavelength=None, zp=None, inhz=False):
    """
    Converts AB or ZP magnitudes into fluxes (erg/s/cm2/A or erg/s/cm2/Hz).

    Parameters
    ----------
    mag, dmag: [float or array]
        AB or ZP magnitude(s).
        
    dmag : [float or array or None]
        AB or ZP magnitude(s) error(s).
        Must be the same size than 'mag'.
        Default is None.

    wavelength: [float or array] -optional-
        Central wavelength [in AA] of the photometric filter.
        // Ignored if inhz=True //

    zp: [float or array] -optional-
        Zero point of for flux.
        // Ignored if inhz=True //
        // Ignored if wavelength is provided //

    inhz:
        Set to true if the flux (and flux) are returned in erg/s/cm2/Hz.
        False means in erg/s/cm2/AA.
        Default is None.


    Returns
    -------
    - float or array (if dmag is None)
    - float or array, float or array (if dmag provided)
    """
    if inhz:
        zp = -48.598 # instaad of -48.60 such that hz to aa is correct
        wavelength = 1
    else:
        if zp is None and wavelength is None:
            raise ValueError("zp or wavelength must be provided")
        if zp is None:
            zp = -2.406 
        else:
            wavelength=1

    flux = 10**(-(mag-zp)/2.5) / wavelength**2
    if magerr is None:
        return flux, None
    
    dflux = np.abs(flux*(-magerr/2.5*np.log(10))) # df/f = dcount/count
    return flux, dflux


def flux_aa_to_hz(flux, wavelength):
    """
    Convert fluxes from erg/s/cm2/AA to erg/s/cm2/Hz.
    
    Parameters
    ----------
    flux : [float or array]
        Flux(es) is erg/s/cm2/AA.
    
    wavelength : [float or array]
        Depend on the case:
        - Central wavelength [in AA] of the photometric filter.
        - Wavelength array (e.g. converting a spectrum).
    
    
    Returns
    -------
    float or array
    """
    return flux * (wavelength**2 / constants.c.to("AA/s").value)
    
def flux_hz_to_aa(flux, wavelength):
    """
    Convert fluxes from erg/s/cm2/Hz to erg/s/cm2/AA.
    
    Parameters
    ----------
    flux : [float or array]
        Flux(es) is erg/s/cm2/Hz.
    
    wavelength : [float or array]
        Depend on the case:
        - Central wavelength [in AA] of the photometric filter.
        - Wavelength array (e.g. converting a spectrum).
    
    
    Returns
    -------
    float or array
    """
    return flux / (wavelength**2 / constants.c.to("AA/s").value)

def get_lephare_input_data(data, sn_z, sn_name=None, filters=["sdss.u", "sdss.g", "sdss.r", "sdss.i", "sdss.z"]):
    """
    Return a DataFrame in a LePhare input catalog compatible.
    
    Parameters
    ----------
    data : [dict or pandas.DataFrame]
        Database containing fluxes and errors.
    
    sn_z : [float]
        SN redshift.
    
    filters : [list(string)]
        List of filters to be included in the LePhare input catalog.
        Must be in the format "project.band" (e.g. "sdss.r", "ps1.g", "galex.FUV",...).
    
    Options
    -------
    sn_name : [string]
        SN name.
        Default is None.
    
    
    Results
    -------
    pandas.DataFrame
    """
    data_out = {}
    filters = [filters] if len(np.array([filters]).shape) == 1 else filters
    for _filt in filters:
        try:
            key = [k for k in data.keys() if (_filt in k and "err" not in k)][0]
            key_err = [k for k in data.keys() if (_filt in k and "err" in k)][0]
            data_out[_filt] = [data[key]]
            data_out[_filt+".err"] = [data[key_err]]
        except:
            warn("No data recognized for the filter '{}'")
            data_out[_filt] = [-999]
            data_out[_filt+".err"] = [-999]
    context = [ii for ii, _f in enumerate(filters) if data_out[_f][0] > -1]
    data_out["CONTEXT"] = [np.sum([2**ii for ii in context])]
    data_out["Z-SPEC"] = [sn_z]
    data_out["STRING"] = [str(sn_name)]
    return pandas.DataFrame(data_out)

def is_documented_by(original):
    """
    
    """
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target
    return wrapper

@is_documented_by(LP_format_input_data)
def LePhare_format_input_data(data, sn_z, filters=["sdss.u", "sdss.g", "sdss.r", "sdss.i", "sdss.z"], sn_name=None):
    return LP_format_input_data(data=data, sn_z=sn_z, filters=filters, sn_name=sn_name).iloc[0]
