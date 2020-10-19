
import numpy as np
from astropy import constants

# --------------------------- #
# - Conversion Tools        - #
# --------------------------- #
def flux_to_mag(flux, dflux=None, wavelength=None, zp=None, inhz=False):
    """
    Convert fluxes into AB or zp magnitudes.

    Parameters
    ----------
    flux : [float or array]
        Flux(es).
    
    dflux : [float or array or None]
        Flux uncertainties if any.
        Default is None.

    wavelength : [float or array]
        // Ignored if inhz=True //
        Central wavelength(s) [in AA] of the photometric filter(s).
        Default is None.

    zp : [float or array]
        // Ignored if inhz=True //
        // Ignored if wavelength is provided //
        Zero point of the flux(es).
        Default is None.

    inhz : [bool]
        Set to true if the flux(es) (and uncertainties) is/are given in erg/s/cm2/Hz
        False means in erg/s/cm2/AA.
        Default is False.
        
    Returns
    -------
    - float or array, None (if magerr is None)
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

def mag_to_flux(mag, magerr=None, wavelength=None, zp=None, inhz=False):
    """
    Convert AB or zp magnitudes into fluxes

    Parameters
    ----------
    mag : [float or array]
        AB magnitude(s)

    magerr : [float or array or None]
        Magnitude uncertainties if any.
        Default is None.

    wavelength : [float or array]
        // Ignored if inhz=True //
        Central wavelength(s) [in AA] of the photometric filter(s).
        Default is None.

    zp : [float or array]
        // Ignored if inhz=True //
        // Ignored if wavelength is provided //
        Zero point of the flux(es).
        Default is None.

    inhz : [bool]
        Set to true if the flux(es) (and uncertainties) is/are returned in erg/s/cm2/Hz
        False means in erg/s/cm2/AA.
        Default is False.

    Returns
    -------
    - float or array, None (if magerr is None)
    - float or array, float or array (if magerr provided)
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
    flux_aa : [float or array]
        Flux(es) in erg/s/cm2/AA.
    
    wavelength : [float or array]
        Central wavelength(s) [in AA] of the photometric filter(s).
    
    
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
    flux_aa : [float or array]
        Flux(es) in erg/s/cm2/Hz.
    
    wavelength : [float or array]
        Central wavelength(s) [in AA] of the photometric filter(s).
    
    
    Returns
    -------
    float or array
    """
    return flux / (wavelength**2 / constants.c.to("AA/s").value)

def convert_flux_unit(flux, unit_in, unit_out, wavelength=None):
    """
    Convert fluxes' unit.
    Available units are:
        - "Hz": erg/s/cm2/Hz
        - "AA": erg/s/cm2/AA
        - "mgy": maggies
        - "Jy": Jansky
    
    Parameters
    ----------
    flux : [float or array]
        Flux(es) to convert.
    
    unit_in : [string]
        Input flux unit.
        Must be either "Hz", "AA", "mgy" or "Jy".
    
    unit_out : [string]
        Desired output flux unit.
        Must be either "Hz", "AA", "mgy" or "Jy".
    
    wavelength : [string]
        Wavelengths corresponding to the given fluxes.
        Not necessary in every conversions.
        Default is None.
    
    
    Returns
    -------
    float or array
    """
    if unit_in == unit_out:
        return flux
    
    _flux = np.atleast_1d(flux).copy()
    _flux = _flux[0] if len(_flux) == 1 else _flux
    if unit_in == "AA":
        _flux = flux_aa_to_hz(flux=_flux, wavelength=wavelength)
    elif unit_in in ["mgy", "Jy"]:
        _flux *= 1e-23 * (3631 if unit_in == "mgy" else 1)
    
    if unit_out == "AA":
        return flux_hz_to_aa(flux=_flux, wavelength=wavelength)
    elif unit_out in ["mgy", "Jy"]:
        _flux *= 1e23
        return _flux / 3631 if unit_out == "mgy" else _flux
    return _flux

def convert_unit(data, unit_in, unit_out, data_unc=None, wavelength=None):
    """
    Convert data between units of flux or magnitude.
    
    Parameters
    ----------
    data : [float or array]
        Data to convert.
    
    data_unc : [float or array or None]
        Data uncertaint.y.ies.
    
    unit_in : [string]
        Input data unit.
        Available units are:
            - "Hz": erg/s/cm2/Hz
            - "AA": erg/s/cm2/AA
            - "mgy": maggies
            - "Jy": Jansky
            - "mag": magnitude
    
    unit_out : [string]
        Desired output data unit.
        Available units are:
            - "Hz": erg/s/cm2/Hz
            - "AA": erg/s/cm2/AA
            - "mgy": maggies
            - "Jy": Jansky
            - "mag": magnitude
    
    wavelength : [string]
        Wavelengths corresponding to the given data.
        Not necessary in every conversions.
        Default is None.
    
    
    Returns
    -------
    - float or array, None (if data_unc is None)
    - float or array, float or array (if data_unc provided)
    """
    if unit_in == unit_out:
        return data if data_unc is None else (data, data_unc)
    
    if unit_in == "mag":
        data, data_unc = mag_to_flux(data, magerr=data_unc, wavelength=wavelength, zp=None, inhz=False)
        unit_in = "AA"
    
    data = convert_flux_unit(data, unit_in, ("AA" if unit_out=="mag" else unit_out), wavelength)
    if data_unc is not None:
        data_unc = convert_flux_unit(data_unc, unit_in, ("AA" if unit_out=="mag" else unit_out), wavelength)
    
    if unit_out == "mag":
        data, data_unc = flux_to_mag(data, dflux=data_unc, wavelength=wavelength, zp=None, inhz=False)
    
    return data if data_unc is None else (data, data_unc)

def get_unit_label(unit):
    """
    Return a string label depending on the given unit.
    
    Parameters
    ----------
    unit : [string]
        Available units are:
            - "Hz": erg/s/cm2/Hz
            - "AA": erg/s/cm2/AA
            - "mgy": maggies
            - "Jy": Jansky
    
    
    Returns
    -------
    string
    """
    _labels = {"Hz":r"$\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}$",
               "AA":r"$\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$",
               "mgy":"mgy",
               "Jy":"Jy",
               "mag":"mag"}
    return _labels[unit]

def fix_trans(lbda, trans):
    """
    Check if the transmission curve start and end with zero.
    Fix it if not.
    Return the tuple (lbda, trans) (fixed if needed).
    
    Parameters
    ----------
    lbda : [array]
        Wavelength of the transmission curve.
        Must be same size as for 'trans'.
    
    trans : [array]
        Transmission data.
        Must be same size as for 'lbda'.
    
    
    Returns
    -------
    np.array, np.array
    """
    _lbda, _trans = np.array(lbda).copy(), np.array(trans).copy()
    _lbda_step = np.mean([_lbda[ii+1] - _l for ii, _l in enumerate(_lbda[:-1])])
    if _trans[0] != 0:
        _lbda = np.insert(_lbda, 0, _lbda[0]-_lbda_step)
        _trans = np.insert(_trans, 0, 0.)
    if _trans[-1] != 0:
        _lbda = np.insert(_lbda, len(_lbda), _lbda[-1]+_lbda_step)
        _trans = np.insert(_trans, len(_trans), 0.)
    return _lbda, _trans

def deredshift(lbda=None, flux=None, z=0, variance=None, exp=3):
    """
    Deredshift spectrum from z to 0, and apply a (1+z)**exp flux-correction.
    
    exp=3 is for erg/s/cm2/A spectra to be later corrected using proper (comoving) distance but *not* luminosity distance.
    
    Return the restframed wavelength, flux and, if any, variance.

    Parameters
    ----------
    lbda, flux : [array]
        Wavelength and flux or the spectrum.
        Flux is expected to be in erg/s/cm2/AA.
        
    z : [float]
        Cosmological redshift

    variance : [array or None]
        Spectral variance if any (square of flux units).
        Default is None.

    exp : [float]
        Exposant for the redshift flux dilution (exp=3 is for erg/s/cm2/AA spectra).
        Default is 3.

    Returns
    -------
    np.array, np.array, np.array if variance is not None
    """
    zp1 = 1 + z
    _lbda, _flux, _variance = None, None, None
    if lbda is not None:
        _lbda = lbda/zp1           # Wavelength correction
    if flux is not None:
        _flux = flux*(zp1**exp)      # Flux correction
    if variance is not None:
        _variance = variance*(zp1**exp)**2
    
    _out = [ii for ii in [_lbda, _flux, _variance] if ii is not None]
    return _out[0] if len(_out) == 1 else tuple(_out)

def synthesize_photometry(lbda, flux, filter_lbda, filter_trans, normed=True):
    """
    Return photometry from the given spectral information through the given filter.

    This function converts the flux into photons since the transmission provides the fraction of photons that goes though.

    Parameters
    -----------
    lbda, flux : [array]
        Wavelength and flux of the spectrum from which you want to synthetize photometry.
        
    filter_lbda, filter_trans : [array]
        Wavelength and transmission of the filter.
    
    Options
    -------
    normed : [bool]
        Shall the fitler transmission be normalized (True) or not (False).
        Default is True.

    Returns
    -------
    float (photometric point)
    """
    # ---------
    # The Tool
    def integrate_photons(lbda, flux, step, flbda, fthroughput):
        filter_interp = np.interp(lbda, flbda, fthroughput)
        dphotons = (filter_interp * flux) * lbda * 5.006909561e7
        return np.trapz(dphotons,lbda) if step is None else np.sum(dphotons*step)
    
    # ---------
    # The Code
    normband = 1. if not normed else integrate_photons(lbda, np.ones(len(lbda)),None,filter_lbda,filter_trans)
      
    return integrate_photons(lbda,flux,None,filter_lbda,filter_trans)/normband

def sfh_delay_tau_to_sfr(tage, tau, mass):
    """
    Conversion from SFH to SFR based on a delay-tau model.
    The returned SFR is in Msun/year.
    
    Parameters
    ----------
    tage : [float]
    
    tau : [float]
    
    mass : [float]
    
    
    Returns
    -------
    float
    """
    from scipy.special import gamma, gammainc
    return mass * (tage/tau**2) * np.exp(-tage/tau) / (gamma(2) * gammainc(2, tage/tau)) * 1e-9

def get_box_title(title, box="\n#=#\n#   {}   #\n#=#\n"):
    """
    Return the title in a custom box as a string.
    
    Parameters
    ----------
    title : [string]
        Title to prinnt in the custom box.
    
    box : [string]
        Custom box format which will be adapted to the given title.
        Must contain "{}" in which the title will be included.
        Default is "\n#=#\n#   {}   #\n#=#\n".
    
    
    Returns
    -------
    string
    """
    _box = [ii.format(title) for ii in box.split("\n")]
    _title = [_b for _b in _box if title in _b][0]
    for ii, _b in enumerate(_box):
        if _b != _title and _b != "" and _b is not None:
            _box_edge = list(_b)
            _ii_mid = int(len(_box_edge) / 2)
            _box_edge[_ii_mid] = "".join([_box_edge[1]]*(len(_title)-(len(_box_edge)-1)))
            _box[ii] = "".join(_box_edge)
    return "\n".join(_box)
