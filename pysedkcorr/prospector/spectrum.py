
import pandas
import numpy as np

from ..utils import tools

class ProspectorSpectrum():
    """
    This class build a spectrum object based on the prospector fit results.
    """
    
    def __init__(self, **kwargs):
        """
        
        """
        if kwargs != {}:
            self.set_data(**kwargs)
    
    def set_data(self, chains, model, obs, sps):
        """
        
        """
        self._chains = chains
        self._model = model
        self._obs = obs
        self._sps = sps
        self._param_chains = {_p:np.array([jj[ii] for jj in self.chains]) for ii, _p in enumerate(self.theta_labels)}
    
    def load_spectra(self, size=None, savefile=None, **kwargs):
        """
        
        """
        if size is not None and isinstance(size, int):
            self._mask_chains = np.random.choice(self.len_chains, size, replace=False)
        _chains = self.chains[self.mask_chains] if self.has_mask_chains() else self.chains
        self._spectrum_chain = np.array([self.get_theta_spectrum(theta=_theta, unit="mgy") for _theta in _chains])
        self._spectrum_chain = pandas.DataFrame(self._spectrum_chain.T)
        self._spectrum_chain.index = self.wavelengths
        self._spectrum_chain.index.names = ["lbda"]
        if savefile is not None:
            self.save_spec(savefile=savefile, **kwargs)
    
    def save_spec(self, savefile, **kwargs):
        """
        
        """
        self.spectrum_chain.to_csv(savefile, **kwargs)
    
    def get_theta_spectrum(self, theta, unit="mgy"):
        """
        
        """
        _spec, _, _ = self.model.sed(theta, obs=self.obs, sps=self.sps)
        return tools.convert_flux_unit(_spec, "mgy", unit, self.wavelengths)
    
    def get_spectral_data(self, restframe=False, unit="Hz", lbda_lim=(None, None)):
        """
        
        """
        _lbda = self.wavelengths.copy()
        _spec = self.spectrum.copy()
        _spec_low, _spec_up = np.percentile(self.spectrum_chain, [16, 84], axis=1)
        _spec = [_spec, _spec_low, _spec_up]
        _unit_in = "mgy"
        
        # Deredshift
        if restframe:
            for ii, _s in enumerate(_spec):
                _s = tools.convert_flux_unit(_s, _unit_in, "AA", wavelength=_lbda)
                _spec[ii] = tools.deredshift(lbda=None, flux=_s, z=self.z, variance=None, exp=3)
            _lbda = tools.deredshift(lbda=_lbda, flux=None, z=self.z, variance=None, exp=3)
            _unit_in = "AA"
        
        # Change unit
        for ii, _s in enumerate(_spec):
            _spec[ii] = tools.convert_flux_unit(_s, _unit_in, unit_out=("AA" if unit=="mag" else unit), wavelength=_lbda)
            if unit == "mag":
                _spec[ii] = np.array(tools.flux_to_mag(_spec[ii], dflux=None, wavelength=_lbda, zp=None, inhz=False))[0]
        
        # Build the mask given by the wavelength desired limits
        _mask_lbda = np.ones(len(_lbda), dtype = bool)
        if isinstance(lbda_lim, tuple) and len(lbda_lim) == 2:
            _lim_low, _lim_up = lbda_lim
            if _lim_low is not None:
                _mask_lbda &= _lbda > _lim_low
            if _lim_up is not None:
                _mask_lbda &= _lbda < _lim_up
                
        return _lbda[_mask_lbda], _spec[0][_mask_lbda], _spec[1][_mask_lbda], _spec[2][_mask_lbda]
    
    def show(self, ax=None, figsize=[7,3.5], ax_rect=[0.1,0.2,0.8,0.7], unit="Hz", restframe=False,
             lbda_lim=(None, None), spec_prop={}, spec_unc_prop={}, phot_prop={}, savefile=None):
        """
        Plot the spectrum.
        
        Options
        -------
        ax : [plt.Axes or None]
            If an axes is given, draw the spectrum on it.
            Default is None.
        
        figsize : [list(float)]
            Two values list to fix the figure size (in inches).
            Default is [7,3.5].
        
        ax_rect : [list(float)]
            The dimensions [left, bottom, width, height] of the new axes.
            All quantities are in fractions of figure width and height.
            Default is [0.1,0.2,0.8,0.7].
        
        unit : [string]
            Flux unit to plot. Available units are:
                - "Hz": erg/s/cm2/Hz
                - "AA": erg/s/cm2/AA
                - "mgy": maggies
                - "Jy": Jansky
                - "mag": magnitude
        
        restframe : [bool]
            If True, the spectrum is first deredshifted before doing the synthetic photometry.
            Default is False.
        
        spec_prop : [dict]
            Spectrum pyplot.plot kwargs.
            Default is {}.
        
        spec_unc_prop : [dict]
            Spectrum uncertainties pyplot.fill_between kwargs.
            If {}, the uncertainties are not plotted.
            Default is {}.
        
        phot_prop : [dict]
            Photometry pyplot.scatter kwargs.
            Default is {}.
        
        savefile : [string or None]
            Give a directory to save the figure.
            Default is None (not saved).
        
        
        Returns
        -------
        
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes(ax_rect)
        else:
            fig = ax.figure
            
        _lbda, _spec, _spec_low, _spec_up = self.get_spectral_data(restframe=restframe, unit=unit, lbda_lim=lbda_lim)
        ax.plot(_lbda, _spec, **spec_prop)
        if spec_unc_prop:
            ax.fill_between(_lbda, _spec_low, _spec_up, **spec_unc_prop)

        ax.set_xlabel(r"wavelentgh [$\AA$]", fontsize="large")
        ax.set_ylabel(tools.get_unit_label(unit), fontsize="large")
        
        if savefile is not None:
            fig.savefig(savefile)
            
        return {"fig":fig, "ax":ax}
    
        
        
    
    
    #----------------#
    #   Properties   #
    #----------------#
    @property
    def chains(self):
        """ List of the fitted parameter chains """
        return self._chains
    
    @property
    def len_chains(self):
        """ Length of the chains --> number of steps for the MCMC """
        return len(self.chains)
    
    @property
    def mask_chains(self):
        """ List of random index of chains """
        if not hasattr(self, "_mask_chains"):
            self._mask_chains = None
        return self._mask_chains
    
    def has_mask_chains(self):
        """ Test that a mask of the chains exists """
        return self.mask_chains is not None
    
    @property
    def param_chains(self):
        """ Dictionary containing the fitted parameter chains """
        return self._param_chains
    
    @property
    def model(self):
        """ Prospector SED Model """
        return self._model
    
    @property
    def obs(self):
        """ Prospector 'obs' dictionary """
        return self._obs
    
    @property
    def sps(self):
        """ Prospector 'sps' object """
        return self._sps
    
    @property
    def theta_labels(self):
        """ List of the fitted parameters """
        return self.model.theta_labels()
    
    @property
    def z(self):
        """ Redshift """
        return self.model.params.get("zred", 0.)[0]
    
    @property
    def wavelengths(self):
        """ Wavelength array corresponding with the spectra """
        if self.obs["wavelength"] is None:
            return self.sps.wavelengths * (1.0 + self.z)
        else:
            return self.obs["wavelength"]
    
    @property
    def spectrum_chain(self):
        """ Array containing spectrum chain """
        if not hasattr(self, "_spectrum_chain"):
            self.load_spectra()
        return self._spectrum_chain
    
    @property
    def spectrum(self):
        """ Median of the spectrum chain """
        if not hasattr(self, "_spectrum_chain"):
            self.load_spectra()
        return np.median(self.spectrum_chain, axis=1)
