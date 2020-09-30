
import pandas
import numpy as np

from ..utils import tools
from . import io

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
    
    @classmethod
    def from_h5(cls, filename, warnings=True):
        """
        
        """
        from .prospector import Prospector
        import h5py
        _prosp = Prospector.from_h5(filename=filename, warnings=warnings)
        _this = cls(chains=_prosp.chains, model=_prosp.model, obs=_prosp.obs, sps=_prosp.sps)
        with h5py.File(filename, "r") as _h5f:
            if "spec_chain" in _h5f:
                import pickle
                _h5_group = _h5f["spec_chain"].attrs
                _this._spec_chain = np.array([pickle.loads(_h5_group[str(ii)]) for ii in np.arange(len(_h5_group)-1)])
                _this._spec_chain = pandas.DataFrame(_this._spec_chain.T)
                _this._spec_chain.index = pickle.loads(_h5_group["lbda"])
                _this._spec_chain.index.names = ["lbda"]
        return _this
    
    #------------------#
    #   Spectrometry   #
    #------------------#
    def load_spectra(self, size=None, savefile=None, **kwargs):
        """
        
        """
        if size is not None and isinstance(size, int):
            self._mask_chains = np.random.choice(self.len_chains, size, replace=False)
        else:
            self._mask_chains = None
        _chains = self.chains[self.mask_chains] if self.has_mask_chains() else self.chains
        self._spec_chain = np.array([self.get_theta_spec(theta=_theta, unit="mgy") for _theta in _chains])
        self._spec_chain = pandas.DataFrame(self._spec_chain.T)
        self._spec_chain.index = self.wavelengths
        self._spec_chain.index.names = ["lbda"]
        if savefile is not None:
            self.write_spec(savefile=savefile, **kwargs)
    
    def write_spec(self, savefile, **kwargs):
        """
        
        """
        if savefile.endswith(".h5"):
            import h5py
            import pickle
            with h5py.File(savefile, "a") as _h5f:
                if "spec_chain" in _h5f:
                    del _h5f["spec_chain"]
                _h5_group = _h5f.create_group("spec_chain")
                _h5_group.attrs["lbda"] = np.void(pickle.dumps(self.wavelengths))
                for ii, _chain in enumerate(np.array(self.spec_chain).T):
                    _h5_group.attrs[str(ii)] = np.void(pickle.dumps(_chain))
                _h5f.flush()
        else:
            self.spec_chain.to_csv(savefile, **kwargs)
    
    def get_theta_spec(self, theta, unit="mgy"):
        """
        
        """
        _spec, _, _ = self.model.sed(theta, obs=self.obs, sps=self.sps)
        return tools.convert_flux_unit(_spec, "mgy", unit, self.wavelengths)
    
    def get_spectral_data(self, restframe=False, unit="Hz", lbda_lim=(None, None)):
        """
        
        """
        _lbda = self.wavelengths.copy()
        _spec_chain = np.array(self.spec_chain.T)
        _unit_in = "mgy"
        
        # Deredshift
        if restframe:
            _spec_chain = tools.convert_flux_unit(_spec_chain, _unit_in, "AA", wavelength=_lbda)
            _lbda, _spec_chain = tools.deredshift(lbda=_lbda, flux=_spec_chain, z=self.z, variance=None, exp=3)
            _unit_in = "AA"
        
        # Change unit
        _spec_chain = tools.convert_flux_unit(_spec_chain, _unit_in, unit_out=("AA" if unit=="mag" else unit), wavelength=_lbda)
        if unit == "mag":
            _spec_chain = tools.flux_to_mag(_spec_chain, dflux=None, wavelength=_lbda, zp=None, inhz=False)
        
        # Build the mask given by the wavelength desired limits
        _mask_lbda = np.ones(len(_lbda), dtype = bool)
        if isinstance(lbda_lim, tuple) and len(lbda_lim) == 2:
            _lim_low, _lim_up = lbda_lim
            if _lim_low is not None:
                _mask_lbda &= _lbda > _lim_low
            if _lim_up is not None:
                _mask_lbda &= _lbda < _lim_up
        
        _spec_low, _spec, _spec_up = np.percentile(_spec_chain.T, [16, 50, 84], axis=1)
        
        return {"lbda":_lbda[_mask_lbda], "spec_chain":_spec_chain, "spec":_spec[_mask_lbda],
                "spec_low":_spec_low[_mask_lbda], "spec_up":_spec_up[_mask_lbda]}
    
    #----------------#
    #   Photometry   #
    #----------------#
    def get_synthetic_photometry(self, filter, restframe=False, unit="Hz"):
        """
        Return photometry synthesized through the given filter/bandpass.
        The returned data are (effective wavelength, synthesize flux/mag) in an array with same size as for 'self.spec_chain'.

        Parameters
        ----------
        filter : [string, sncosmo.BandPass, 2D array]
            The filter through which the spectrum will be synthesized.
            Accepted input format:
            - string: name of a known filter (instrument.band), the actual bandpass will be grabbed using using data from 'self.obs["filters"]'
            - 2D array: containing wavelength and transmission --> sncosmo.bandpass.BandPass(*filter)
            - sncosmo.bandpass.BandPass
        
        Options
        -------
        restframe : [bool]
            If True, the spectrum is first deredshifted before doing the synthetic photometry.
            Default is False.

        unit : [string]
            Unit of the returned photometry. Available units are:
                - "Hz": erg/s/cm2/Hz
                - "AA": erg/s/cm2/AA
                - "mgy": maggies
                - "Jy": Jansky
                - "mag": magnitude
            Default is "Hz".
        
        
        Returns
        -------
        np.array, np.array
        """
        from sncosmo import bandpasses
        
        # - Get the corresponding bandpass
        try:
            filter = io.filters_to_pysed(filter)[0]
        except:
            pass
        if filter in self.obs["filternames"]:
            _filter = self.obs["filters"][self.obs["filternames"].index(filter)]
            _lbda, _trans = tools.fix_trans(_filter.wavelength, _filter.transmission)
            _bp = bandpasses.Bandpass(_lbda, _trans)
        elif type(filter) == bandpasses.Bandpass:
            _bp = filter
        elif len(filter) == 2:
            _lbda, _trans = tools.fix_trans(*filter)
            _bp = bandpasses.Bandpass(_lbda, _trans)
        else:
            raise TypeError("'filter' must either be a filter name like filter='sdss.u', "+
                            "a sncosmo.BandPass or a 2D array filter=[wave, transmission].\n"+
                            f"Your input : {filter}")

        # - Synthesize through bandpass
        _sflux_aa = self.synthesize_photometry(_bp.wave, _bp.trans, restframe=restframe)
        _slbda = _bp.wave_eff/(1+self.z) if restframe else _bp.wave_eff

        if unit == "mag":
            return _slbda, tools.flux_to_mag(_sflux_aa, None, wavelength=_slbda)
        
        return _slbda, tools.convert_flux_unit(_sflux_aa, "AA", unit, wavelength=_slbda)
    
    def synthesize_photometry(self, filter_lbda, filter_trans, restframe=False):
        """
        Return the synthetic flux in erg/s/cm2/AA.
        
        Parameters
        ----------
        filter_lbda, filter_trans : [array, array]
            Wavelength and transmission of the filter.
        
        Options
        -------
        restframe : [bool]
            If True, the spectrum is first deredshifted before doing the synthetic photometry.
            Default is False.
        
        
        Returns
        -------
        float
        """
        _spec_data = self.get_spectral_data(restframe=restframe, unit="AA", lbda_lim=(None, None))
        return tools.synthesize_photometry(_spec_data["lbda"], _spec_data["spec_chain"], filter_lbda, filter_trans, normed=True)
    
    def get_phot_data(self, filters=None, restframe=False, unit="mag"):
        """
        
        """
        if filters is None:
            filters = np.array(self.obs["filternames"])
        elif filters == "mask":
            filters = np.array(self.obs["filternames"])[self.obs["phot_mask"]]
        else:
            filters = np.atleast_1d(filters)
        _phot_chains = {_f:self.get_synthetic_photometry(filter=_f, restframe=restframe, unit=unit) for _f in filters}
        _lbda = np.array([_phot_chains[_f][0] for _f in filters])
        _phot_low, _phot, _phot_up = np.array([np.percentile(_phot_chains[_f][1], [16, 50, 84]) for _f in filters]).T
        return {"lbda":_lbda, "phot_chains":_phot_chains, "phot":_phot, "phot_low":_phot_low, "phot_up":_phot_up}
    
    def get_phot_obs(self, filters="mask", restframe=False, unit="mag"):
        """
        
        """
        if filters is None:
            filters = np.array(self.obs["filternames"])
        elif filters == "mask":
            filters = np.array(self.obs["filternames"])[self.obs["phot_mask"]]
        else:
            filters = np.atleast_1d(filters)
            try:
                filters = np.array([(_f if _f in self.obs["filternames"] else io.filters_to_pysed(_f)[0]) for _f in filters])
            except KeyError:
                raise ValueError(f"One or more of the given filters are not available \n(filters = {filters}).\n"+
                                 "They must be like 'instrument.band' (eg: 'sdss.u') or prospector compatible filter names.")
        _filt_flag = [np.where(np.array(self.obs["filternames"]) == _f)[0][0] for _f in filters]
        _filters = np.array(self.obs["filters"])[_filt_flag]
        _lbda = np.array([_f.wave_average for _f in _filters])
        _phot = np.array(self.obs["maggies"])[_filt_flag]
        _phot_unc = np.array(self.obs["maggies_unc"])[_filt_flag]
        _unit_in = "mgy"
        
        if restframe:
            _phot, _phot_unc = tools.convert_flux_unit([_phot, _phot_unc], _unit_in, "AA", wavelength=_lbda)
            _lbda, _phot, _phot_unc = tools.deredshift(lbda=_lbda, flux=_phot, z=self.z, variance=_phot_unc**2, exp=3)
            _unit_in = "AA"
        
        _phot, _phot_unc = tools.convert_flux_unit([_phot, _phot_unc], _unit_in, unit_out=("AA" if unit=="mag" else unit), wavelength=_lbda)
        if unit == "mag":
            _phot, _phot_unc = tools.flux_to_mag(_phot, dflux=_phot_unc, wavelength=_lbda, zp=None, inhz=False)
        
        return {"lbda":_lbda, "phot":_phot, "phot_unc":_phot_unc}
    
    #--------------#
    #   Plotting   #
    #--------------#
    def show(self, ax=None, figsize=[7,3.5], ax_rect=[0.1,0.2,0.8,0.7], unit="Hz", restframe=False,
             lbda_lim=(None, None), spec_prop={}, spec_unc_prop={},
             filters=None, phot_prop={}, phot_unc_prop={}, show_obs={},
             show_legend={}, savefile=None):
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
            Which unit for the data to be plotted. Available units are:
                - "Hz": erg/s/cm2/Hz
                - "AA": erg/s/cm2/AA
                - "mgy": maggies
                - "Jy": Jansky
                - "mag": magnitude
        
        restframe : [bool]
            If True, the spectrum is first deredshifted before doing the synthetic photometry.
            Default is False.
        
        spec_prop : [dict or None]
            Spectrum pyplot.plot kwargs.
            If bool(spec_prop) == False, the spectrum is not plotted.
            Default is {}.
        
        spec_unc_prop : [dict or None]
            Spectrum uncertainties pyplot.fill_between kwargs.
            If bool(spec_unc_prop) == False, the uncertainties are not plotted.
            Default is {}.
        
        phot_prop : [dict or None]
            Photometry pyplot.scatter kwargs.
            If bool(phot_prop) == False, the photometric points are not plotted.
            Default is {}.
        
        phot_unc_prop : [dict or None]
            Photometry uncertainties pyplot.errorbar kwargs.
            If bool(phot_unc_prop) == False, the photometric uncertainties are not plotted.
            Default is {}.
        
        show_obs : [dict or None]
            Observed photometry uncertainties pyplot.errorbar kwargs.
            If bool(show_obs) == False, the observed photometric points are not plotted.
            Default is {}.
        
        show_legend : [dict or None]
            If True, plot the legend.
            Default is {}.
        
        savefile : [string or None]
            Give a directory to save the figure.
            Default is None (not saved).
        
        
        Returns
        -------
        
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes(ax_rect)
        else:
            fig = ax.figure
        
        _handles = []
        _labels = []
        # SED
        if spec_prop:
            _spec_data = self.get_spectral_data(restframe=restframe, unit=unit, lbda_lim=lbda_lim)
            ax.plot(_spec_data["lbda"], _spec_data["spec"], **spec_prop)
            _handles.append(mlines.Line2D([], [], **spec_prop))
            _labels.append("Fitted spectrum")
        if spec_unc_prop:
            if "_spec_data" not in locals():
                _spec_data = self.get_spectral_data(restframe=restframe, unit=unit, lbda_lim=lbda_lim)
            _handles.append(ax.fill_between(_spec_data["lbda"], _spec_data["spec_low"], _spec_data["spec_up"], **spec_unc_prop))
            _labels.append(r"1-$\sigma$ fitted spectrum")
        
        # Photometry
        if phot_prop:
            _phot_data = self.get_phot_data(filters=filters, restframe=restframe, unit=unit)
            _handles.append(ax.scatter(_phot_data["lbda"], _phot_data["phot"], **phot_prop))
            _labels.append("Fitted photometry")
        if phot_unc_prop:
            if "_phot_data" not in locals():
                _phot_data = self.get_phot_data(filters=filters, restframe=restframe, unit=unit)
            _handles.append(ax.errorbar(_phot_data["lbda"], _phot_data["phot"],
                                        yerr=[_phot_data["phot"]-_phot_data["phot_low"],
                                              _phot_data["phot_up"]-_phot_data["phot"]],
                                        **phot_unc_prop))
            _labels.append(r"1-$\sigma$ fitted photometry")
        
        if show_obs:
            _phot_obs = self.get_phot_obs(filters=filters, restframe=restframe, unit=unit)
            _handles.append(ax.errorbar(_phot_obs["lbda"], _phot_obs["phot"], yerr=_phot_obs["phot_unc"], **show_obs))
            _labels.append("Observed photometry")

        ax.set_xlabel(r"wavelentgh [$\AA$]", fontsize="large")
        ax.set_ylabel(f"{'magnitude' if unit=='mag' else 'flux'} [{tools.get_unit_label(unit)}]", fontsize="large")
        
        if show_legend:
            ax.legend(_handles, _labels, **show_legend)
        
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
    def spec_chain(self):
        """ Array containing spectrum chain (unit is maggies: "mgy") """
        if not hasattr(self, "_spec_chain"):
            self.load_spectra()
        return self._spec_chain
