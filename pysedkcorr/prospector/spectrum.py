
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
        Initialization.
        Can run 'set_data'.
        
        Options
        -------
        chains : [np.array]
            Fitted parameter chains.
            Shape must be (nb of fitting step, nb of fitted parameters).
        
        model : [prospect.models.sedmodel.SedModel]
            Prospector model object.
        
        obs : [dict]
            Prospector compatible 'obs' dictionary.
        
        sps : [SPS object]
            Prospector SPS object.
        
        
        Returns
        -------
        Void
        """
        if kwargs != {}:
            self.set_data(**kwargs)
    
    def set_data(self, chains, model, obs, sps):
        """
        Set up input data as attributes.
        
        Parameters
        ----------
        chains : [np.array]
            Fitted parameter chains.
            Shape must be (nb of fitting step, nb of fitted parameters).
        
        model : [prospect.models.sedmodel.SedModel]
            Prospector model object.
        
        obs : [dict]
            Prospector compatible 'obs' dictionary.
        
        sps : [SPS object]
            Prospector SPS object.
        
        
        Returns
        -------
        Void
        """
        self._chains = chains
        self._model = model
        self._obs = obs
        self._sps = sps
        self._param_chains = {_p:np.array([jj[ii] for jj in self.chains]) for ii, _p in enumerate(self.theta_labels)}
    
    @classmethod
    def from_h5(cls, filename, warnings=True):
        """
        Build a ProspectorSpectrum object directly from a .h5 file containing the results of a prospector fit.
        If the given file contains the spectrum chain, it will be automatically loaded.
        
        Parameters
        ----------
        filename : [string]
            File name/directory containing prospector fit results.
            Must be a .h5 file.
        
        Options
        -------
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        
        Returns
        -------
        ProspectorSpectrum
        """
        from .prospector import Prospector
        import h5py
        _prosp = Prospector.from_h5(filename=filename, warnings=warnings)
        _this = cls(chains=_prosp.chains, model=_prosp.model, obs=_prosp.obs, sps=_prosp.sps)
        with h5py.File(filename, "r") as _h5f:
            _this._spec_chain = _this.read_file(filename=filename, warnings=warnings)
        return _this
    
    @staticmethod
    def read_file(filename, warnings=True, **kwargs):
        """
        Extract and return the spectrum chain dataframe from the given file.
        
        Parameters
        ----------
        filename : [string]
            File name/directory from which to extract the dataframe.
            If a .h5 file is given, data must be saved as attributes in the group named "spec_chain".
            Else, return pandas.read_csv(filename, **kwargs).
        
        Options
        -------
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        **kwargs
            pandas.DataFrame.to_csv kwargs.
            
        Returns
        -------
        pandas.DataFrame
        """
        import h5py
        if filename.endswith(".h5"):
            with h5py.File(filename, "r") as _h5f:
                if "spec_chain" in _h5f:
                    import pickle
                    _h5_group = _h5f["spec_chain"].attrs
                    _spec_chain = np.array([pickle.loads(_h5_group[str(ii)]) for ii in np.arange(len(_h5_group)-1)])
                    _lbda = pickle.loads(_h5_group["lbda"]) if "lbda" in _h5_group else None
                    _spec_chain = ProspectorSpectrum._build_spec_chain_dataframe_(_spec_chain, _lbda, warnings)
                    return _spec_chain
        else:
            return pandas.read_csv(filename, **kwargs)
    
    #------------------#
    #   Spectrometry   #
    #------------------#
    def load_spectra(self, from_file=None, size=None, savefile=None, warnings=True, **kwargs):
        """
        Build a dataframe of the spectrum chain and set it as attribute.
        
        Options
        -------
        from_file : [string or None]
            File name/directory from which to extract the dataframe.
            If a .h5 file is given, data must be saved as attributes in the group named "spec_chain".
            Else, return pandas.read_csv(filename, **kwargs).
            Default is None.
        
        size : [int or None]
            Number of step to build the spectrum chain.
            Must be lower than the total number of fit steps.
            The parameter sets building the spectra are randomly chosen over the parameter chains.
            If None, the whole chains are used.
            Default is None.
        
        savefile : [string or None]
            Save the built dataframe in the given file name.
            (see 'write_spec' method for more details).
            Default is None (meaning not saved).
        
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        **kwargs
            pandas.read_csv kwargs.
            pandas.DataFrame.to_csv kwargs.
        
        
        Returns
        -------
        Void
        """
        if from_file:
            self._spec_chain = self.read_file(from_file, warnings=warnings, **kwargs)
            return
        
        if size is not None and isinstance(size, int):
            if size > self.len_chains:
                raise ValueError(f"'size' must lower than the number of fit steps (= {self.len_chains}) (your input: {size}).")
            self._mask_chains = np.random.choice(self.len_chains, size, replace=False)
        else:
            self._mask_chains = None
        _chains = self.chains[self.mask_chains] if self.has_mask_chains() else self.chains
        _spec_chain = np.array([self.get_theta_spec(theta=_theta, unit="mgy") for _theta in _chains])
        self._spec_chain = self._build_spec_chain_dataframe_(_spec_chain, self.wavelengths, warnings)
        if savefile is not None:
            self.write_spec(savefile=savefile, **kwargs)
    
    def write_spec(self, savefile, **kwargs):
        """
        Save the spectrum chain dataframe into a given file.
        
        Parameters
        ----------
        savefile : [string or None]
            File name/directory to save the spectrum chain dataframe in.
            If a .h5 is given, create a group which will contain every spectrum step and the wavelengths as attributes.
            Else, execute pandas.DataFrame.to_csv(savefile, **kwargs) method.
        
        **kwargs
            pandas.DataFrame.to_csv kwargs.
        
        
        Returns
        -------
        Void
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
        Build and return a spectrum giving a set of parameters (running prospect.models.sedmodel.SedModel.sed())
        The corresponding wavelengths can be found in 'wavelengths' attribute (coming from self.sps.wavelengths).
        
        Parameters
        ----------
        theta : [list]
            Set of parameters to give to the SedModel to generate a spectrum.
            The parameter names are saved in 'theta_labels' attribute.
        
        unit : [string]
            Unit of the returned spectrum. Available units are:
                - "Hz": erg/s/cm2/Hz
                - "AA": erg/s/cm2/AA
                - "mgy": maggies
                - "Jy": Jansky
                - "mag": magnitude
            Default is "mgy".
        
        
        Returns
        -------
        np.array
        """
        _spec, _, _ = self.model.sed(theta, obs=self.obs, sps=self.sps)
        return tools.convert_unit(_spec, "mgy", unit, wavelength=self.wavelengths)
    
    def get_spectral_data(self, restframe=False, unit="Hz", lbda_lim=(None, None)):
        """
        Return a dictionary of the treated spectral data.
        The dictionary contains:
            - "lbda": the wavelength array of the spectrum
            - "spec_chain": the treated spectrum chain
            - "spec": the treated spectrum chain median.
            - "spec_low", "spec_up": 16% and 84% resp. of the spectrum chain
        
        Parameters
        ----------
        restframe : [bool]
            If True, the returned spectrum is restframed.
            Default is False.
        
        unit : [string]
            Unit of the returned spectrum. Available units are:
                - "Hz": erg/s/cm2/Hz
                - "AA": erg/s/cm2/AA
                - "mgy": maggies
                - "Jy": Jansky
                - "mag": magnitude
            Default is "Hz".
        
        lbda_lim : [tuple(float or None, float or None)]
            Limits on wavelength (thus in AA) to mask the spectrum.
            None for a limit means that the spectrum is not masked in the corresponding direction.
            Default is (None, None).
        
        
        Returns
        -------
        dict
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
        _spec_chain = tools.convert_unit(_spec_chain, _unit_in, unit, wavelength=_lbda)
        
        # Build the mask given by the wavelength desired limits
        _mask_lbda = self._build_spec_mask_(_lbda, lbda_lim)
        
        # Get the median and 1-sigma statistics
        _spec_low, _spec, _spec_up = np.percentile(_spec_chain.T, [16, 50, 84], axis=1)
        
        return {"lbda":_lbda[_mask_lbda], "spec_chain":_spec_chain, "spec":_spec[_mask_lbda],
                "spec_low":_spec_low[_mask_lbda], "spec_up":_spec_up[_mask_lbda]}
    
    @staticmethod
    def _build_spec_chain_dataframe_(spec_chain, lbda=None, warnings=True):
        """
        Build and return a dataframe of the spectrum chain.
        
        Parameters
        ----------
        spec_chain : [np.array]
            Array containing the spectrum chain.
        
        lbda : [np.array or None]
            Corresponding wavelengths.
            Set it as index of the dataframe, if not None.
            Default is None.
        
        Options
        -------
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        
        Returns
        -------
        pandas.DataFrame
        """
        _spec_chain = pandas.DataFrame(spec_chain.T)
        if lbda is not None:
            _spec_chain.index = lbda
            _spec_chain.index.names = ["lbda"]
        else:
            warn("No wavelengths set.")
        return _spec_chain
    
    @staticmethod
    def _build_spec_mask_(lbda, lbda_lim=(None, None)):
        """
        Build and return a boolean array to apply a mask on spectrum.
        
        Parameters
        ----------
        lbda : [np.array]
            Wavelength array.
        
        lbda_lim : [tuple(float or None, float or None)]
            Limits on wavelength (thus in AA) to mask the spectrum.
            None for a limit means that the spectrum is not masked in the corresponding direction.
            Default is (None, None).
        
        
        Returns
        -------
        np.array
        """
        _mask_lbda = np.ones(len(lbda), dtype = bool)
        if isinstance(lbda_lim, tuple) and len(lbda_lim) == 2:
            _lim_low, _lim_up = lbda_lim
            if _lim_low is not None:
                _mask_lbda &= lbda > _lim_low
            if _lim_up is not None:
                _mask_lbda &= lbda < _lim_up
        return _mask_lbda
    
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
        
        return _slbda, tools.convert_unit(_sflux_aa, "AA", unit, wavelength=_slbda)
    
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
        Build and return a dictionary containing photometry data extracted from the fitted spectrum.
        The dictionary contains:
            - "lbda": the wavelength array of the filters
            - "phot_chains": the extracted photometry chains
            - "phot": the extracted photometry chain medians.
            - "phot_low", "phot_up": 16% and 84% resp. of the extracted photometry chains
            - "filters": the filters names
        
        Parameters
        ----------
        filters : [string or list(string) or None]
            List of filters from which to get the photometry.
            Must be on the format instrument.band (e.g. sdss.u, ps1.g, ...).
            If None, extract photometry for the whole list of filters availble in 'obs' attribute.
            If "mask", extract photometry for the filters used in the SED fitting.
            Default is None.
        
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
        dict
        """
        _filternames = self._get_filternames_(filters, self.obs)
        _phot_chains = {_f:self.get_synthetic_photometry(filter=_f, restframe=restframe, unit=unit) for _f in _filternames}
        _lbda = np.array([_phot_chains[_f][0] for _f in _filternames])
        _phot_low, _phot, _phot_up = np.array([np.percentile(_phot_chains[_f][1], [16, 50, 84]) for _f in _filternames]).T
        return {"lbda":_lbda, "phot_chains":_phot_chains, "phot":_phot, "phot_low":_phot_low, "phot_up":_phot_up, "filters":_filternames}
    
    def get_phot_obs(self, filters=None, unit="mag"):
        """
        Build and return a dictionary containing photometry data used for SED fitting.
        The dictionary contains:
            - "lbda": the wavelength array of the filters
            - "phot": observed photometry
            - "phot_unc": observed photometry uncertainty
            - "filters": the filters names
        
        Parameters
        ----------
        filters : [string or list(string) or None]
            List of filters from which to get the photometry.
            Must be on the format instrument.band (e.g. sdss.u, ps1.g, ...).
            If None, extract photometry for the whole list of filters availble in 'obs' attribute.
            If "mask", extract photometry for the filters used in the SED fitting.
            If "~mask", extract photometry for the filters not used in the SED fitting.
            Default is None.
        
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
        dict
        """
        _filternames, _filt_idx = self._get_filternames_(filters, self.obs, return_idx=True)
        _filters = np.array(self.obs["filters"])[_filt_idx]
        _lbda = np.array([_f.wave_average for _f in _filters])
        _phot = np.array(self.obs["maggies"])[_filt_idx]
        _phot_unc = np.array(self.obs["maggies_unc"])[_filt_idx]
        _unit_in = "mgy"
        
        # Deredshift
        #if restframe:
        #    _phot, _phot_unc = tools.convert_flux_unit([_phot, _phot_unc], _unit_in, "AA", wavelength=_lbda)
        #    _lbda, _phot, _phot_unc = tools.deredshift(lbda=_lbda, flux=_phot, z=self.z, variance=_phot_unc**2, exp=3)
        #    _unit_in = "AA"
        
        # Change unit
        _phot, _phot_unc = tools.convert_unit(data=_phot, data_unc=_phot_unc, unit_in=_unit_in,
                                              unit_out=("AA" if unit=="mag" else unit), wavelength=_lbda)
        
        return {"lbda":_lbda, "phot":_phot, "phot_unc":_phot_unc, "filters":_filternames}
    
    @staticmethod
    def _get_filternames_(filters, obs, return_idx=False):
        """
        Build and return an array of compatible filter names.
        
        Parameters
        ----------
        filters : [string or list(string) or None]
            List of filters from which to get a convenient format of filter names.
            Must be on the format instrument.band (e.g. sdss.u, ps1.g, ...).
            If None, return the whole list of filters availble in 'obs'.
            If "mask", return the filters used in the SED fitting.
            If "~mask", return the filters not used in the SED fitting.
            Default is None.
        
        obs : [dict]
            Prospector compatible 'obs' dictionary.
        
        Options
        -------
        return_idx : [bool]
            If True, return in addition an array containing the indexes of the given filters in 'self.obs["filternames"]'.
            Default is False.
        
        
        Returns
        -------
        np.array
        """
        if filters is None:
            _filters = io.pysed_to_filters(obs["filternames"])
        elif filters == "mask":
            _filters = io.pysed_to_filters(obs["filternames"])[obs["phot_mask"]]
        elif filters == "~mask":
            _filters = io.pysed_to_filters(obs["filternames"])[~obs["phot_mask"]]
        else:
            _filters = np.atleast_1d(filters)
            try:
                _filters = io.pysed_to_filters([(_f if _f in obs["filternames"] else io.filters_to_pysed(_f)[0]) for _f in _filters])
            except KeyError:
                raise ValueError(f"One or more of the given filters are not available \n(filters = {filters}).\n"+
                                 "They must be like 'instrument.band' (eg: 'sdss.u') or prospector compatible filter names.")
        if return_idx:
            _filt_idx = [np.where(np.array(obs["filternames"]) == _f)[0][0] for _f in io.filters_to_pysed(_filters)]
            return _filters, _filt_idx
        return _filters
    
    #--------------#
    #   Plotting   #
    #--------------#
    def show(self, ax=None, figsize=[7,3.5], ax_rect=[0.1,0.2,0.8,0.7], unit="Hz", restframe=False,
             lbda_lim=(None, None), spec_prop={}, spec_unc_prop={},
             filters=None, phot_prop={}, phot_unc_prop={}, show_obs={},
             show_legend={}, set_logx=True, set_logy=True, show_filters={}, savefile=None):
        """
        Plot the spectrum.
        Return the figure and the axe: {"fig":pyplot.Figure, "ax":pyplot.Axes}.
        
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
            If True, data will be deredshifted.
            Default is False.
        
        lbda_lim : [tuple(float or None, float or None)]
            Limits on wavelength (thus in AA) to mask the spectrum.
            None for a limit means that the spectrum is not masked in the corresponding direction.
            Default is (None, None).
        
        spec_prop : [dict or None]
            Spectrum pyplot.plot kwargs.
            If bool(spec_prop) == False, the spectrum is not plotted.
            Default is {}.
        
        spec_unc_prop : [dict or None]
            Spectrum uncertainties pyplot.fill_between kwargs.
            If bool(spec_unc_prop) == False, the uncertainties are not plotted.
            Default is {}.
        
        filters : [string or list(string) or None]
            List of filters from which to get the photometry.
            Must be on the format instrument.band (e.g. sdss.u, ps1.g, ...).
            If None, extract photometry for the whole list of filters availble in 'obs'.
            If "mask", extract photometry for the filters used in the SED fitting.
            Default is None.
        
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
            pyplot.legend kwargs.
            If bool(show_legend) == False, the legend is not plotted.
            Default is {}.
        
        set_logx, set_logy : [bool]
            If True, set the corresponding axis in log scale.
            Default is True (for both).
        
        show_filters : [dict or None]
            Filters' transmission pyplot.plot kwargs.
            If bool(show_filters) == False, the filters' transmission are not plotted.
            Default is {}.
        
        savefile : [string or None]
            Give a directory to save the figure.
            Default is None (not saved).
        
        
        Returns
        -------
        dict
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_axes(ax_rect)
        else:
            fig = ax.figure
        
        _handles = []
        _labels = []
        # SED
        _h_spec = []
        _l_spec = []
        if spec_prop:
            _spec_data = self.get_spectral_data(restframe=restframe, unit=unit, lbda_lim=lbda_lim)
            ax.plot(_spec_data["lbda"], _spec_data["spec"], **spec_prop)
            _h_spec.append(mlines.Line2D([], [], **spec_prop))
            _l_spec.append("Fitted spectrum")
        if spec_unc_prop:
            if "_spec_data" not in locals():
                _spec_data = self.get_spectral_data(restframe=restframe, unit=unit, lbda_lim=lbda_lim)
            _h_spec.insert(0, ax.fill_between(_spec_data["lbda"], _spec_data["spec_low"], _spec_data["spec_up"], **spec_unc_prop))
            _l_spec.append(r"1-$\sigma$ spectrum confidence")
        if _h_spec and _l_spec:
            _handles.append(tuple(_h_spec))
            _labels.append("\n".join(_l_spec))
        
        # Photometry
        _h_phot = []
        _l_phot = []
        if phot_prop:
            _phot_data = self.get_phot_data(filters=filters, restframe=restframe, unit=unit)
            _h_phot.append(ax.scatter(_phot_data["lbda"], _phot_data["phot"], **phot_prop))
            _l_phot.append("Fitted photometry")
        if phot_unc_prop:
            if "_phot_data" not in locals():
                _phot_data = self.get_phot_data(filters=filters, restframe=restframe, unit=unit)
            _h_phot.insert(0, ax.errorbar(_phot_data["lbda"], _phot_data["phot"],
                                          yerr=[_phot_data["phot"]-_phot_data["phot_low"],
                                                _phot_data["phot_up"]-_phot_data["phot"]],
                                          **phot_unc_prop))
            _l_phot.append(r"1-$\sigma$ photometry confidence")
        if _h_phot and _l_phot:
            _handles.append(tuple(_h_phot))
            _labels.append("\n".join(_l_phot))
        
        if show_obs:
            _phot_obs = self.get_phot_obs(filters="mask", unit=unit)
            _handles.append(ax.errorbar(_phot_obs["lbda"], _phot_obs["phot"], yerr=_phot_obs["phot_unc"], **show_obs))
            _labels.append("Observed photometry")
            _phot_obs_mask = self.get_phot_obs(filters="~mask", unit=unit)
            _show_obs_mask = show_obs.copy()
            _show_obs_mask["alpha"] = _show_obs_mask["alpha"] * 0.3 if "alpha" in _show_obs_mask else 0.3
            _handles.append(ax.errorbar(_phot_obs_mask["lbda"], _phot_obs_mask["phot"],
                                        yerr=_phot_obs_mask["phot_unc"], **_show_obs_mask))
            _labels.append("Observed photometry\n(not used for SED fitting)")
            
        ax.set_xlabel(r"wavelentgh [$\AA$]", fontsize="large")
        ax.set_ylabel(f"{'magnitude' if unit=='mag' else 'flux'} [{tools.get_unit_label(unit)}]", fontsize="large")
        if set_logx:
            ax.set_xscale("log")
        if set_logy:
            ax.set_yscale("log")
        
        if show_filters:
            _filternames, _filt_idx = self._get_filternames_(filters, self.obs, return_idx=True)
            _filters = np.array(self.obs["filters"])[_filt_idx]
            _ymin, _ymax = ax.get_ylim()
            ax.set_ylim(_ymin, _ymax)
            for _f in _filters:
                _w, _t = tools.fix_trans(_f.wavelength, _f.transmission)
                _t = _t*(1./100. if _t.max() > 1. else 1.)
                if set_logy:
                    _t = 10**(0.25*(np.log10(_ymax/_ymin))) * _t * _ymin + _ymin
                else:
                    _t = 0.35 * (_ymax - _ymin) * _t + _ymin
                ax.plot(_w, _t, **show_filters)
        
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
        if not hasattr(self, "_spec_chain") or self._spec_chain is None:
            self.load_spectra()
        return self._spec_chain
