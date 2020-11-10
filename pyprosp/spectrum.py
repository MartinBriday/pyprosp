
import pandas
import numpy as np
from warnings import warn

from . import tools
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
        
        Options
        -------
        filename : [string or None]
            File name/directory from which to extract the spectrum chain.
            If a .h5 file is given, data must be saved as attributes in the group named "spec_chain".
            Else, the code try to execute pandas.read_csv(filename, **kwargs).
        
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        **kwargs
            pandas.DataFrame.read_csv kwargs.
        
        
        Returns
        -------
        Void
        """
        if kwargs != {}:
            self.set_data(**kwargs)
    
    def set_data(self, chains, model, obs, sps, filename=None, warnings=True, **kwargs):
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
        
        Options
        -------
        filename : [string or None]
            File name/directory from which to extract the spectrum chain.
            If a .h5 file is given, data must be saved as attributes in the group named "spec_chain".
            Else, the code try to execute pandas.read_csv(filename, **kwargs).
        
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        **kwargs
            pandas.DataFrame.read_csv kwargs.
        
        
        Returns
        -------
        Void
        """
        self._chains = chains
        self._model = model
        self._obs = obs
        self._sps = sps
        self._param_chains = {_p:np.array([jj[ii] for jj in self.chains]) for ii, _p in enumerate(self.theta_labels)}
        if filename:
            self._spec_chain = self.read_file(filename=filename, warnings=warnings, **kwargs)
    
    @classmethod
    def from_h5(cls, filename, warnings=True, **kwargs):
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
        
        **kwargs
            pandas.DataFrame.read_csv kwargs.
        
        
        Returns
        -------
        ProspectorSpectrum
        """
        from .prospector import Prospector
        _prosp = Prospector.from_h5(filename=filename, build_sps=True, warnings=warnings)
        _this = cls(chains=_prosp.chains, model=_prosp.model, obs=_prosp.obs, sps=_prosp.sps,
                    filename=filename, warnings=warnings, **kwargs)
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
            pandas.DataFrame.read_csv kwargs.
            
        Returns
        -------
        pandas.DataFrame
        """
        if filename.endswith(".h5"):
            import h5py
            with h5py.File(filename, "r") as _h5f:
                if "spec_chain" in _h5f:
                    import pickle
                    _h5_group = _h5f["spec_chain"].attrs
                    _spec_chain = np.array([pickle.loads(_h5_group[str(ii)]) for ii in np.arange(len(_h5_group)-1)])
                    _lbda = pickle.loads(_h5_group["lbda"]) if "lbda" in _h5_group else None
                    _spec_chain = ProspectorSpectrum._build_spec_chain_dataframe_(_spec_chain, _lbda, warnings)
                    return _spec_chain
                else:
                    if warnings:
                        warn("""There is no group named "spec_chain" in the .h5 file you gave.""")
        else:
            try:
                return pandas.read_csv(filename, **kwargs)
            except Exception:
                if warnings:
                    warn(f"There is an error trying to run 'pandas.read_csv(filename, **kwargs)' with your inputs: {Exception}.")
        if warnings:
            warn("The spectrum chain is NOT loaded!")
    
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
            self._spec_chain = self.read_file(filename=from_file, warnings=warnings, **kwargs)
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
    
    def get_spectral_data(self, restframe=False, exp=3, unit="Hz", lbda_lim=(None, None)):
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
        
        exp : [float]
            // Ignored if 'restframe = False' //
            Exposant for the redshift flux dilution (exp=3 is for erg/s/cm2/AA spectra).
            Default is 3.
        
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
            _lbda, _spec_chain = tools.deredshift(lbda=_lbda, flux=_spec_chain, z=self.z, variance=None, exp=exp)
            _unit_in = "AA"
        
        # Change unit
        _spec_chain = tools.convert_unit(_spec_chain, _unit_in, unit, wavelength=_lbda)
        
        # Build the mask given by the wavelength desired limits
        _mask_lbda = self._build_spec_mask_(_lbda, lbda_lim)
        
        # Get the median and 1-sigma statistics
        _spec_low, _spec, _spec_up = np.percentile(_spec_chain.T, [16, 50, 84], axis=1)
        
        return {"lbda":_lbda[_mask_lbda], "spec_chain":_spec_chain, "spec":_spec[_mask_lbda],
                "spec_low":_spec_low[_mask_lbda], "spec_up":_spec_up[_mask_lbda]}
    
    def get_spec_obs(self, restframe=False, exp=3, unit="Hz", lbda_lim=(None, None)):
        """
        Return dictionar.y.ies of the treated observed spectral data.
        The dictionar.y.ies contains:
            - "lbda": the wavelength array of the spectrum
            - "spec": the treated observed spectrum.
            - "spec_err": the treated observed spectrum uncertainty.

        Parameters
        ----------
        restframe : [bool]
            If True, the returned spectrum is restframed.
            Default is False.
        
        exp : [float]
            // Ignored if 'restframe = False' //
            Exposant for the redshift flux dilution (exp=3 is for erg/s/cm2/AA spectra).
            Default is 3.
        
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
            If you give "mask", return a tuple of dictionaries corresponding to the mask used for SED fitting.
            If you give "~mask", return a tuple of dictionaries corresponding to the inversed mask used for SED fitting.
            Default is (None, None).
        

        Returns
        -------
        dict
        """
        _spec = [np.array([]), np.array([]), np.array([])]
        for ii, _k in enumerate(["wavelength", "spectrum", "unc"]):
            try:
                _spec[ii] = np.array(self.obs[_k].copy())
            except AttributeError:
                continue
        _lbda, _spec, _spec_err = _spec
        _unit_in = "mgy"
        
        if len(_lbda) + len(_spec) + len(_spec_err) == 0:
            return ({"lbda":_lbda, "spec":_spec, "spec_err":_spec_err} if lbda_lim not in ["mask", "~mask"] else 
                    [{"lbda":_lbda, "spec":_spec, "spec_err":_spec_err}])
        
        # Deredshift
        if restframe:
            _spec = tools.convert_flux_unit(_spec, _unit_in, "AA", wavelength=_lbda)
            _spec_err = tools.convert_flux_unit(_spec_err, _unit_in, "AA", wavelength=_lbda)
            _lbda, _spec, _spec_err = tools.deredshift(lbda=_lbda, flux=_spec, z=self.z, variance=_spec_err**2, exp=exp)
            _unit_in = "AA"
        
        # Change unit
        _spec = tools.convert_unit(_spec, _unit_in, unit, wavelength=_lbda)
        _spec_err = tools.convert_flux_unit(_spec_err, _unit_in, unit, wavelength=_lbda)
        
        #Build mask
        _spec_mask = self._build_obs_spec_mask_(self.obs, _lbda, lbda_lim)
        
        if isinstance(_spec_mask[0], np.ndarray):
            return [{"lbda":_lbda[_sm], "spec":_spec[_sm], "spec_err":_spec_err[_sm]} for _sm in _spec_mask]
        else:
            return {"lbda":_lbda[_spec_mask], "spec":_spec[_spec_mask], "spec_err":_spec_err[_spec_mask]}
    
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
    
    @staticmethod
    def _build_obs_spec_mask_(obs, lbda=None, lbda_lim="mask"):
        """
        Build and return a boolean array to apply a mask on the observed spectrum.
        
        Parameters
        ----------
        obs : [dict]
            Prospector compatible 'obs' dictionary.
        
        lbda : [np.array or None]
            Wavelength array to apply the boundaries on.
            If None, 'lbda_lim' is expected to be either "mask" or "~mask".
            Default is None.
        
        lbda_lim : [tuple(float or None, float or None)]
            Limits on wavelength (thus in AA) to mask the spectrum.
            None for a limit means that the spectrum is not masked in the corresponding direction.
            If you give "mask", return a tuple of dictionaries corresponding to the mask used for SED fitting.
            If you give "~mask", return a tuple of dictionaries corresponding to the inversed mask used for SED fitting.
            Default is "mask".
        
        
        Returns
        -------
        np.array
        """
        _spec_mask = ProspectorSpectrum._build_spec_mask_(lbda, lbda_lim)
        try:
            _spec_mask = obs["mask"]
            if lbda_lim == "mask":
                _spec_mask = np.where(_spec_mask)[0]
            elif lbda_lim == "~mask":
                _spec_mask = np.where(~_spec_mask)[0]
            else:
                raise AttributeError()
            _spec_mask = np.split(_spec_mask, np.where(np.diff(_spec_mask) != 1)[0] + 1)
            for ii, _sm in enumerate(_spec_mask.copy()):
                if len(_sm) > 0:
                    if _sm[0] != 0 and (lbda_lim == "mask" 
                                        and (_sm[0] != (_spec_mask[ii-1][-1] + 2) if ii > 0 else True) 
                                        or len(_spec_mask[ii]) == 1):
                        _sm = np.insert(_sm, 0, _sm[0]-1)
                    if _sm[-1] != (len(obs["mask"]) - 1) and (lbda_lim == "mask" 
                                                              and (_sm[-1] != (_spec_mask[ii+1][0] - 2) 
                                                                   if ii < (len(_spec_mask) - 1)
                                                                   else True)
                                                              or len(_spec_mask[ii]) == 1):
                        _sm = np.append(_sm, _sm[-1]+1)
                    _spec_mask[ii] = _sm
        except AttributeError:
            pass
        except KeyError:
            pass
        return _spec_mask
    
    #----------------#
    #   Photometry   #
    #----------------#
    def get_synthetic_photometry(self, filter, restframe=False, exp=3, unit="Hz"):
        """
        Return photometry synthesized through the given filter/bandpass.
        The returned data are (effective wavelength, synthesize flux/mag) in an array with same size as for 'self.spec_chain'.

        Parameters
        ----------
        filter : [string, sncosmo.BandPass, 2D array]
            The filter through which the spectrum will be synthesized.
            Accepted input format:
            - string: name of a known filter (instrument.band), the actual bandpass will be grabbed using data 
                      from 'self.obs["filters"]'
            - 2D array: containing wavelength and transmission --> sncosmo.bandpass.BandPass(*filter)
            - sncosmo.bandpass.BandPass
        
        Options
        -------
        restframe : [bool]
            If True, the spectrum is first deredshifted before doing the synthetic photometry.
            Default is False.
        
        exp : [float]
            // Ignored if 'restframe = False' //
            Exposant for the redshift flux dilution (exp=3 is for erg/s/cm2/AA spectra).
            Default is 3.

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
        if isinstance(filter, str):
            try:
                _filter = self.obs["filters"][self.obs["filternames"].index(filter)]
            except (KeyError, ValueError):
                from sedpy.observate import load_filters
                _filter = load_filters([filter])[0]
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
        _sflux_aa = self.synthesize_photometry(_bp.wave, _bp.trans, restframe=restframe, exp=exp)
        _slbda = _bp.wave_eff/(1+self.z) if restframe else _bp.wave_eff
        
        return _slbda, tools.convert_unit(_sflux_aa, "AA", unit, wavelength=_slbda)
    
    def synthesize_photometry(self, filter_lbda, filter_trans, restframe=False, exp=3):
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
        
        exp : [float]
            // Ignored if 'restframe = False' //
            Exposant for the redshift flux dilution (exp=3 is for erg/s/cm2/AA spectra).
            Default is 3.
        
        
        Returns
        -------
        float
        """
        _spec_data = self.get_spectral_data(restframe=restframe, exp=exp, unit="AA", lbda_lim=(None, None))
        return tools.synthesize_photometry(_spec_data["lbda"], _spec_data["spec_chain"], filter_lbda, filter_trans, normed=True)
    
    def get_phot_data(self, filters=None, restframe=False, exp=3, unit="mag", plot_format=False):
        """
        Build and return a dictionary containing photometry data extracted from the fitted spectrum.
        
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
        
        exp : [float]
            // Ignored if 'restframe = False' //
            Exposant for the redshift flux dilution (exp=3 is for erg/s/cm2/AA spectra).
            Default is 3.
        
        unit : [string]
            Unit of the returned photometry. Available units are:
                - "Hz": erg/s/cm2/Hz
                - "AA": erg/s/cm2/AA
                - "mgy": maggies
                - "Jy": Jansky
                - "mag": magnitude
            Default is "Hz".
        
        Options
        -------
        plot_format : [bool]
            Returning dictionary format choice.
            If True, return a convenient format for plots:
                - "lbda": the wavelength array of the filters
                - "phot_chains": the extracted photometry chains
                - "phot": the extracted photometry chain medians
                - "phot_low", "phot_up": 16% and 84% resp. of the extracted photometry chains
                - "filters": the filters names
            If False, the returned dictionary is user friendly:
                (for each _filter in "filters")
                - _filter: {"lbda": the wavelength array of the _filter
                            "phot_chain": the extracted photometry chains
                            "phot": the extracted photometry (median, -sigma, +sigma)}
            Default is False.
        
        
        Returns
        -------
        dict
        """
        _filternames = self._get_filternames_(filters, self.obs)
        _phot_chains = {_f:self.get_synthetic_photometry(filter=_f, restframe=restframe, unit=unit, exp=exp) for _f in _filternames}
        _lbda = np.array([_phot_chains[_f][0] for _f in _filternames])
        _phot_chains = np.array([_phot_chains[_f][1] for _f in _filternames])
        try:
            _phot_low, _phot, _phot_up = np.array([np.percentile(_pc, [16, 50, 84]) for _pc in _phot_chains]).T
        except ValueError:
            _phot_low, _phot, _phot_up = np.array([[]]*3)
        dict_out = {"lbda":_lbda, "phot_chains":_phot_chains, "phot":_phot, "phot_low":_phot_low, "phot_up":_phot_up, 
                    "filters":_filternames}
        if plot_format:
            return dict_out
        else:
            return {_f:{"lbda":dict_out["lbda"][ii],
                        "phot_chain":dict_out["phot_chains"][ii],
                        "phot":(dict_out["phot"][ii],
                                dict_out["phot"][ii]-dict_out["phot_low"][ii],
                                dict_out["phot_up"][ii]-dict_out["phot"][ii])}
                    for ii, _f in enumerate(dict_out["filters"])}
    
    def get_phot_obs(self, filters=None, unit="mag", plot_format=False):
        """
        Build and return a dictionary containing photometry data used for SED fitting.
        The dictionary contains:
            - "lbda": the wavelength array of the filters
            - "phot": observed photometry
            - "phot_err": observed photometry uncertainty
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
        
        Options
        -------
        plot_format : [bool]
            Returning dictionary format choice.
            If True, return a convenient format for plots:
                - "lbda": the wavelength array of the filters
                - "phot": measured photometry
                - "phot_err": photometry uncertainty
                - "filters": the filters names
            If False, the returned dictionary is user friendly:
                (for each _filter in the given 'filters')
                - _filter: {"lbda": the wavelength array of the _filter
                            "phot": measured photometry
                            "phot_err": photometry uncertainty}
            Default is False.
        
        
        Returns
        -------
        dict
        """
        _filternames, _filt_idx = self._get_filternames_(filters, self.obs, return_idx=True)
        _filters = np.array(self.obs["filters"] if self.obs["filters"] is not None else [])[_filt_idx]
        _lbda = np.array([_f.wave_average for _f in _filters])
        _phot = np.array(self.obs["maggies"] if self.obs["maggies"] is not None else [])[_filt_idx]
        _phot_err = np.array(self.obs["maggies_unc"] if self.obs["maggies_unc"] is not None else [])[_filt_idx]
        _unit_in = "mgy"
        
        # Change unit
        _phot, _phot_err = tools.convert_unit(data=_phot, data_unc=_phot_err, unit_in=_unit_in,
                                              unit_out=("AA" if unit=="mag" else unit), wavelength=_lbda)
        
        dict_out = {"lbda":_lbda, "phot":_phot, "phot_err":_phot_err, "filters":_filternames}
        if plot_format:
            return dict_out
        else:
            return {_f:{"lbda":dict_out["lbda"][ii],
                        "phot":dict_out["phot"][ii],
                        "phot_err":dict_out["phot_err"][ii]}
                    for ii, _f in enumerate(dict_out["filters"])}
    
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
        try:
            _filternames = obs["filternames"].copy()
        except KeyError:
            _filternames = []
            if filters in ["mask", "~mask"]:
                filters = None
        if filters is None:
            _filters = io.pysed_to_filters(_filternames)
        elif filters == "mask":
            _filters = io.pysed_to_filters(_filternames)[obs["phot_mask"]]
        elif filters == "~mask":
            _filters = io.pysed_to_filters(_filternames)[~obs["phot_mask"]]
        else:
            _filters = np.atleast_1d(filters)
            try:
                _filters = io.pysed_to_filters([(_f if _f in _filternames else io.filters_to_pysed(_f)[0]) for _f in _filters])
            except KeyError:
                raise ValueError(f"One or more of the given filters are not available \n(filters = {filters}).\n"+
                                 "They must be like 'instrument.band' (eg: 'sdss.u') or prospector compatible filter names.")
        if return_idx:
            try:
                _filt_idx = [np.where(np.array(_filternames) == _f)[0][0] for _f in io.filters_to_pysed(_filters)]
            except IndexError:
                _filt_idx = []
            return _filters, _filt_idx
        return _filters
    
    #--------------#
    #   Plotting   #
    #--------------#
    def show(self, ax=None, figsize=(10,6), ax_rect=[0.1, 0.2, 0.8, 0.7], unit="Hz", restframe=False, exp=3, 
             lbda_lim=(1e3, 1e5), spec_prop={"c":"k", "lw":0.5, "zorder":4}, 
             spec_unc_prop={"color":"0.7", "alpha":0.35, "lw":0, "zorder":2},
             filters=None, phot_prop={"marker":"o", "s":60, "fc":"xkcd:azure", "ec":"b", "zorder":6},
             phot_unc_prop={"marker":"", "ls":"None", "ecolor":"C0", "zorder":5},
             obs_spec_prop={"c":"r", "lw":0.5, "zorder":4},
             obs_spec_unc_prop={"color":"r", "lw":0.2, "alpha":0.1, "zorder":2},
             obs_phot_prop={"marker":"s", "s":50, "ls":"None", "facecolors":"r", "edgecolors":"b", "zorder":4},
             obs_phot_unc_prop={"marker":"", "ls":"None", "ecolor":"0.7", "zorder":3},
             show_legend={"loc":"best", "frameon":False, "labelspacing":0.8}, set_logx=True, set_logy=True,
             show_filters={"lw":1.5, "color":"gray", "alpha":0.7, "zorder":1}, savefile=None):
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
        
        exp : [float]
            // Ignored if 'restframe = False' //
            Exposant for the redshift flux dilution (exp=3 is for erg/s/cm2/AA spectra).
            Default is 3.
        
        lbda_lim : [tuple(float or None, float or None)]
            Limits on wavelength (thus in AA) to mask the spectrum.
            None for a limit means that the spectrum is not masked in the corresponding direction.
            Default is (None, None).
        
        spec_prop : [dict or None]
            Fitted spectrum pyplot.plot kwargs.
            If bool(spec_prop) == False, the spectrum is not plotted.
            Default is {"c":"k", "lw":0.5, "zorder":4}.
        
        spec_unc_prop : [dict or None]
            Fitted spectrum uncertainties pyplot.fill_between kwargs.
            If bool(spec_unc_prop) == False, the uncertainties are not plotted.
            Default is {"color":"0.7", "alpha":0.35, "lw":0, "zorder":2}.
        
        filters : [string or list(string) or None]
            List of filters from which to get the photometry.
            Must be on the format instrument.band (e.g. sdss.u, ps1.g, ...).
            If None, extract photometry for the whole list of filters availble in 'obs'.
            If "mask", extract photometry for the filters used in the SED fitting.
            Default is None.
        
        phot_prop : [dict or None]
            Photometry pyplot.scatter kwargs.
            If bool(phot_prop) == False, the photometric points are not plotted.
            Default is {"marker":"o", "s":60, "fc":"xkcd:azure", "ec":"b", "zorder":6}.
        
        phot_unc_prop : [dict or None]
            Photometry uncertainties pyplot.errorbar kwargs.
            If bool(phot_unc_prop) == False, the photometric uncertainties are not plotted.
            Default is {"marker":"", "ls":"None", "ecolor":"C0", "zorder":5}.
        
        obs_spec_prop : [dict or None]
            Observed spectrum pyplot.errorbar kwargs.
            If bool(obs_spec_prop) == False, the observed spectrum is not plotted.
            Default is {"c":"r", "lw":0.5, "zorder":4}.
        
        obs_spec_unc_prop : [dict or None]
            Observed spectrum uncertainties pyplot.errorbar kwargs.
            If bool(obs_spec_unc_prop) == False, the observed spectrum uncertainties are not plotted.
            Default is {"color":"r", "lw":0.2, "alpha":0.1, "zorder":2}.
        
        obs_phot_prop : [dict or None]
            Observed photometry pyplot.scatter kwargs.
            If bool(show_obs) == False, the observed photometric points are not plotted.
            Default is {"marker":"s", "s":50, "ls":"None", "facecolors":"r", "edgecolors":"b", "zorder":4}.
        
        obs_phot_unc_prop : [dict or None]
            Observed photometry uncertainties pyplot.errorbar kwargs.
            If bool(show_obs) == False, the observed photometric points are not plotted.
            Default is {"marker":"", "ls":"None", "ecolor":"0.7", "zorder":3}.
        
        show_legend : [dict or None]
            pyplot.legend kwargs.
            If bool(show_legend) == False, the legend is not plotted.
            Default is {"loc":"best", "frameon":False, "labelspacing":0.8}.
        
        set_logx, set_logy : [bool]
            If True, set the corresponding axis in log scale.
            Default is True (for both).
        
        show_filters : [dict or None]
            Filters' transmission pyplot.plot kwargs.
            If bool(show_filters) == False, the filters' transmission are not plotted.
            Default is {"lw":1.5, "color":"gray", "alpha":0.7, "zorder":1}.
        
        savefile : [string or None]
            Give a directory to save the figure.
            Default is None (not saved).
        
        
        Returns
        -------
        dict
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
        _h_spec = []
        if spec_prop:
            _spec_data = self.get_spectral_data(restframe=restframe, exp=exp, unit=unit, lbda_lim=lbda_lim)
            if len(_spec_data["lbda"]) > 0:
                ax.plot(_spec_data["lbda"], _spec_data["spec"], **spec_prop)
                _h_spec.append(mlines.Line2D([], [], **spec_prop))
        if spec_unc_prop:
            if "_spec_data" not in locals():
                _spec_data = self.get_spectral_data(restframe=restframe, exp=exp, unit=unit, lbda_lim=lbda_lim)
            if len(_spec_data["lbda"]) > 0:
                _h_spec.insert(0, ax.fill_between(_spec_data["lbda"], _spec_data["spec_low"], _spec_data["spec_up"], **spec_unc_prop))
        if _h_spec:
            _handles.append(tuple(_h_spec))
            _labels.append("Fitted spectrum")
        
        _h_obs_spec = []
        _h_obs_spec_mask = []
        if obs_spec_prop:
            _alpha_coef = obs_spec_prop.pop("alpha_coef") if "alpha_coef" in obs_spec_prop else 0.3
            _obs_spec_mask_prop = obs_spec_prop.copy()
            _obs_spec_mask_prop["alpha"] = (_obs_spec_mask_prop["alpha"] * _alpha_coef 
                                            if "alpha" in _obs_spec_mask_prop else _alpha_coef)
            _spec_obs_mask = self.get_spec_obs(restframe=restframe, exp=exp, unit=unit, lbda_lim="~mask")
            if len(_spec_obs_mask) > 0 and len(_spec_obs_mask[0]["lbda"]) > 0:
                for _so in _spec_obs_mask:
                    ax.plot(_so["lbda"], _so["spec"], **_obs_spec_mask_prop)
                _h_obs_spec_mask.append(mlines.Line2D([], [], **_obs_spec_mask_prop))
            _spec_obs = self.get_spec_obs(restframe=restframe, exp=exp, unit=unit, lbda_lim="mask")
            if len(_spec_obs) > 0 and len(_spec_obs[0]["lbda"]) > 0:
                for _so in _spec_obs:
                    ax.plot(_so["lbda"], _so["spec"], **obs_spec_prop)
                _h_obs_spec.append(mlines.Line2D([], [], **obs_spec_prop))
        if obs_spec_unc_prop:
            _alpha_coef = obs_spec_unc_prop.pop("alpha_coef") if "alpha_coef" in obs_spec_unc_prop else 0.3
            _obs_spec_unc_mask_prop = obs_spec_unc_prop.copy()
            _obs_spec_unc_mask_prop["alpha"] = (_obs_spec_unc_mask_prop["alpha"] * _alpha_coef 
                                                if "alpha" in _obs_spec_unc_mask_prop else _alpha_coef)
            if "_spec_obs_mask" not in locals():
                _spec_obs_mask = self.get_spec_obs(restframe=restframe, exp=exp, unit=unit, lbda_lim="~mask")
            if len(_spec_obs_mask) > 0 and len(_spec_obs_mask[0]["lbda"]) > 0:
                for _so in _spec_obs_mask:
                    _h_obs_spec_mask.insert(0, ax.fill_between(_so["lbda"], _so["spec"] - _so["spec_err"], 
                                                               _so["spec"] + _so["spec_err"], **_obs_spec_unc_mask_prop))
            if "_spec_obs" not in locals():
                _spec_obs = self.get_spec_obs(restframe=restframe, exp=exp, unit=unit, lbda_lim="mask")
            if len(_spec_obs) > 0 and len(_spec_obs[0]["lbda"]) > 0:
                for _so in _spec_obs:
                    _h_obs_spec.insert(0, ax.fill_between(_so["lbda"], _so["spec"] - _so["spec_err"], 
                                                          _so["spec"] + _so["spec_err"], **obs_spec_unc_prop))
        if _h_obs_spec:
            _handles.append(tuple(_h_obs_spec))
            _labels.append("Observed spectrum")
        if _h_obs_spec_mask:
            _handles.append(tuple(_h_obs_spec_mask))
            _labels.append("Observed spectrum\n(not used for SED fitting)")
        
        # Photometry
        _h_phot = []
        if phot_prop:
            _phot_data = self.get_phot_data(filters=filters, restframe=restframe, unit=unit, plot_format=True)
            if len(_phot_data["lbda"]) > 0:
                _h_phot.append(ax.scatter(_phot_data["lbda"], _phot_data["phot"], **phot_prop))
        if phot_unc_prop:
            if "_phot_data" not in locals():
                _phot_data = self.get_phot_data(filters=filters, restframe=restframe, unit=unit, plot_format=True)
            if len(_phot_data["lbda"]) > 0:
                _h_phot.insert(0, ax.errorbar(_phot_data["lbda"], _phot_data["phot"],
                                              yerr=[_phot_data["phot"]-_phot_data["phot_low"],
                                                    _phot_data["phot_up"]-_phot_data["phot"]],
                                              **phot_unc_prop))
        if _h_phot:
            _handles.append(tuple(_h_phot))
            _labels.append("Fitted photometry")
        
        _h_obs_phot = []
        _h_obs_mask_phot = []
        if obs_phot_prop:
            _alpha_coef = obs_phot_prop.pop("alpha_coef") if "alpha_coef" in obs_phot_prop else 0.3
            _phot_obs = self.get_phot_obs(filters="mask", unit=unit, plot_format=True)
            if len(_phot_obs["lbda"]) > 0:
                _h_obs_phot.append(ax.scatter(_phot_obs["lbda"], _phot_obs["phot"], **obs_phot_prop))
            _phot_obs_mask = self.get_phot_obs(filters="~mask", unit=unit, plot_format=True)
            if len(_phot_obs_mask["lbda"]) > 0:
                _obs_phot_mask_prop = obs_phot_prop.copy()
                _obs_phot_mask_prop["alpha"] = (_obs_phot_mask_prop["alpha"] * _alpha_coef 
                                                if "alpha" in _obs_phot_mask_prop else _alpha_coef)
                _h_obs_mask_phot.append(ax.scatter(_phot_obs_mask["lbda"], _phot_obs_mask["phot"], **_obs_phot_mask_prop))
        if obs_phot_unc_prop:
            _alpha_coef = obs_phot_prop.pop("alpha_coef") if "alpha_coef" in obs_phot_unc_prop else 0.3
            if "_phot_obs" not in locals():
                _phot_obs = self.get_phot_obs(filters="mask", unit=unit, plot_format=True)
            if len(_phot_obs["lbda"]) > 0:
                _h_obs_phot.insert(0, ax.errorbar(_phot_obs["lbda"], _phot_obs["phot"], 
                                                  yerr=_phot_obs["phot_err"], **obs_phot_unc_prop))
            if "_phot_obs_mask" not in locals():
                _phot_obs_mask = self.get_phot_obs(filters="~mask", unit=unit, plot_format=True)
            if len(_phot_obs_mask["lbda"]) > 0:
                _obs_phot_unc_mask_prop = obs_phot_unc_prop.copy()
                _obs_phot_unc_mask_prop["alpha"] = (_obs_phot_unc_mask_prop["alpha"] * _alpha_coef 
                                                    if "alpha" in _obs_phot_unc_mask_prop else _alpha_coef)
                _h_obs_mask_phot.insert(0, ax.errorbar(_phot_obs_mask["lbda"], _phot_obs_mask["phot"],
                                                       yerr=_phot_obs_mask["phot_err"], **_obs_phot_unc_mask_prop))
        if _h_obs_phot:
            _handles.append(tuple(_h_obs_phot))
            _labels.append("Observed photometry")
        if _h_obs_mask_phot:
            _handles.append(tuple(_h_obs_mask_phot))
            _labels.append("Observed photometry\n(not used for SED fitting)")
        
        # Format
        ax.set_xlabel(r"wavelentgh [$\AA$]", fontsize="large")
        ax.set_ylabel(f"{'magnitude' if unit=='mag' else 'flux'} [{tools.get_unit_label(unit)}]", fontsize="large")
        if set_logx:
            ax.set_xscale("log")
        if set_logy:
            ax.set_yscale("log")
        
        if show_legend:
            ax.legend(_handles, _labels, **show_legend)
        
        if show_filters:
            _filternames = self._get_filternames_(filters, self.obs, return_idx=False)
            if len(_filternames) > 0:
                from sedpy.observate import load_filters
                from . import io
                _filters = np.array(load_filters(io.filters_to_pysed(_filternames)))
                _ymin, _ymax = ax.get_ylim()
                ax.set_ylim(_ymin, _ymax)
                for ii, _f in enumerate(_filters):
                    _w, _t = tools.fix_trans(_f.wavelength, _f.transmission)
                    _t = _t * io._DEFAULT_FILTER_TRANS_COEF.get(_filternames[ii].split(".")[0], 1)
                    if set_logy:
                        _t = 10**(0.25*(np.log10(_ymax/_ymin))) * _t * _ymin + _ymin
                        _y_f = 10**(0.25*(np.log10(_ymax/_ymin))) * 0.03 * _ymin + _ymin
                    else:
                        _t = 0.35 * (_ymax - _ymin) * _t + _ymin
                        _y_f = 0.35 * (_ymax - _ymin) * 0.03 + _ymin
                    ax.text(_f.wave_average, _y_f, _filternames[ii], ha="center", va="bottom", rotation="vertical", fontsize="small")
                    ax.plot(_w, _t, **show_filters)
        
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
