#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas
import time
import sys
import os
from warnings import warn
from pprint import pprint

from . import io
from . import tools

class Prospector():
    """
    This class use the python package 'prospector' to fit the SED.
    """
    
    def __init__(self, **kwargs):
        """
        Initialization.
        Can run 'set_data'.
        
        Options
        -------
        phot : [dict or pandas.DataFrame or None]
            Photometry to be fitted.
            The keys must be on the format instrument.band and instrument.band.err (e.g. sdss.u, sdss.u.err, ps1.g, ps1.g.err, ...).
            An uncertainty is required for every given band.
            Default is None.
        
        spec : [dict or pandas.DataFrame or None]
            Spectrometry to be fitted.
            The given data must include wavelength and spectrum measurement under the keys 'lbda' and 'spec' respectively.
            An uncertainty on spectrum can be given under the key 'spec.err'.
            Default is None.
        
        unit : [string]
            Unit of the given measurements.
            Can be:
                - "Hz": erg/s/cm2/Hz
                - "AA": erg/s/cm2/AA
                - "mgy": maggies
                - "Jy": Jansky
                - "mag": magnitudes
            Default is "Hz".
        
        z : [float or None]
            Redshift to be fixed during SED fitting.
            Default is None.
        
        name : [string or None]
            Studied object name to be stored as attribute.
            Default is None.
        
        
        Returns
        -------
        Void
        """
        if kwargs != {}:
            self.set_data(**kwargs)
    
    def set_data(self, phot=None, spec=None, unit="Hz", z=None, name=None):
        """
        Set up input data as attributes.
        At least one of 'phot' or 'spec' must be given.
        
        Parameters
        ----------
        phot : [dict or pandas.DataFrame or None]
            Photometry to be fitted.
            The keys must be on the format instrument.band and instrument.band.err (e.g. sdss.u, sdss.u.err, ps1.g, ps1.g.err, ...).
            An uncertainty is required for every given band.
            Default is None.
        
        spec : [dict or pandas.DataFrame or None]
            Spectrometry to be fitted.
            The given data must include wavelength and spectrum measurement under the keys 'lbda' and 'spec' respectively.
            The given wavelength must be in AA.
            An uncertainty on spectrum can be given under the key 'spec.err'.
            Default is None.
        
        unit : [string]
            Unit of the given measurements.
            Can be:
                - "Hz": erg/s/cm2/Hz
                - "AA": erg/s/cm2/AA
                - "mgy": maggies
                - "Jy": Jansky
                - "mag": magnitudes
            Default is "Hz".
        
        z : [float or None]
            Redshift to be fixed during SED fitting.
            Default is None.
        
        Options
        -------
        name : [string or None]
            Studied object name to be stored as attribute.
            Default is None.
        
        
        Returns
        -------
        Void
        """
        self._phot_in = io.load_phot(phot=phot, unit=unit)
        self._spec_in = io.load_spec(spec=spec, unit=unit)
        self._z = z
        self._name = name
        if self.has_phot_in():
            self._filters = io.keys_to_filters(self.phot_in.keys())
    
    def _set_data_from_obs(self, obs, warnings=True):
        """
        Set up attributes coming from an 'obs' dictionary.
        
        Parameters
        ----------
        obs : [dictionary]
            Prospector compatible 'obs' dictionary.
        
        Options
        -------
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        
        Returns
        -------
        Void
        """
        for _attr, _obs_key in {"_filters":"filternames", "_z":"zspec", "_name":"name"}.items():
            try:
                if _attr == "_filters":
                    self._filters = io.pysed_to_filters(self.obs["filternames"])
                else:
                    self.__dict__[_attr] = self.obs[_obs_key]
            except(KeyError):
                self.__dict__[_attr] = None
                if warnings:
                    warn(f"Cannot set the '{_attr}' attribute as there is no '{_obs_key}' key in the 'obs' dictionary.")
        
        try:
            self._phot_in = {_filt:self.obs["maggies"][ii] for ii, _filt in enumerate(self.filters)}
            self._phot_in.update({_filt+".err":self.obs["maggies_unc"][ii] for ii, _filt in enumerate(self.filters)})
        except(KeyError):
            self._phot_in = None
            if warnings:
                warn("Cannot set 'phot_in' attribute either because there is no 'maggies' or no 'maggies_unc' in the 'obs' dictionary.")
        except(TypeError):
            self._phot_in = None
            if warnings:
                warn("Cannot set 'phot_in' attribute because there is no 'filters' attribute.")
        
        try:
            self._spec_in = {"lbda":self.obs["wavelength"], "spec":self.obs["spectrum"], "spec.err":self.obs["unc"]}
        except(KeyError):
            self._spec_in = None
            if warnings:
                warn("Cannot set '_spec_in' attribute because there is no 'wavelength', 'spectrum' and 'unc' in the 'obs' dictionary.")
        if self.spec_in["lbda"] is None and self.spec_in["spec"] is None and self.spec_in["spec.err"]:
            self._spec_in = None
    
    #=================#
    #   Build 'obs'   #
    def build_obs(self, obs=None, phot_mask=None, spec_mask=None, verbose=False, warnings=True, set_data=False):
        """
        Build a dictionary containing observations in a prospector compatible format.
        
        Options
        -------
        obs : [dict or None]
            Can load an already existing prospector compatible 'obs' dictionary.
            Automatically change the attributes to the given 'obs' dictionary.
            Default is None.
        
        phot_mask : [list(string) or None]
            Give a list of filters you want the SED fitting to be run on.
            The other filter measurements will be masked.
            If None, no mask is applied.
            Default is None.
        
        spec_mask : [tuple(float or None, float or None) or np.array or None]
            Give a tuple of two wavelength in AA (lower_limit, upper_limit) to run 
            the SED fitting on the input spectrum between the given limits.
            The rest of the spectrum is masked.
            You can give None for a limit to keep the whole spectrum in that direction.
            You also can directly give your own mask array.
            If None, the whole spectrum is fitted.
            Default is None.
        
        verbose : [bool]
            If True, print the built observation dictionary.
            Default is False.
        
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        set_data : [bool]
            If 'obs' is not None, set to True if you want to extract and set attributes from the given 'obs' dictionary.
            Default is False.
        
        
        Returns
        -------
        Void
        """
        from prospect.utils.obsutils import fix_obs
        if obs is not None:
            self._obs = fix_obs(obs)
            if set_data:
                self._set_data_from_obs(obs=self.obs, warnings=warnings)
            return
        
        from sedpy.observate import load_filters
        self._obs = {"filters":load_filters(io.filters_to_pysed(self.filters)) if self.has_phot_in() else None,
                     "zspec":self.z,
                     "name":self.name}
        ### Photometry ###
        if self.has_phot_in():
            self._obs.update({"maggies":np.array([self.phot_in[_filt] for _filt in self.filters]),
                              "maggies_unc":np.array([self.phot_in[_filt+".err"] for _filt in self.filters])})
            if phot_mask is not None:
                self._obs["phot_mask"] = np.array([_f in phot_mask for _f in self.filters])
        ### Spectrometry ###
        self._obs["wavelength"] = None
        if self.has_spec_in():
            self._obs.update({"wavelength":np.array(self.spec_in["lbda"]),
                              "spectrum":np.array(self.spec_in["spec"]),
                              "unc":np.array(self.spec_in["spec.err"]) if "spec.err" in self.spec_in.keys() else None})
            if spec_mask is not None:
                if type(spec_mask) in [list, tuple, np.ndarray] and len(spec_mask) == 2:
                    _lim_low, _lim_up = spec_mask
                    _spec_mask = np.ones(len(self.obs["wavelength"]), dtype = bool)
                    if _lim_low is not None:
                        _spec_mask &= self.obs["wavelength"] > _lim_low
                    if _lim_up is not None:
                        _spec_mask &= self.obs["wavelength"] < _lim_up
                    self._obs["mask"] = _spec_mask
                elif type(spec_mask) in [list, tuple, np.ndarray] and len(spec_mask)==len(self.spec_in["lbda"]):
                    self._obs["mask"] = np.array(spec_mask)
                else:
                    if warnings:
                        warn(f"Cannot apply a mask on spectrum because the 'spec_mask' you give is not compliant ({spec_mask})")
        self._obs = fix_obs(self._obs)
        if verbose:
            print(tools.get_box_title(title="Built obs", box="\n#=#\n#   {}   #\n#=#\n"))
            pprint(self.obs)
    
    #===================#
    #   Build 'model'   #
    def build_model(self, model=None, templates=None, verbose=False, describe=False):
        """
        Build the model parameters to fit on measurements.
        
        Parameters
        ----------
        templates : [string or list(string) or None]
            Prospector prepackaged parameter set(s) to load.
            Can be one template, or a list.
            Default is None.
        
        Options
        -------
        model :  [prospect.models.sedmodel.SedModel or dict or list or None]
            Can load an already existing prospector compatible 'model' SedModel.
            Can update a SedModel compatible 'model' dictionary before to create the SedModel on it.
            Default is None.
        
        verbose : [bool]
            If True, print the built model.
            Default is False.
        
        describe : [bool]
            If True, print the description for any used SED model template and not native prior.
            Default is False.
        
        
        Returns
        -------
        Void
        """
        from prospect.models.sedmodel import SedModel
        from prospect.models.templates import TemplateLibrary
        
        if type(model) is SedModel:
            self._model = model
            return
        
        if describe:
            self.describe_templates(templates=templates)
        
        _model = {} if model is None else model
        if isinstance(_model, list):
            _model = {_p["name"]:_p for _p in _model}
        for _t in np.atleast_1d(templates):
            if _t in TemplateLibrary._descriptions.keys():
                _model.update(TemplateLibrary[_t])
        _describe_priors = []
        for _p, _pv in _model.items():
            if "prior" in _pv.keys() and type(_pv["prior"]) == dict:
                _describe_priors.append(_pv["prior"])
                _pv["prior"] = self.build_prior(_pv["prior"])
        if describe and len(_describe_priors) > 0:
            self.describe_priors(_describe_priors)
        
        if self.has_z():
            _model["zred"].update({"init":self.obs["zspec"], "isfree":False})
        else:
            _model["zred"].update({"isfree":True})
        
        if verbose:
            print(tools.get_box_title(title="Built model", box="\n#=#\n#   {}   #\n#=#\n"))
            pprint(_model)
        
        self._model = SedModel(_model)
    
    def modify_model(self, changes=None, removings=None, verbose=False, warnings=True, describe=False):
        """
        Apply changes on the loaded model.
        
        Parameters
        ----------
        changes : [dict or None]
            Dictionary containing the desired changes for the desired parameters.
            As an example, changes={"zred"={"init"=2.}}) will change the initial value of the 'zred' parameter, without changing anything else.
            Default is None.
        
        removings : [string or list(string) or None]
            List of every parameters you want to remove from the model.
            Default is None.
        
        Options
        -------
        verbose : [bool]
            If True, print the SED model before and after modifications.
            Default is False.
        
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        describe : [bool]
            If True, print the description for any used not native prior.
            Default is False.
        
        
        Returns
        -------
        Void
        """
        from prospect.models.sedmodel import SedModel
        
        _model = self.model.config_dict
        if verbose:
            print(tools.get_box_title(title="Previous model", box="\n#=#\n#   {}   #\n#=#\n"))
            pprint(_model)
        
        #Changes
        _describe_priors = []
        if changes is not None:
            _model, _describe_priors = self._model_changes_(_model, changes)
        if describe and len(_describe_priors) > 0:
            self.describe_priors(_describe_priors)
        
        #Removing
        if removings is not None:
            _model = self._model_removings_(_model, removings, warnings)
        
        if verbose:
            print(tools.get_box_title(title="New model", box="\n\n\n#=#\n#   {}   #\n#=#\n"))
            pprint(_model)
        
        self._model = SedModel(_model)
    
    @staticmethod
    def _model_changes_(model, changes, warnings=True):
        """
        Apply changes on model parameters.
        Return the modified model dictionary and the list of priors in 'changes' (useful for description).
        
        Parameters
        ----------
        model : [dict]
            Model dictionary on which to apply modifications.
        
        changes : [dict]
            Dictionary containing the desired changes for the desired parameters.
            As an example, changes={"zred"={"init"=2.}}) will change the initial value of the 'zred' parameter, without changing anything else.
        
        Options
        -------
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        
        Returns
        -------
        dict, list
        """
        _model, _changes = model.copy(), changes.copy()
        _describe_priors = []
        for _p, _pv in _changes.items():
            if _p not in _model.keys():
                if warnings:
                    warn(f"'{_p}' is not included in model parameters. Adding it...")
                #_changes.pop(_p)
                _model[_p] = _pv
            for _k, _kv in _pv.items():
                if _k not in _model[_p].keys():
                    if warnings:
                        warn(f"'{_k}' is not included in '{_p}' model parameters. Adding it...")
                    #_pv.pop(_k)
                    _model[_p][_k] = _kv
                if _k == "prior" and type(_kv) == dict:
                    _describe_priors.append(_kv)
                    _pv[_k] = Prospector.build_prior(_kv)
            _model[_p].update(_pv)
        return _model, _describe_priors
    
    @staticmethod
    def _model_removings_(model, removings, warnings):
        """
        Apply removings on model parameters.
        Return the modified model dictionary.
        
        Parameters
        ----------
        model : [dict]
            Model dictionary on which to apply modifications.
        
        removings : [list]
            List of every parameters you want to remove from the model.
        
        Options
        -------
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        
        Returns
        -------
        dict
        """
        _model = model.copy()
        for _t in np.atleast_1d(removings):
            if _t in _model.keys():
                _model.pop(_t)
            else:
                if warnings:
                    warn(f"Cannot remove '{_t}' as it doesn't exist in the current model.")
        return _model
    
    @staticmethod
    def build_prior(priors, verbose=False):
        """
        Build and return a prospector SedModel compatible prior.
        
        Parameters
        ----------
        prior : [dict]
            Dictionary containing every required parameters to build a prior object.
        
        Options
        -------
        verbose : [bool]
            If True, print information about the given prior to build.
            Default is False.
            
            
        Returns
        -------
        prospect.models.priors.[chosen prior]
        """
        from prospect.models import priors as p_priors
        if verbose:
            Prospector.describe_priors(priors=priors)
        _priors = priors.copy()
        _name = _priors.pop("name")
        return eval(f"p_priors.{_name}")(**_priors)
    
    #=================#
    #   Build 'sps'   #
    def build_sps(self, sps=None, zcontinuous=1):
        """
        Create the appropriate sps.
        
        Parameters
        ----------
        zcontinuous : [float]
            python-fsps parameter controlling how metallicity interpolation of the SSPs is acheived :
                - 0: use discrete indices (controlled by parameter "zmet")
                - 1: linearly interpolate in log Z/Z_\sun to the target metallicity (the parameter "logzsol")
                - 2: convolve with a metallicity distribution function at each age (the MDF is controlled by the parameter "pmetals")
            A value of '1' is recommended.
            Default is 1.
        
        Options
        -------
        sps : [sps instance]
            If not None, this will set the 'sps' attribute with the given sps.
            Default is None.
        
        
        Returns
        -------
        Void
        """
        if sps is not None:
            self._sps = sps
            return
        
        self.run_params["zcontinuous"] = zcontinuous
        if not self.has_model():
            raise AttributeError("You must run 'build_model first to be able automatically build the corresponding SPS.")
        if "agebins" not in self.model.params.keys():#"mass" not in self.model.params.keys():
            from prospect.sources import CSPSpecBasis
            self._sps = CSPSpecBasis(zcontinuous=zcontinuous)
        else:
            from prospect.sources import FastStepBasis
            self._sps = FastStepBasis(zcontinuous=zcontinuous)
    
    #=================#
    #   SED fitting   #
    def run_fit(self, which="dynesty", run_params={}, savefile=None, set_chains=True, verbose=False):
        """
        Run the SED fitting.
        
        Parameters
        ----------
        which : [bool]
            SED fitting method choice between:
                - "dynesty"
                - "emcee"
                - "optimize"
            Default is "dynesty".
        
        Options
        -------
        run_params : [dict]
            Dictionary containing running parameters associated to the chosen fitting method.
            You can look at more information for this input with the function 'describe_run_parameters'.
            Default is {} (meaning that the default values for every parameters are applied).
        
        savefile : [string or None]
            If a file name is given, the fitting results are saved in this file.
            Must end with ".h5".
            Default is None.
        
        set_chains : [bool]
            If True, set the fitted parameter walkers as attributes.
            !!! Warning !!! Only works with "dynesty" and "emcee" SED fitting.
            Default is True.
        
        verbose : [bool]
            If True, print out the time taken by the SED fitting.
            Default is False.
        
        
        Returns
        -------
        Void
        """
        from prospect.fitting import lnprobfn
        from prospect.fitting import fit_model
        
        #Running parameters
        self._run_params = {"optimize":False, "emcee":False, "dynesty":False, "zcontinuous":self.run_params["zcontinuous"]}
        self.run_params[which] = True
        self.run_params.update(run_params)
        
        #Fit
        self._fit_output = fit_model(obs=self.obs, model=self.model, sps=self.sps, 
                                     lnprobfn=lnprobfn, verbose=verbose, **self.run_params)
        
        if savefile is not None:
            self.write_h5(savefile=savefile)
        
        if verbose:
            print("Done '{}' in {:.0f}s.".format(which, self.fit_output["sampling"][1]))
        
        #Load chains
        if set_chains:
            self.set_chains(data=self.fit_output, start=None)
    
    def set_chains(self, data=None, start=None):
        """
        Load chains from the fit output.
        
        Parameters
        ----------
        data : [dict or tuple or None]
            Give either 'self.fit_output' or the results saved in a .h5 file (run 'Prospector.read_h5').
            If None, automatically load chains from the fit output.
            Default is None.
        
        Options
        -------
        start : [int or None]
            Chain index from which starting to save the walkers.
            If None, the start depends on the SED fitter :
                - "emcee" --> 0
                - "dynesty" --> index when "samples_id" (see dynesty fit output) exceeds 100.
                                (seems to correspond to the dynamic phase of dynesty)
            Default is None.
        
        
        Returns
        -------
        Void
        """
        if data is None:
            data = self.fit_output
        try:
            _sampling = data["sampling"][0]
            _chains = _sampling.chain if self.run_params["emcee"] else _sampling["samples"]
        except KeyError:
            _sampling = data
            _chains = _sampling["chain"]
        except TypeError:
            _sampling = data[0]
            _chains = _sampling["chain"]
        
        _start = (0 if self.run_params["emcee"] else np.argmax(_sampling["samples_id"]>100)) if start is None else start
        if len(_chains.shape) == 2:
            _chains = _chains.copy()[_start:]
        elif len(_chains.shape) == 3:
            _chains = np.concatenate(_chains.copy()[:, _start:, :])
        else:
            raise ValueError(f"There is an issue with the chains!!!\n\n{_chains}")
        
        self._chains = np.array(_chains)
        self._param_chains = {_p:np.array([jj[ii] for jj in self.chains]) for ii, _p in enumerate(self.theta_labels)}
    
    def load_spectrum(self, filename=None, warnings=True, **kwargs):
        """
        Load a spectrum object and set it as attribute.
        
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
        from .spectrum import ProspectorSpectrum
        self._spectrum = ProspectorSpectrum(chains=self.chains, model=self.model, obs=self.obs, sps=self.sps,
                                            filename=filename, warnings=warnings, **kwargs)
    
    def load_phys_params(self):
        """
        Load a physical parameters instance and set it as attribute.
        
        
        Returns
        -------
        Void
        """
        from .phys_params import ProspectorPhysParams
        self._phys_params = ProspectorPhysParams(chains=self.chains, model=self.model)
    
    def write_h5(self, savefile):
        """
        Save the fitting results and every fitter inputs (apart from the 'sps') in a given file.
        
        Parameters
        ----------
        savefile : [string]
            File name in which to save the results.
        
        
        Returns
        -------
        Void
        """
        import h5py
        import pickle
        
        from prospect.io import write_results as writer
        writer.write_hdf5(hfile=savefile, run_params=self.run_params, model=self.model, obs=self.obs,
                          sampler=self.fit_output["sampling"][0], optimize_result_list=self.fit_output["optimization"][0],
                          tsample=self.fit_output["sampling"][1], toptimize=self.fit_output["optimization"][1])
        with h5py.File(savefile, "a") as _h5f:
            _h5f.create_dataset("model", data=np.void(pickle.dumps(self.model)))
            _h5f.create_dataset("sps", data=np.void(pickle.dumps(self.sps)))
            _h5f.flush()
        
    @classmethod
    def from_h5(cls, filename, build_sps=True, warnings=True):
        """
        Build and return a Prospector object from a .h5 file.
        
        Parameters
        ----------
        filename : [string]
            File name in which are saved the results to be read and loaded.
        
        Options
        -------
        build_sps : [bool]
            If True, build the SPS instance based on the built model and "zcontinuous" running parameter saved in the file.
            Default is True.
        
        warnings : [bool]
            If True, allow the warnings to be printed.
            Default is True.
        
        
        Returns
        -------
        Prospector
        """
        import h5py
        import pickle
        
        if not filename.endswith(".h5"):
            raise ValueError(f"You must give a .h5 file (your input: {filename}).")
        
        _this = cls()
        
        _this._h5_results, _obs, _ = _this.read_h5(filename, dangerous=False)
        _this.build_obs(obs=_obs, verbose=False, warnings=warnings, set_data=True)
        _this._run_params = _this.h5_results["run_params"]
        
        with h5py.File(filename, "r") as _h5f:
            # Load model
            try:
                _this.build_model(model=pickle.loads(_h5f["model"][()]), verbose=False)
            except(KeyError):
                try:
                    _this.build_model(model=_this._read_model_params(_this.h5_results["model_params"]), verbose=False)
                    if warnings:
                        warn("The model has been built with dependance functions, if any, with their default inputs.")
                except:
                    if warnings:
                        warn("Cannot build the 'model' for this object as it doesn't exist in the .h5 file and ",+
                             "there is an error building it from the saved 'model_params', you must build it yourself.")
            # Load SPS
            #try:
            #    _this.build_sps(sps=pickle.loads(_h5f["sps"][()]))
            #except(KeyError):
            if _this.has_model() and build_sps:
                _this.build_sps(zcontinuous=_this.run_params["zcontinuous"])
            elif not _this.has_model() and build_sps:
                if warnings:
                    warn("Cannot build the SPS as it doesn't exist in the .h5 file and the 'model' is not built. "+
                         "You will have to build it yourself.")
        
        # Set chains
        _this.set_chains(data=_this.h5_results, start=None)
        
        return _this
    
    @staticmethod
    def read_h5(filename, **kwargs):
        """
        Read and return the prospector fit results saved in a given .h5 file.
        
        Parameters
        ----------
        filename : [string]
            File name in which are saved the results to be read and returned.
        
        
        Returns
        -------
        dict, dict, SedModel
        """
        import prospect.io.read_results as reader
        return reader.results_from(filename, **kwargs)
    
    
    @staticmethod
    def _read_model_params(model_params, warnings=True):
        """
        Return a 'build_model' compatible dictionary given by the 'model_params' saved in the prospector .h5 files.
        
        Parameters
        ----------
        model_params : [list(dict)]
            List containing the prospector's model parameters.
        
        Options
        -------
        warnings : [bool]
            If True, allow the warnings to be printed.
            Default is True.
        
        
        Returns
        -------
        dict
        """
        import pickle
        import prospect.models.transforms as ptrans
        _model = {_p["name"]:{k:pickle.loads(v) if k=="prior" else v for k, v in _p.items()} for _p in model_params}
        for _p, _pv in _model.items():
            _pv.pop("name")
            if "depends_on" in _pv.keys():
                if _pv["depends_on"][1] == "prospect.models.transforms":
                    _pv["depends_on"] = eval(f"ptrans.{_pv['depends_on'][0]}")()
                else:
                    _pv.pop("depends_on")
                    if warnings:
                        warn("Cannot build the dependance as it is not comming from 'prospect.models.transforms' package.")
        return _model
    
    
    #---------------#
    #   Plottings   #
    #---------------#
    def _get_plot_data_(self):
        """
        Return a dictionary compatible to plot fit results using prospector methods.
        
        
        Returns
        -------
        dict
        """
        _data = {"model":self.model,
                 "chain":np.array(self.chains),
                 "lnprobability":None}
        return _data
        
    def show_walkers(self, savefile=None, **kwargs):
        """
        Plot the fitted parameters walkers.
        Return the figure.
        
        Options
        -------
        savefile : [string or None]
            Give a directory to save the figure.
            Default is None (not saved).
        
        showpars : [list(string) or None]
            List of the parameters to show.
            If None, plot every fitted parameters (listed in 'self.theta_labels').
            Defaults is None.

        start : [int]
            Integer giving the iteration number from which to start plotting.
            Default is 0.

        chains : [np.array or slice]
            If results are from an ensemble sampler, setting 'chain' to an integer
            array of walker indices will cause only those walkers to be used in
            generating the plot.
            Useful for to keep the plot from getting too cluttered.
            Default is slice(None, None, None).
        
        truths : [list(float) or None]
            List of truth values for the chosen parameters to add them on the plot.
            Default is None.
    
        **plot_kwargs:
            Extra keywords are passed to the 'matplotlib.axes._subplots.AxesSubplot.plot()' method.
        
        Returns
        -------
        matplotlib.Figure
        """
        import prospect.io.read_results as reader
        _data = self._get_plot_data_()
        _fig = reader.traceplot(_data, **kwargs)
        if savefile is not None:
            _fig.savefig(savefile)
        return _fig
    
    def show_corner(self, savefile=None, **kwargs):
        """
        Plot a corner plot of the fitted parameters.
        Return the figure.
        
        Options
        -------
        savefile : [string or None]
            Give a directory to save the figure.
            Default is None (not saved).

        showpars : [list(string) or None]
            List of the parameters to show.
            If None, plot every fitted parameters (listed in 'self.theta_labels').
            Defaults is None.
            
        truths : [list(float) or None]
            List of truth values for the chosen parameters to add them on the plot.
            Default is None.

        start : [int]
            Integer giving the iteration number from which to start plotting.
            Default is 0.
        
        thin : [int]
            The thinning of each chain to perform when drawing samples to plot.
            Default is 1.

        chains : [np.array or slice]
            If results are from an ensemble sampler, setting 'chain' to an integer
            array of walker indices will cause only those walkers to be used in
            generating the plot.
            Useful for to keep the plot from getting too cluttered.
            Default is slice(None, None, None).
        
        logify : [list(string)]
            A list of parameter names to plot in log scale.
            Default is ["mass", "tau"].
    
        **kwargs:
            Remaining keywords are passed to the 'corner' plotting package.
        
        Returns
        -------
        matplotlib.Figure
        """
        import prospect.io.read_results as reader
        _data = self._get_plot_data_()
        _fig = reader.subcorner(_data, **kwargs)
        if savefile is not None:
            _fig.savefig(savefile)
        return _fig
    
    
    
    #------------------#
    #   Descriptions   #
    #------------------#
    @staticmethod
    def describe_priors(priors=None):
        """
        Describe the prospector available priors.
        
        Parameters
        ----------
        priors : [string or list(string) or None]
            Prior choice.s for which you ask for description.
            Can be one prior, or a list.
            If "*" or "all" is given, every available priors will be described.
            Default is None.
        
        
        Returns
        -------
        Void
        """
        from prospect.models import priors as p_priors
        print(tools.get_box_title(title="Prior descriptions", box="\n#=#\n#   {}   #\n#=#\n"))
        _prior_list = p_priors.__all__.copy()
        _prior_list.remove("Prior")
        print(f"Available priors: {', '.join(_prior_list)}.\n")
        if priors is not None:
            print("(Required arguments must be included in addition with the 'name' argument in a "+
                  "dictionary to correctly build the prior, \n"+
                  " for example, try to describe 'priors={'name':'Normal', 'mean':0., 'sigma':1.}').\n")
            if priors in ["*", "all"]:
                priors = _prior_list
            _priors = np.atleast_1d(priors)
            for _p in _priors:
                _name = _p.pop("name") if type(_p) == dict else _p
                if _name not in _prior_list:
                    warn(f"'{_name}' is not an available prior.")
                    continue
                print(tools.get_box_title(title=_name, box="\n+-+\n|   {}   |\n+-+\n"))
                _pdoc = eval("p_priors."+_name).__doc__
                _pdoc = _pdoc.replace(":param", "-")
                if type(_p) == dict:
                    for _pp in eval("p_priors."+_name).prior_params:
                        _pdoc = _pdoc.replace(_pp+":", _pp+f": (your input is: {_p[_pp]})" if _pp in _p.keys()
                                                  else _pp+": (!!!WARNING!!! No input!)")
                print(_pdoc)
    
    @staticmethod
    def describe_templates(templates=None):
        """
        Describe the prospector prepackaged parameter sets.
        
        Parameters
        ----------
        templates : [string or list(string) or None]
            Template choice.s for which you ask for description.
            Can be one template, or a list.
            If "*" or "all" is given, every available templates will be described.
            Default is None.
        
        
        Returns
        -------
        Void
        """
        from prospect.models.templates import TemplateLibrary
        
        print(tools.get_box_title(title="Availbale templates", box="\n#=#\n#   {}   #\n#=#\n"))
        TemplateLibrary.show_contents()
        
        if templates in ["*", "all"]:
            templates = list(TemplateLibrary._descriptions.keys())
        if templates is not None and not np.all([_t not in TemplateLibrary._descriptions.keys() for _t in np.atleast_1d(templates)]):
            print(tools.get_box_title(title="Template descriptions", box="\n\n\n#=#\n#   {}   #\n#=#\n"))
            for _t in np.atleast_1d(templates):
                if _t in TemplateLibrary._descriptions.keys():
                    print(tools.get_box_title(title=_t, box="+-+\n|   {}   |\n+-+"))
                    TemplateLibrary.describe(_t)
                    print("\n")
    
    @staticmethod
    def describe_run_parameters(which="*"):
        """
        Describe the fitter meta-parameters.
        
        Parameters
        ----------
        which : [string or list(string)]
            Fitter choice between:
                - "optimize"
                - "emcee"
                - "dynesty"
            Can be a list of them.
            "*" or "all" automaticaly describe the three of them.
            Default is "*".
        
        
        Returns
        -------
        Void
        """
        from prospect.fitting.fitting import run_emcee, run_dynesty
        from prospect.fitting.fitting import run_minimize as run_optimize
        _fitters = ["optimize", "emcee", "dynesty"]
        _rejected_params = ["obs", "sps", "model", "noise", "lnprobfn", "hfile"]
        _list_fitter = np.atleast_1d(which) if which not in ["*", "all"] else _fitters
        print(tools.get_box_title(title="Running parameters description", box="\n#=#\n#   {}   #\n#=#\n"))
        for _f in _list_fitter:
            if _f not in _fitters:
                warn("'{}' is not a know fitter.".format(_f))
                continue
            print(tools.get_box_title(title=_f, box="\n+-+\n|   {}   |\n+-+\n"))
            _doc = eval("run_"+_f).__doc__.split(":param ")
            _doc = [_d for _d in _doc if _d.split(":")[0] not in _rejected_params and len(_d.split(":")[0].split(" "))==1]
            _doc[-1] = _doc[-1].split("Returns")[0]
            print("".join(["    "]+_doc))
            if _f == "dynesty":
                print("    There are more informations on these parameters at:\n"+
                      "        https://dynesty.readthedocs.io/en/latest/api.html#dynesty.dynesty.DynamicNestedSampler\n")
    
    
    
    #----------------#
    #   Properties   #
    #----------------#
    @property
    def phot_in(self):
        """ Input photometry """
        if not hasattr(self,"_phot_in"):
            self._phot_in = None
        return self._phot_in
    
    def has_phot_in(self):
        """ Test that phot_in is not void """
        return self.phot_in is not None
    
    @property
    def spec_in(self):
        """ Input spectrometry """
        if not hasattr(self,"_spec_in"):
            self._spec_in = None
        return self._spec_in
    
    def has_spec_in(self):
        """ Test that spec_in is not void """
        return self.spec_in is not None
    
    @property
    def z(self):
        """ Input redshift """
        return self._z
    
    def has_z(self):
        """ Test that z is not void """
        return self.z is not None
    
    @property
    def name(self):
        """ Target's name """
        return self._name
    
    @property
    def filters(self):
        """ List of filters of the input photometry """
        if not hasattr(self,"_filters"):
            self._filters = None
        return self._filters
    
    def has_filters(self):
        """ Test that filters is not void """
        return self.filters is not None
    
    ### prospector ###
    @property
    def obs(self):
        """ Dictionary containing observations in a prospector compatible format """
        if not hasattr(self,"_obs"):
            self._obs = None
        return self._obs
    
    def has_obs(self):
        """ Test that 'obs' is not void """
        return self.obs is not None
    
    @property
    def sps(self):
        """ SPS object """
        if not hasattr(self,"_sps"):
            self._sps = None
        return self._sps
    
    def has_sps(self):
        """ Test that 'sps' is not void """
        return self.sps is not None
    
    @property
    def model(self):
        """ Prospector's SedModel object """
        if not hasattr(self,"_model"):
            self._model = None
        return self._model
    
    def has_model(self):
        """ Test that 'model' is not void """
        return self.model is not None
    
    @property
    def theta_labels(self):
        """ List of the fitted parameters """
        return self.model.theta_labels()
    
    @property
    def run_params(self):
        """ Running parameters (dictionary) """
        if not hasattr(self,"_run_params"):
            self._run_params = {}
        return self._run_params
    
    @property
    def fit_output(self):
        """ Fitter output (dictionary with two keys, 'optimization' and 'sampling') """
        if not hasattr(self,"_fit_output"):
            self._fit_output = None
        return self._fit_output
    
    def has_fit_output(self):
        """ Test that 'fit_output' is not void """
        return self.fit_output is not None
    
    @property
    def h5_results(self):
        """ Dictionary containg the fit results, loaded from .h5 file """
        if not hasattr(self,"_h5_results"):
            self._h5_results = None
        return self._h5_results
    
    def has_h5_results(self):
        """ Test that 'h5_results' is not void """
        return self.h5_results is not None
    
    @property
    def chains(self):
        """ List of the fitted parameters chains """
        if not hasattr(self,"_chains"):
            self._chains = None
        return self._chains
    
    @property
    def len_chains(self):
        """ Length of the chains --> number of steps for the MCMC """
        return len(self.chains)
    
    @property
    def param_chains(self):
        """ Dictionary containing the fitted parameters chains """
        if not hasattr(self,"_param_chains"):
            self._param_chains = None
        return self._param_chains
    
    @property
    def fitted_params(self):
        """ Dictionary containing the fitted paramters (median, -sigma, +sigma) """
        _perc = {_p:np.percentile(_chain, [16, 50, 84]) for _p, _chain in self.param_chains.items()}
        return {_p:(_pv[1], _pv[1]-_pv[0], _pv[2]-_pv[1]) for _p, _pv in _perc.items()}
    
    @property
    def spectrum(self):
        """ spectra estimated from run() ; LePhareSpectrumCollection object """
        if not hasattr(self, "_spectrum"):
            if self.has_fit_output():
                self.set_chains(data=self.fit_output, start=None)
                self.load_spectrum()
            elif self.has_h5_results():
                self.set_chains(data=self.h5_results, start=None)
                self.load_spectrum()
            else:
                raise AttributeError("You did not run the SED fit ('self.run_fit').")
        return self._spectrum
    
    @property
    def phys_params(self):
        """ ProspectorPhysParams instance """
        if not hasattr(self, "_phys_params"):
            if self.has_fit_output():
                self.set_chains(data=self.fit_output, start=None)
                self.load_phys_params()
            elif self.has_h5_results():
                self.set_chains(data=self.h5_results, start=None)
                self.load_phys_params()
            else:
                raise AttributeError("You did not run the SED fit ('self.run_fit').")
        return self._phys_params
    
    
