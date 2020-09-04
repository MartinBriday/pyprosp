
import numpy as np
import pandas
import time
import sys
import os
from warnings import warn
from pprint import pprint

from . import io

class Prospector():
    """
    
    """
    
    def __init__(self, **kwargs):
        """
        
        """
        if kwargs != {}:
            self.set_data(**kwargs)
    
    def set_data(self, phot=None, spec=None, unit="Hz", z=None, name=None):
        """
        
        """
        self._phot_in = io.load_phot(phot=phot, unit=unit)
        self._spec_in = io.load_spec(spec=spec, unit=unit)
        self._z = z
        self._name = name
        if self.has_phot_in():
            self._filters = io.keys_to_filters(self.phot_in.keys())
    
    def build_obs(self, obs=None, verbose=False):
        """
        Build a dictionary containing observations in a prospector compatible format.
        
        Options
        -------
        obs : [dict]
            Can load an already existing prospector compatible 'obs' dictionary.
            Default is None.
        
        verbose : [bool]
            If True, print the built observation dictionary.
            Default is False.
        
        
        Returns
        -------
        Void
        """
        if obs is not None:
            self._obs = obs
            return
        
        from sedpy.observate import load_filters
        from prospect.utils.obsutils import fix_obs
        self._obs = {"filters":load_filters(io.filters_to_pysed(self._filters)) if self.has_phot_in else None,
                     "zspec":self.z}
        ### Photometry ###
        if self.has_phot_in():
            self._obs.update({"maggies":np.array([self.phot_in[_filt] for _filt in self.filters]),
                              "maggies_unc":np.array([self.phot_in[_filt+".err"] for _filt in self.filters])})
        ### Spectrometry ###
        self._obs["wavelength"] = None
        if self.has_spec_in():
            self._obs.update({"wavelength":np.array(self.spec_in["lbda"]),
                              "spectrum":np.array(self.spec_in["flux"]),
                              "unc":np.array(self.spec_in["flux.err"]) if "flux.err" in self.spec_in.keys() else None})
        self._obs = fix_obs(self._obs)
        if verbose:
            pprint(self.obs)
        
    def build_model(self, model=None, templates=None, verbose=False, describe=False):
        """
        Build the model parameters to fit on measurements.
        
        Parameters
        ----------
        model :  [prospect.models.sedmodel.SedModel or dict]
            Can load an already existing prospector compatible 'model' SedModel.
            Can update a SedModel compatible 'model' dictionary before to create the SedModel on it.
            Default is None.
        
        templates : [string or list(string) or None]
            Prospector prepackaged parameter set(s) to load.
            Can be one template, or a list.
            Default is None.
        
        Options
        -------
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
        
        if verbose:
            print("\n# =============== #\n#   Built model   #\n# =============== #\n")
            pprint(_model)
        
        self._model = SedModel(_model)
    
    def modify_model(self, changes=None, removing=None, verbose=False, describe=False):
        """
        Apply changes on the loaded model.
        
        Parameters
        ----------
        changes : [dict or None]
            Dictionary containing the desired changes for the desired parameters.
            As an example, changes={"zred"={"init"=2.}}) will change the initial value of the 'zred' parameter, without changing anything else.
            Default is None.
        
        removing : [string or list(string) or None]
            Fill this input with every parameters you want to remove from the model.
            Default is None.
        
        Options
        -------
        verbose : [bool]
            If True, print the SED model before and after modifications.
            Default is False.
        
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
            print("\n# ================== #\n#   Previous model   #\n# ================== #\n")
            pprint(_model)
        
        #Changes
        _describe_priors = []
        if changes is not None:
            _changes = changes.copy()
            for _p, _pv in _changes.items():
                if _p not in _model.keys():
                     warn("'{}' is not included in model parameters.".format(_p))
                     _pv.pop(_p)
                for _k, _kv in _pv.items():
                    if _k not in _model[_p].keys():
                        warn("'{}' is not included in '{}' model parameters.".format(_k, _p))
                        _kv.pop(_k)
                    if _k == "prior" and type(_kv) == dict:
                        _describe_priors.append(_kv)
                        _kv = self.build_prior(_kv)
                _model[_p].update(_pv)
        if describe and len(_describe_priors) > 0:
            self.describe_priors(_describe_priors)
        
        #Removing
        if removing is not None:
            for _t in np.atleast_1d(removing):
                if _t in _model.keys():
                    _model.pop(_t)
                else:
                    warn("Cannot remove '{}' as it doesn't exist in the current model.".format(_t))
        
        if verbose:
            print("\n\n\n# ============= #\n#   New model   #\n# ============= #\n")
            pprint(_model)
        
        self._model = SedModel(_model)
    
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
            self.describe_priors(priors=priors)
        _priors = priors.copy()
        _name = _priors.pop("name")
        return eval("p_priors.{}".format(_name))(**_priors)
    
    @staticmethod
    def describe_priors(priors=None):
        """
        
        """
        from prospect.models import priors as p_priors
        print("\n# ====================== #\n#   Prior descriptions   #\n# ====================== #\n")
        _prior_list = p_priors.__all__.copy()
        _prior_list.remove("Prior")
        print("Available priors: {}.\n".format(", ".join(_prior_list)))
        if priors is not None:
            print("(Required parameters must contained in addition with a 'name' parameter in a dictionary to correctly build the prior, \n"+
                  " for example, try to describe 'priors={'name':'Normal', 'mean':0., 'sigma':1.}').\n")
            if priors in ["*", "all"]:
                priors = _prior_list
            _priors = np.atleast_1d(priors)
            for _p in _priors:
                _name = _p.pop("name") if type(_p) == dict else _p
                if _name not in _prior_list:
                    warn("'{}' is not an available prior.".format(_name))
                    continue
                dash_string = "".join(["-"]*(len(_name)+6))
                print("+{}+\n|   {}   |\n+{}+".format(dash_string, _name, dash_string))
                _pdoc = eval("p_priors."+_name).__doc__
                _pdoc = _pdoc.replace(":param", "-")
                if type(_p) == dict:
                    for _pp in eval("p_priors."+_name).prior_params:
                        _pdoc = _pdoc.replace(_pp+":", _pp+": (your input is: {})".format(_p[_pp]) if _pp in _p.keys()
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
        
        print("\n# ======================= #\n#   Availbale templates   #\n# ======================= #\n")
        TemplateLibrary.show_contents()
        
        if templates in ["*", "all"]:
            templates = list(TemplateLibrary._descriptions.keys())
        if templates is not None and not np.all([_t not in TemplateLibrary._descriptions.keys() for _t in np.atleast_1d(templates)]):
            print("\n\n\n# ========================= #\n#   Template descriptions   #\n# ========================= #\n")
            for _t in np.atleast_1d(templates):
                if _t in TemplateLibrary._descriptions.keys():
                    dash_string = "".join(["-"]*(len(_t)+6))
                    print("+{}+\n|   {}   |\n+{}+".format(dash_string, _t, dash_string))
                    TemplateLibrary.describe(_t)
                    print("\n")
    
    def build_sps(self, zcontinuous=1, sps=None):
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
        
        if "mass" not in self.model.params.keys() and "agebins" not in self.model.params.keys():
            from prospect.sources import CSPSpecBasis
            self._sps = CSPSpecBasis(zcontinuous=zcontinuous)
        else:
            from prospect.sources import FastStepBasis
            self._sps = FastStepBasis(zcontinuous=zcontinuous)
    
    def run_fit(self, which="dynesty", obs=None, model=None, sps=None, run_params={}, verbose=False):
        """
        
        """
        from prospect.fitting import lnprobfn
        from prospect.fitting import fit_model
        
        _model = self.model if model is None else model
        _obs = self.obs if obs is None else obs
        _sps = self.sps if sps is None else sps
        
        _run_params = {"optimize":False, "emcee":False, "dynesty":False}
        _run_params[which] = True
        _run_params.update(run_params)
        
        self._fit_output = fit_model(obs=_obs, model=_model, sps=_sps, lnprobfn=lnprobfn, **_run_params)
        
        if verbose:
            print("Done '{}' in {:.0f}s.".format(which, self.fit_output["sampling"][1]))
    
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
        print("\n# ================================== #\n#   Running parameters description   #\n# ================================== #\n")
        for _f in _list_fitter:
            if _f not in _fitters:
                warn("'{}' is not a know fitter.".format(_f))
                continue
            dash_string = "".join(["-"]*(len(_f)+6))
            print("+{}+\n|   {}   |\n+{}+".format(dash_string, _f, dash_string))
            _doc = eval("run_"+_f).__doc__.split(":param ")
            _doc = [_d for _d in _doc if _d.split(":")[0] not in _rejected_params and len(_d.split(":")[0].split(" "))==1]
            _doc[-1] = _doc[-1].split("Returns")[0]
            print("".join(["    "]+_doc))
    
    def write(self):
        """
        
        """
        return
    
    @staticmethod
    def read(self):
        """
        
        """
        return
    
    def show(self):
        """
        
        """
        return
            
        
        
        
    
    
    #-------------------#
    #   Properties      #
    #-------------------#
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
    def fit_output(self):
        """ Fitter output (dictionary with two keys, 'optimization' and 'sampling') """
        if not hasattr(self,"_fit_output"):
            self._fit_output = None
        return self._fit_output
    
    def has_fit_output(self):
        """ Test that 'fit_output' is not void """
        return self.fit_output is not None
