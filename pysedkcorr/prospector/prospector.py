
import numpy as np
import pandas
import time
import sys
import os
import warnings

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
    
    def build_obs(self, obs=None):
        """
        Build a dictionary containing observations in a prospector compatible format.
        
        Options
        -------
        obs : [dict]
            Can load an already existing prospector compatible 'obs' dictionary.
            Default is None.
        
        
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
        if self.has_spec_in():
            self._obs.update({"wavelength":np.array(self.spec_in["lbda"]),
                              "spectrum":np.array(self.spec_in["flux"]),
                              "unc":np.array(self.spec_in["flux.err"]) if "flux.err" in self.spec_in.keys() else None})
        self._obs = fix_obs(self._obs)
        
    def build_model(self, model=None, templates=None, verbose=False):
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
        model :  [prospect.models.sedmodel.SedModel or dict]
            Can load an already existing prospector compatible 'model' SedModel.
            Can update a SedModel compatible 'model' dictionary before to create the SedModel on it.
            Default is None.
        
        verbose : [bool]
            If True, print information.
            Default is False.
        
        
        Returns
        -------
        Void
        """
        from prospect.models.sedmodel import SedModel
        from prospect.models.templates import TemplateLibrary
        from prospect.models import priors
        
        if type(model) is SedModel:
            self._model = model
            return
        
        if verbose:
            self.describe_templates(templates=templates)
        
        _model = {} if model is None else model
        for _t in np.atleast_1d(templates):
            if _t in TemplateLibrary._descriptions.keys():
                _model.update(TemplateLibrary[_t])
        for _p, _pv in _model.items():
            try:
                if _pv["prior"].__module__ != priors.__name__:
                    _pv["prior"] = self._build_prior_(_pv["prior"])
            except(AttributeError):
                pass
        
        if verbose:
            print("# ================= #\n#   Current model   #\n# ================= #\n")
            print(_model)
        
        self._model = SedModel(_model)
    
    def modify_model(self, changes=None, removing=None, verbose=False):
        """
        Apply changes on the loaded model.
        
        Parameters
        ----------
        params : [dict or None]
            Dictionary containing the desired changes for the desired parameters.
            As an example, params={"zred"={"init"=2.}}) will change the initial value of the 'zred parameter, without changing anything else.
            Default is None.
        
        removing : [string or list(string) or None]
            Fill this input with every parameters you want to remove from the model.
            Default is None.
        
        Options
        -------
        verbose : [bool]
            If True, print information.
            Default is False.
        
        
        Returns
        -------
        Void
        """
        _model = self.model.config_dict
        if verbose:
            print("# ================== #\n#   Previous model   #\n# ================== #\n")
            print(_model)
        
        #Changes
        if changes is not None:
            _changes = changes.copy()
            for _p, _pv in _changes.items():
                for _k, _kv in _pv.items():
                    if _k is not in _model[_p].keys():
                        warnings.warn("'{}' is not included in '{}' model parameters.".format(_k, _p))
                        _pv.pop(_k)
                    if _k == "prior":
                        _pv[_k] = self._build_prior_(_kv)
                _model[_p].update(_c)
        
        #Removing
        if removing is not None:
            for _t in np.atleast_1d(removing):
                if _t in _model.keys():
                    _model.pop(_t)
        
        if verbose:
            print("\n\n\n# ============= #\n#   New model   #\n# ============= #\n")
            print(_model)
        
        self._model = SedModel(_model)
    
    @staticmethod
    def _build_prior_(prior):
        """
        Build and return a prospector SedModel compatible prior.
        
        Parameters
        ----------
        prior : [dict]
            Dictionary containing every required parameters to build a prior object.
            
            
        Returns
        -------
        prospect.models.priors.[chosen prior]
        """
        from prospect.models import priors
        _name = prior.pop("name")
        return eval("priors.{}".format(_name))(**prior)
    
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
        
        print("# ======================= #\n#   Availbale templates   #\n# ======================= #\n")
        TemplateLibrary.show_contents()
        
        if templates in ["*", "all"]:
            templates = list(TemplateLibrary._descriptions.keys())
        if templates is not None and not np.all([_t not in TemplateLibrary._descriptions.keys() for _t in np.atleast_1d(templates)]):
            print("\n\n\n# ========================= #\n#   Detailed descriptions   #\n# ========================= #\n")
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
