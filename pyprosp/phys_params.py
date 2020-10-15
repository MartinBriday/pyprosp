
import pandas
import numpy as np

from . import tools
from . import io
from prospect.models import transforms


class ProspectorPhysParams():
    """
    This class a physical parameters extractor from 'prospector' SED fitting results.
    """
    
    def __init__(self, **kwargs):
        """
        
        """
        if kwargs != {}:
            self.set_data(**kwargs)
        
    def set_data(self, chains, model):
        """
        Set up input data as attributes.
        
        Parameters
        ----------
        chains : [np.array]
            Fitted parameter chains.
            Shape must be (nb of fitting step, nb of fitted parameters).
        
        model : [prospect.models.sedmodel.SedModel]
            Prospector model object.
        
        
        Returns
        -------
        Void
        """
        self._chains = chains
        self._model = model
        self._param_chains = {_p:np.array([jj[ii] for jj in self.chains]) for ii, _p in enumerate(self.theta_labels)}
    
    @classmethod
    def from_h5(cls, filename, warnings=True):
        """
        Build a ProspectorPhysParams object directly from a .h5 file containing the results of a prospector fit.
        
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
        ProspectorPhysParams
        """
        from .prospector import Prospector
        _prosp = Prospector.from_h5(filename=filename, build_sps=False, warnings=warnings)
        _this = cls(chains=_prosp.chains, model=_prosp.model)
        return _this
    
    def _has_param_(self, param):
        """
        Test that a given parameter is in the model.
        
        Parameters
        ----------
        param : [string]
            Parameter's name.
            
        
        Returns
        -------
        bool
        """
        return param in self.model_dict
    
    def _is_param_free_(self, param):
        """
        Test that a given parameter is free (return True) or not (return False).
        If the given parameter is not included in the model, return False per default.
        
        Parameters
        ----------
        param : [string]
            Parameter's name.
            
        
        Returns
        -------
        bool
        """
        return (param in self.param_chains) if self._has_param_(param) else False
    
    def _is_param_dependant_(self, param):
        """
        Test that a given parameter has a dependance (return True) or not (return False).
        If the given parameter is not included in the model, return False per default.
        
        Parameters
        ----------
        param : [string]
            Parameter's name.
            
        
        Returns
        -------
        bool
        """
        return bool(self.model_dict[param].get("depends", False)) if self._has_param_(param) else False
    
    def _get_param_init_(self, param, default=-999):
        """
        Return the initial value of the given parameter.
        
        Parameters
        ----------
        param : [string]
            Parameter's name.
        
        Options
        -------
        default : [bool, float, string or list, etc.]
            If it cannot either find the given parameter in the model or the initial value for this parameter, return 'default'.
            Default is -999.
        
        
        Returns
        -------
        bool, float, string or list, etc.
        """
        return self.model_dict[param].get("init", default) if self._has_param_(param) else default
    
    def _get_param_chain_(self, param, default=-999)
        """
        Return a chain of the given parameter, either the fitted one or an array filled by the initial value.
        
        Parameters
        ----------
        param : [string]
            Parameter's name.
        
        Options
        -------
        default : [bool, float, string or list, etc.]
            If it cannot either find the given parameter in the model or the initial value for this parameter,
                the returned array is filled with the 'default' value.
            Default is -999.
        
        
        Returns
        -------
        array
        """
        if self._is_param_free_(param):
            return self.param_chains[param]
        else:
            return np.ones(self.len_chains)*self._get_param_init_(param=param, default=default)
    
    def set_z(self):
        """
        Set up redshift.
        
        
        Returns
        -------
        Void
        """
        self.phys_params_chains["z"] = self._get_param_chain_(io._DEFAULT_PHYS_PARAMS["z"])
    
    def set_mass(self):
        """
        Set up stellar mass.
        
        
        Returns
        -------
        Void
        """
        _mass = io._DEFAULT_PHYS_PARAMS["mass"]
        if self._is_param_free_(_mass):
            if self.model_dict[_mass]["N"] == 1
                self.phys_params_chains["mass"] = self.param_chains[_mass]
            else:
                _masses = self.param_chains[f"{_mass}_{ii+1}"] for ii in np.arange(self.model_dict[_mass]["N"])]
                self.phys_params_chains["mass"] = np.sum(_masses, axis=0)
        elif self.has_agebins():
            if self._has_param_(io._DEFAULT_PHYS_PARAMS["z_fraction"]):
                self.phys_params_chains["mass"] = self._get_param_chain_(io._DEFAULT_PHYS_PARAMS["total_mass"])
            elif self._has_param_(io._DEFAULT_PHYS_PARAMS["logmass"]):
                self.phys_params_chains["mass"] = 10 ** self._get_param_chain_(io._DEFAULT_PHYS_PARAMS["logmass"])
    
    def set_sfr(self):
        """
        Set up Star Formation Rate (SFR).
        
        
        Returns
        -------
        Void
        """
        _sfr = io._DEFAULT_PHYS_PARAMS["sfr"]
        if self._is_param_free_(_sfr):
            self.phys_params_chains["sfr"] = self.param_chains[_sfr]
            
    
    
    
    
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
    def param_chains(self):
        """ Dictionary containing the fitted parameter chains """
        return self._param_chains
    
    @property
    def model(self):
        """ Prospector SED Model """
        return self._model
    
    @property
    def model_dict(self):
        """ Prospector SED Model 'config_dict' attribute """
        return self.model.config_dict
    
    @property
    def theta_labels(self):
        """ List of the fitted parameters """
        return self.model.theta_labels()
    
    @property
    def has_agebins(self):
        """ Test that 'agebins' parameter is in the model """
        return "agebins" in self.model.config_dict
    
    @property
    def phys_params_chain(self):
        """ Dictionary containing the chain of every physical parameters """
        if not hasattr(self, "_phys_params_chains"):
            self._phys_params_chains = {}
        return self._phys_params_chains
    
    @property
    def phys_params(self):
        """ Dictionary containing for every physical parameters a tuple (median, -sigma, +sigma) """
        _phys_params = {_pp:np.percentiles(_ppc, [16, 50, 84]) for _pp, _ppv in self.phys_params_chain.items()}
        return {_pp:(_ppv[1], _ppv[1]-_ppv[0], _ppv[2]-_ppv[1]) for _pp, _ppv in _phys_params}
    
    
