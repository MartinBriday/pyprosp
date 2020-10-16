
import pandas
import numpy as np
from warnings import warn

from . import tools
from . import io
from prospect.models import transforms


PHYS_PARAMS = io._DEFAULT_PHYS_PARAMS.copy()

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
            If there is no initial value for the given parameter, return 'default'.
            Default is -999.
        
        
        Returns
        -------
        bool, float, string or list, etc.
        """
        if self._has_param_(param):
            return self.model_dict[param].get("init", default)
        else:
            raise KeyError(f"'{param}' is not in the model.")
    
    def _get_param_N_(self, param, default=None):
        """
        Return the length of the given parameter (single value or vector).
        
        Parameters
        ----------
        param : [string]
            Parameter's name.
        
        Options
        -------
        default : [bool, float, string or list, etc.]
            If there is no initial value for the given parameter, return 'default'.
            Default is None.
        
        
        Returns
        -------
        int
        """
        if self._has_param_(param):
            return self.model_dict[param].get("N", default)
        else:
            raise KeyError(f"'{param}' is not in the model.")
    
    def _get_init_chain_(self, param, default=-999):
        """
        Return an array filled with the initial value of the given parameter with same length as for the chains.
        
        Parameters
        ----------
        param : [string]
            Parameter's name.
        
        Options
        -------
        default : [bool, float, string or list, etc.]
            If there is no initial value for the given parameter, return 'default'.
            Default is -999.
        
        
        Returns
        -------
        np.array
        """
        return np.ones(self.len_chains)*self._get_param_init_(param=param, default=default)
    
    def _get_param_chain_(self, param, default=-999):
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
        _n = self._get_param_N_(param)
        if self._is_param_free_(param):
            return self.param_chains[param] if _n == 1 else \
                   np.array([self.param_chains[f"{param}_{ii+1}"] for ii in np.arange(_n)])
        else:
            return self._get_init_chain_(param, default) if _n == 1 else \
                   np.array([self._get_init_chain_(param, default) for ii in np.arange(_n)])
    
    def set_z(self):
        """
        Set up redshift.
        
        
        Returns
        -------
        Void
        """
        self.phys_params_chains["z"] = self._get_param_chain_(PHYS_PARAMS["z"]["prosp_name"])
    
    def set_mass(self):
        """
        Set up stellar mass.
        
        
        Returns
        -------
        Void
        """
        if self._is_param_free_(PHYS_PARAMS["mass"]["prosp_name"]):
            _mass = self._get_param_chain_(PHYS_PARAMS["mass"]["prosp_name"])
            self.phys_params_chains["mass"] = np.sum(np.atleast_2d(_mass), axis=0)
        elif self.has_agebins():
            if self._has_param_(io._DEFAULT_NON_PARAMETRIC_SFH["z_fraction"]):
                self.phys_params_chains["mass"] = self._get_param_chain_(PHYS_PARAMS["total_mass"]["prosp_name"])
            elif self._has_param_(io._DEFAULT_NON_PARAMETRIC_SFH["logsfr_ratios"]):
                #_logmass = self._get_param_chain_(PHYS_PARAMS["log(mass)"]["prosp_name"])
                #_logsfr_ratios = self._get_param_chain_(io._DEFAULT_NON_PARAMETRIC_SFH["logsfr_ratios"])
                #_agebins = self.model_dict[io._DEFAULT_NON_PARAMETRIC_SFH["agebins"]]
                #_mass = [np.sum(transforms.logsfr_ratios_to_masses(logmass=_logmass[ii],
                #                                                   logsfr_ratios=_logsfr_ratios.T[ii],
                #                                                   agebins=_agebins))
                #         for ii in np.arange(self.len_chains)]
                #self.phys_params_chains["mass"] = np.array(_mass)
                self.phys_params_chains["mass"] = 10 ** self._get_param_chain_(PHYS_PARAMS["logmass"]["prosp_name"])
    
    def set_sfr(self):
        """
        Set up Star Formation Rate (SFR).
        
        
        Returns
        -------
        Void
        """
        if self._is_param_free_(PHYS_PARAMS["sfr"]["prosp_name"]):
            _sfr = self._get_param_chain_(PHYS_PARAMS["sfr"]["prosp_name"])
            self.phys_params_chains["sfr"] = np.sum(np.atleast_2d(_sfr), axis=0)
        #elif self.has_agebins():
        
    
    def reset(self):
        """
        Reset the 'phys_params_chains' attribute to None.
        
        
        Returns
        -------
        Void
        """
        self._phys_params_chains = None
    
    def set_phys_params(self, params="*", reset=False):
        """
        Extract the desired physical parameters and store their chain in the attribute 'phys_params_chains'.
        
        Parameters
        ----------
        params : [string or list(string)]
            List of physical parameters.
            If "*", set every available physical parameters.
            Default is "*".
        
        Options
        -------
        reset : [bool]
            If True, reset the attribute 'phys_params_chains'.
            Default is False.
        
        
        Returns
        -------
        Void
        """
        if reset:
            self.reset()
        if params in ["*", "all"]:
            params = list(PHYS_PARAMS.keys())
        for _param in np.atleast_1d(params):
            if _param in ["total_mass"]:
                continue
            try:
                eval(f"self.set_{_param}")()
            except AttributeError:
                try:
                    self.phys_params_chains[_param] = self._get_param_chain_(PHYS_PARAMS[_param]["prosp_name"])
                except KeyError:
                    continue
            except Exception:
                warn(f"You tried to set the physical parameter '{_param}' for which it raised the error: {Exception}")
    
    def get_phys_params(self, params="*", reset=False):
        """
        Return the physical parameters in a dictionary containing, for each desired parameter,
        a tuple (median, -sigma, +sigma) from the corresponding chain.
        
        Parameters
        ----------
        params : [string or list(string)]
            List of physical parameters.
            If "*", set every available physical parameters.
            Default is "*".
        
        Options
        -------
        reset : [bool]
            If True, reset the attribute 'phys_params_chains'.
            Default is False.
        
        
        Returns
        -------
        dict
        """
        self.set_phys_params(params=params, reset=reset)
        return self.phys_params
    
    
    
    
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
    def phys_params_chains(self):
        """ Dictionary containing the chain of every physical parameters """
        if not hasattr(self, "_phys_params_chains") or self._phys_params_chains is None:
            self._phys_params_chains = {}
        return self._phys_params_chains
    
    @property
    def phys_params(self):
        """ Dictionary containing for every physical parameters a tuple (median, -sigma, +sigma) """
        _phys_params = {_pp:np.percentile(_ppc, [16, 50, 84]) for _pp, _ppc in self.phys_params_chains.items()}
        return {_pp:(_ppv[1], _ppv[1]-_ppv[0], _ppv[2]-_ppv[1]) for _pp, _ppv in _phys_params.items()}
    
    
