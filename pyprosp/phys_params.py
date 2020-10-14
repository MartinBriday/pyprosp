
import pandas
import numpy as np

from . import tools
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
        
        """
        self._chains = chains
        self._model = model
        self._param_chains = {_p:np.array([jj[ii] for jj in self.chains]) for ii, _p in enumerate(self.theta_labels)}
    
    @classmethod
    def from_h5(cls, filename, warnings=True):
        """
        
        """
        from .prospector import Prospector
        _prosp = Prospector.from_h5(filename=filename, build_sps=False, warnings=warnings)
        _this = cls(chains=_prosp.chains, model=_prosp.model)
        return _this
    
    
    
    
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
            self._phys_params = {}
        return self._phys_params
    
    @property
    def phys_params(self):
        """ Dictionary containing for every physical parameters a tuple (median, -sigma, +sigma) """
        _phys_params = {_pp:np.percentiles(_ppc, [16, 50, 84]) for _pp, _ppv in self.phys_params_chain.items()}
        return {_pp:(_ppv[1], _ppv[1]-_ppv[0], _ppv[2]-_ppv[1]) for _pp, _ppv in _phys_params}
    
    
