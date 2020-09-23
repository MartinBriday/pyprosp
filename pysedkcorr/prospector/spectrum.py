
import pandas
import numpy as np

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
    
    def _load_spectra_(self):
        """
        
        """
        self._spectrum_chain = np.array([self._get_spectrum_(theta=_theta) for _theta in self.chains])
    
    def _get_spectrum_(self, theta):
        """
        
        """
        _spec, _, _ = self.model.sed(theta, obs=self.obs, sps=self.sps)
        return _spec
    
    def show(self):
        """
        
        """
        return
    
        
        
    
    
    #----------------#
    #   Properties   #
    #----------------#
    @property
    def chains(self):
        """ List of the fitted parameter chains """
        return self._chains
    
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
    def spectrum_chain(self):
        """ Array containing spectrum chain """
        if not hasattr(self, "_spectrum_chain"):
            self._load_spectra_()
        return self._spectrum_chain
    
    @property
    def wavelengths(self):
        """ Wavelength array corresponding with the spectra """
        return self.sps.wavelengths * (1.0 + self.model.params.get("zred", 0.0))
