
import numpy as np
import pandas
import time
import sys
import os

from . import io

class Prospector():
    """
    
    """
    
    def __init__(self):
        """
        
        """
        return
    
    def set_data(self, data, zcmb, sn_name=None):
        """
        
        """
        self._data = data.copy()
        self._zcmb = zcmb
        self._sn_name = sn_name
        self._filters = self._get_filters()
    
    def build_obs(self):
        """
        
        """
        from sedpy.observate import load_filters
        
    
    def build_model(self):
        """
        
        """
    
    def build_sps(self, sps=None, zcontinuous=1):
        """
        Create the appropriate sps.
        
        Options
        -------
        sps : [sps instance]
            If not None, this will set the 'sps' attribute with the given sps.
            Default is None.
        
        zcontinuous : [float]
            python-fsps parameter controlling how metallicity interpolation of the SSPs is acheived :
                - 0: use discrete indices (controlled by parameter "zmet")
                - 1: linearly interpolate in log Z/Z_\sun to the target metallicity (the parameter "logzsol")
                - 2: convolve with a metallicity distribution function at each age (the MDF is controlled by the parameter "pmetals")
            A value of '1' is recommended.
            Default is 1.
        
        
        Returns
        -------
        Void
        """
        if self.run_params["model_params"] == "parametric_sfh":
            self._sps = CSPSpecBasis(zcontinuous=self.run_params["zcontinuous"]) if sps is None else sps
        else:
            self._sps = FastStepBasis(zcontinuous=self.run_params["zcontinuous"]) if sps is None else sps
