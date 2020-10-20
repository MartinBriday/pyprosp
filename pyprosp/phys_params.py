
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
        Initialization.
        Can execute 'set_data'.
        
        Options
        -------
        chains : [np.array]
            Fitted parameter chains.
            Shape must be (nb of fitting step, nb of fitted parameters).
        
        model : [prospect.models.sedmodel.SedModel]
            Prospector model object.
        
        
        Returns
        -------
        Void
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
        Test that a given parameter.s is/are in the model.
        
        Parameters
        ----------
        param : [string or list(string)]
            Parameter's name(s).
            
        
        Returns
        -------
        bool
        """
        return np.all([_param in self.model_dict for _param in np.atleast_1d(param)])
    
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
    
    def get_z(self):
        """
        Return redshift chain.
        
        
        Returns
        -------
        np.array
        """
        return self._get_param_chain_(PHYS_PARAMS["z"]["prosp_name"])
    
    def get_mass(self):
        """
        Return stellar mass chain.
        
        
        Returns
        -------
        np.array
        """
        if self._is_param_free_(PHYS_PARAMS["total_mass"]["prosp_name"]):
            return self._get_param_chain_(PHYS_PARAMS["total_mass"]["prosp_name"])
        elif self._is_param_free_(PHYS_PARAMS["log_mass"]["prosp_name"]):
            return 10 ** self._get_param_chain_(PHYS_PARAMS["log_mass"]["prosp_name"])
        else:
            _mass = self._get_param_chain_(PHYS_PARAMS["mass"]["prosp_name"])
            return np.sum(np.atleast_2d(_mass), axis=0)
    
    def get_log_mass(self):
        """
        Return stellar mass chain in log scale.
        
        
        Returns
        -------
        np.array
        """
        return np.log10(self.get_mass())
    
    def get_sfr(self):
        """
        Return Star Formation Rate (SFR) chain.
        If there are age bins in the fitted model, return the most recent SFR.
        
        
        Returns
        -------
        np.array
        """
        if self.has_agebins():
            _mass = self.get_mass()
            _agebins = self.model_dict[io._DEFAULT_NON_PARAMETRIC_SFH["agebins"]]
            if self._has_param_(io._DEFAULT_NON_PARAMETRIC_SFH["z_fraction"]):
                _z_fraction = self._get_param_chain_(io._DEFAULT_NON_PARAMETRIC_SFH["z_fraction"])
                _sfr = [transforms.zfrac_to_sfr(total_mass=_mass[ii], z_fraction=_z_fraction.T[ii], agebins=_agebins)[0]
                        for ii in np.arange(self.len_chains)]
            elif self._has_param_(io._DEFAULT_NON_PARAMETRIC_SFH["logsfr_ratios"]):
                _logsfr_ratios = self.model_dict[io._DEFAULT_NON_PARAMETRIC_SFH["logsfr_ratios"]]["init"]
                _logmass = np.log10(_mass)
                _sfr = [transforms.zfrac_to_sfr(logmass=_logmass, logsfr_ratios=_logsfr_ratios.T[ii], agebins=_agebins)[0]
                        for ii in np.arange(self.len_chains)]
            else:
                _sfr = None
            return np.array(_sfr)
        elif not self.has_agebins() and self._has_param_([PHYS_PARAMS[_p]["prosp_name"] for _p in ["tage", "tau", "mass"]]):
            _tage, _tau, _mass = [self._get_param_chain_(PHYS_PARAMS[_p]["prosp_name"]) for _p in ["tage", "tau", "mass"]]
            return tools.sfh_delay_tau_to_sfr(tage=_tage, tau=_tau, mass=_mass)
        else:
            _sfr = np.atleast_2d(self._get_param_chain_(PHYS_PARAMS["sfr"]["prosp_name"]))
            return _sfr[0]
    
    def get_log_sfr(self):
        """
        Return Star Formation Rate (SFR) chain in log scale.
        
        
        Returns
        -------
        np.array
        """
        return np.log10(self.get_sfr())
    
    def get_ssfr(self):
        """
        Return specific Star Formation Rate (SFR) chain.
        
        
        Returns
        -------
        np.array
        """
        return self.get_sfr() / self.get_mass()
    
    def get_log_ssfr(self):
        """
        Return specific Star Formation Rate (SFR) chain in log scale.
        
        
        Returns
        -------
        np.array
        """
        return np.log10(self.get_ssfr())
    
    def get_dust1(self):
        """
        Return the extra optical depth towards young stars (the FSPS 'dust1' parameter) chain.
        
        
        Returns
        -------
        np.array
        """
        _dust2, _dust_ratio = [self._get_param_chain_(PHYS_PARAMS[_p]["prosp_name"]) for _p in ["dust2", "dust_ratio"]]
        return _dust2 * _dust_ratio
    
    def get_tburst(self):
        """
        Return the "Time at wich SF burst happens" chain.
        
        
        Returns
        -------
        np.array
        """
        _tage, _fage_burst = [self._get_param_chain_(PHYS_PARAMS[_p]["prosp_name"]) for _p in ["tage", "fage_burst"]]
        return _tage * _fage_burst
    
    def reset(self):
        """
        Reset the 'phys_params_chains' attribute to None.
        
        
        Returns
        -------
        Void
        """
        self._phys_params_chains = None
    
    def _get_phys_param_chain_(self, param, warnings=True):
        """
        Extract and return the desired physical parameter chain.
        
        Parameters
        ----------
        params : [string]
            Physical parameter's name.
        
        Options
        -------
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        
        Returns
        -------
        np.array
        """
        try:
            return eval(f"self.get_{param}")()
        except AttributeError:
            try:
                return self._get_param_chain_(PHYS_PARAMS[param]["prosp_name"])
            except KeyError:
                return
        except Exception as _excpt:
            if warnings:
                warn(f"You tried to set the physical parameter '{param}' for which it raised the error:\n"+
                     f"{_excpt.__class__} --> {_excpt}.")
            return
    
    def set_phys_params(self, params="*", reset=False, warnings=True):
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
        
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        
        Returns
        -------
        Void
        """
        if reset:
            self.reset()
        if params in ["*", "all"]:
            params = list(PHYS_PARAMS.keys())
        for _param in np.atleast_1d(params):
            if _param in ["total_mass", "logmass", "dust_ratio", "fage_burst"]:
                continue
            _chain = self._get_phys_param_chain_(param=_param, warnings=warnings)
            if _chain is not None:
                self.phys_params_chains[_param] = _chain
    
    def get_phys_params(self, params="*", reset=False, warnings=True):
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
        
        warnings : [bool]
            If True, allow warnings to be printed.
            Default is True.
        
        
        Returns
        -------
        dict
        """
        self.set_phys_params(params=params, reset=reset, warnings=warnings)
        return self.phys_params
    
    #---------------#
    #   Plottings   #
    #---------------#
    def show(self, params="*", savefile=None, **kwargs):
        """
        Draw a corner plot of the physical parameters.
        
        Parameters
        ----------
        params : [string or list(string)]
            List of physical parameters.
            If "*", set every available physical parameters.
            Default is "*".
        
        Options
        -------
        savefile : [string or None]
            Give a directory to save the figure.
            Default is None (not saved).
        
        **kwargs:
            corner.corner (or triangle.corner) kwargs
        
        
        Returns
        -------
        matplotlib.Figure
        """
        try:
            import corner as triangle
        except(ImportError):
            import triangle
        except:
            raise ImportError("Please install the `corner` package.")
        
        if params in ["*", "all"]:
            params = list(PHYS_PARAMS.keys())
        
        _labels = []
        _chains = []
        for _param in np.atleast_1d(params):
            if _param in ["total_mass", "logmass", "dust_ratio", "fage_burst"]:
                continue
            _chain = self._get_phys_param_chain_(param=_param, warnings=False)
            if _chain is not None:
                if np.all([_cv == _chain[0] for _cv in _chain]):
                    continue
                _labels.append(_param)
                _chains.append(_chain)
        _chains = np.array(_chains)
        
        _kwargs = {"plot_datapoints": False, "plot_density": False, "fill_contours": True, "show_titles": True}
        _kwargs.update(kwargs)
        
        _fig = triangle.corner(_chains.T, labels=_labels, quantiles=[0.16, 0.5, 0.84], **_kwargs)
        
        if savefile is not None:
            _fig.savefig(savefile)
        
        return _fig
    
    
    #------------------#
    #   Descriptions   #
    #------------------#
    @staticmethod
    def describe_phys_params():
        """
        Print description on available physical parameters:
            - Key name to call/set it
            - Description
            - Unit
            - Prospector/FSPS parameter name
        
        
        Returns
        -------
        Void
        """
        from IPython.display import Latex
        print(tools.get_box_title(title="Physical parameters description", box="\n#=#\n#   {}   #\n#=#\n"))
        print("You can get physical parameter values running 'get_phys_params' method on a loaded instance.")
        print("""You can either give as argument one or a list of the following keys, or "*" to get all of them.""")
        print("(Note that you'll finally only get the available ones depending on the SED fitted model)")
        for _p, _pv in PHYS_PARAMS.items():
            if _p in ["total_mass", "dust_ratio", "fage_burst"]:
                continue
            display(Latex(f"""- {_p}:{f" [{_pv['unit']}]" if _pv['unit'] else ""}"""))
            display(Latex("$\hspace{1cm}$"+f"{_pv['def']}"+
                          (f" (FSPS param's name = '{_pv['prosp_name']}')." if _pv["prosp_name"] else "")))
    
    
    
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
    
    
