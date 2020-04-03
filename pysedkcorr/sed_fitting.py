import numpy as np
import pandas
from warnings import warn

from propobject import BaseObject

#from sedkcorr.sed_fitting import lephare
from pylephare import montecarlo
from . import tools


SED_FITTER = {"LePhare":montecarlo.MCLePhare,
                  }

def SEDFitting( data, sn_z=None, sn_name=None, sed_fitter="LePhare",
                filters=["sdss.u", "sdss.g", "sdss.r", "sdss.i", "sdss.z"],
                **extras ):
    """
    This class fit the SED with the given photometric measurements.
    """
    kwargs = eval("tools.{}_format_input_data".format(sed_fitter))(data=data, sn_z=sn_z, sn_name=sn_name, filters=filters)
    return SED_FITTER[sed_fitter](**{**kwargs, **extras})

    
