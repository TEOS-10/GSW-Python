"""
Wrapper to provide the Gibbs potential
"""
import numpy

from . import _gsw_ufuncs
from ctypes import CDLL, c_double


def _set_gibbs():
    name = "gsw_gibbs"
    libname = _gsw_ufuncs.__file__
    libobj = CDLL(libname)
    gibbs_compiled = getattr(libobj, name)
    gibbs_compiled.restype = c_double

    def gibbs(ns, nt, np, SA, CT, p):
        """
        Calculates the specific Gibbs free energy and derivatives up to order 2

        Parameters
        ----------
        ns : int
            order of SA derivative
        nt : int
            order of CT derivative
        np : int
            order of p derivative
        SA : array-like
            Absolute Salinity, g/kg
        CT : array-like
            Conservative Temperature (ITS-90), degrees C
        p : array-like
            Sea pressure (absolute pressure minus 10.1325 dbar), dbar

        Returns
        -------
        gibbs : specific Gibbs energy [J/kg] or its derivative


        """
        if isinstance(SA, numpy.ndarray):
            shape = SA.shape
            assert CT.shape == SA.shape
            assert p.shape == SA.shape
            value = numpy.zeros(shape)
            for k in numpy.ndindex(shape):
                value[k] = gibbs_compiled(ns, nt, np,
                                          c_double(SA[k]), c_double(CT[k]), c_double(p[k]))
        else:
            value = gibbs_compiled(ns, nt, np,
                                   c_double(SA), c_double(CT), c_double(p))

        return value
    return gibbs


gibbs = _set_gibbs()
