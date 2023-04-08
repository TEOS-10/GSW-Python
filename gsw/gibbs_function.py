"""
Wrapper to provide the Gibbs potential
"""
import numpy

from . import _gsw_ufuncs
from ctypes import CDLL, c_double


def _get_compiled_gibbs():
    name = "gsw_gibbs"
    libname = _gsw_ufuncs.__file__
    libobj = CDLL(libname)
    gibbs_compiled = getattr(libobj, name)
    gibbs_compiled.restype = c_double
    return gibbs_compiled


_gibbs_compiled = _get_compiled_gibbs()


def _basic_gibbs(ns, nt, np, SA, CT, p):
    """ this doc is lost during the ufunc-ification"""
    return _gibbs_compiled(ns, nt, np, c_double(SA), c_double(CT), c_double(p))


_gibbs_ufunc = numpy.frompyfunc(_basic_gibbs, 6, 1)


def gibbs(ns, nt, np, SA, CT, p, **kwargs):
    """Calculates the specific Gibbs free energy and derivatives up to order 2

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

   **kwargs: If either SA, CT or p is an nd.array then `gibbs` behaves
        like a `ufunc` and **kwargs can be any of `ufunc`
        keywords. See the :ref:`ufunc docs <ufuncs.kwargs>`.


    Returns
    -------
    gibbs : specific Gibbs energy [J/kg] or its derivative

    """
    if (isinstance(SA, numpy.ndarray)
        or isinstance(CT, numpy.ndarray)
            or isinstance(p, numpy.ndarray)):
        return _gibbs_ufunc(ns, nt, np, SA, CT, p, **kwargs)

    return _gibbs_compiled(ns, nt, np,
                           c_double(SA), c_double(CT), c_double(p))
