"""
Internally import from this, not from _wrapped_ufuncs.
Users should import only from non-private modules, of course.
"""

import numpy

from ._wrapped_ufuncs import *

_p_from_z = p_from_z
def p_from_z(z, lat, geo_strf_dyn_height=0, sea_surface_geopotential=0):
    return _p_from_z(z, lat, geo_strf_dyn_height, sea_surface_geopotential)
p_from_z.__doc__ = _p_from_z.__doc__

_z_from_p = z_from_p
def z_from_p(p, lat, geo_strf_dyn_height=0, sea_surface_geopotential=0):
    return _z_from_p(p, lat, geo_strf_dyn_height, sea_surface_geopotential)
z_from_p.__doc__ = _z_from_p.__doc__

_gibbs = gibbs
def gibbs(ns, nt, np, SA, t, p):
    params = {"ns": ns, "nt": nt, "np": np}
    for k, v in params.items():
        u = numpy.unique(v)
        if u.min() < 0 or u.max() > 2 or u.dtype.kind != "i":
            raise ValueError("ns, nt, np must contain integers 0, 1, or 2;"
                             f" found {k}={v}")
    return _gibbs(ns, nt, np, SA, t, p)
gibbs.__doc__ = _gibbs.__doc__


_gibbs_ice = gibbs_ice
def gibbs_ice(nt, np, t, p):
    params = {"nt": nt, "np": np}
    for k, v in params.items():
        u = numpy.unique(v)
        if u.min() < 0 or u.max() > 2 or u.dtype.kind != "i":
            raise ValueError("nt, np must contain integers 0, 1, or 2;"
                             f" found {k}={v}")
    return _gibbs_ice(nt, np, t, p)
gibbs_ice.__doc__ = _gibbs_ice.__doc__
