"""
Internally import from this, not from _wrapped_ufuncs.
Users should import only from non-private modules, of course.
"""

import numpy

from ._wrapped_ufuncs import *

attrs = ["__doc__", "types", "nin", "nout"]


def transfer_attrs(new, orig):
    for attr in attrs:
        setattr(new, attr, getattr(orig, attr))


_p_from_z = p_from_z


def p_from_z(z, lat, geo_strf_dyn_height=0, sea_surface_geopotential=0):
    if numpy.ma.any(numpy.ma.asarray(z) > 5):
        raise ValueError(f"z should be generally negative; found max(z) = {numpy.ma.max(z)}")
    return _p_from_z(z, lat, geo_strf_dyn_height, sea_surface_geopotential)


transfer_attrs(p_from_z, _p_from_z)

_z_from_p = z_from_p


def z_from_p(p, lat, geo_strf_dyn_height=0, sea_surface_geopotential=0):
    return _z_from_p(p, lat, geo_strf_dyn_height, sea_surface_geopotential)


transfer_attrs(z_from_p, _z_from_p)

_gibbs = gibbs


def gibbs(ns, nt, np, SA, t, p):
    params = {"ns": ns, "nt": nt, "np": np}
    for k, v in params.items():
        u = numpy.unique(v)
        if u.min() < 0 or u.max() > 2 or u.dtype.kind != "i":
            raise ValueError(f"ns, nt, np must contain integers 0, 1, or 2; found {k}={v}")
    return _gibbs(ns, nt, np, SA, t, p)


transfer_attrs(gibbs, _gibbs)


_gibbs_ice = gibbs_ice


def gibbs_ice(nt, np, t, p):
    params = {"nt": nt, "np": np}
    for k, v in params.items():
        u = numpy.unique(v)
        if u.min() < 0 or u.max() > 2 or u.dtype.kind != "i":
            raise ValueError(f"nt, np must contain integers 0, 1, or 2; found {k}={v}")
    return _gibbs_ice(nt, np, t, p)


transfer_attrs(gibbs_ice, _gibbs_ice)
