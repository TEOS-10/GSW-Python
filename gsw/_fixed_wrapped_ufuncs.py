"""
Internally import from this, not from _wrapped_ufuncs.
Users should import only from non-private modules, of course.
"""

from ._wrapped_ufuncs import *

_p_from_z = p_from_z
def p_from_z(z, lat, geo_strf_dyn_height=0, sea_surface_geopotential=0):
    return _p_from_z(z, lat, geo_strf_dyn_height, sea_surface_geopotential)
p_from_z.__doc__ = _p_from_z.__doc__

_z_from_p = z_from_p
def z_from_p(p, lat, geo_strf_dyn_height=0, sea_surface_geopotential=0):
    return _z_from_p(p, lat, geo_strf_dyn_height, sea_surface_geopotential)
z_from_p.__doc__ = _z_from_p.__doc__



