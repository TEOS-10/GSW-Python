
import numpy as np

from . import _gsw_ufuncs
from ._utilities import match_args_return, indexer


@match_args_return
def geo_strf_dyn_height(SA, CT, p, p_ref=0, axis=0):
    if SA.shape != CT.shape:
        raise ValueError('Shapes of SA and CT must match; found %s and %s'
                         % (SA.shape, CT.shape))
    if p.ndim == 1 and SA.ndim > 1:
        if len(p) != SA.shape[axis]:
            raise ValueError('With 1-D p, len(p) must be SA.shape[axis];\n'
                             ' found %d versus %d on specified axis, %d'
                             % (len(p), SA.shape[axis], axis))
        ind = [np.newaxis] * SA.ndim
        ind[axis] = slice(None)
        p = p[tuple(ind)]
    p_ref = float(p_ref)
    if (np.diff(p, axis=axis) <= 0).any():
        raise ValueError('p must be increasing along the specified axis')
    p = np.broadcast_to(p, SA.shape)
    goodmask = ~(np.isnan(SA) | np.isnan(CT) | np.isnan(p))
    dh = np.empty(SA.shape, dtype=float)
    dh.fill(np.nan)

    print(SA.shape)

    order = 'F' if SA.flags.fortran else 'C'
    for ind in indexer(SA.shape, axis, order=order):
        print(ind)
        igood = goodmask[ind]
        print(goodmask)
        print(goodmask[ind])
        print(dh.shape)
        # If p_ref is below the deepest value, skip the profile.
        pgood = p[ind][igood]
        if pgood[-1] >= p_ref and len(pgood) > 1:
            dh[ind][igood] = _gsw_ufuncs.geo_strf_dyn_height(
                                         SA[ind][igood],
                                         CT[ind][igood],
                                         pgood, p_ref)
    return dh


@match_args_return
def distance(lon, lat, p=0):
    """
    Great-circle distance in m between lon, lat points.

    Parameters
    ----------
    lon, lat : array-like, 1-D
        Longitude, latitude, in degrees.
    p : float or 1-D array-like, optional, default is 0
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    distance : 1-D array
        distance in meters between adjacent points.

    """
    # This uses the algorithm from pycurrents rather than the one
    # in GSW-Matlab.

    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError('lon, lat must be 1-D; found shapes %s and %s'
                         % (lon.shape, lat.shape))
    if lon.shape != lat.shape:
        raise ValueError('lon, lat must have same 1-D shape; found %s and %s'
                         % (lon.shape, lat.shape))
    if p != 0:
        if np.iterable(p) and p.shape != lon.shape:
            raise ValueError('lon, non-scalar p must have same 1-D shape;'
                             ' found %s and %s'
                             % (lon.shape, lat.shape))

        p = np.broadcast_to(p, lon.shape)

    slm = slice(None, -1)
    slp = slice(1, None)

    radius = 6371e3

    lon = np.radians(lon)
    lat = np.radians(lat)
    if p != 0:
        p_mid = 0.5 * (p[slp] + p[slm])
        lat_mid = 0.5 * (lat[slp] + lat[slm])
        z_mid = gsw.z_from_p(p_mid, lat_mid)
        radius += z_mid

    d = np.arccos(cos(lat[slm]) * cos(lat[slp]) * cos(lon[slp] - lon[slm])
                  + sin(lat[slm]) * sin(lat[slp])) * radius

    return d

