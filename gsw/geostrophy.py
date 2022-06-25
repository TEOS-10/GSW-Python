"""
Functions for calculating geostrophic currents.
"""

import numpy as np

from . import _gsw_ufuncs
from ._utilities import match_args_return, indexer
from .conversions import z_from_p

__all__ = ['geo_strf_dyn_height',
           'distance',
           'f',
           'geostrophic_velocity',
           ]

@match_args_return
def geo_strf_dyn_height(SA, CT, p, p_ref=0, axis=0, max_dp=1.0,
                        interp_method='pchip'):
    """
    Dynamic height anomaly as a function of pressure.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    p_ref : float or array-like, optional
        Reference pressure, dbar
    axis : int, optional, default is 0
        The index of the pressure dimension in SA and CT.
    max_dp : float
        If any pressure interval in the input p exceeds max_dp, the dynamic
        height will be calculated after interpolating to a grid with this
        spacing.
    interp_method : string {'pchip', 'linear'}
        Interpolation algorithm.

    Returns
    -------
    dynamic_height : array
        This is the integral of specific volume anomaly with respect
        to pressure, from each pressure in p to the specified
        reference pressure.  It is the geostrophic streamfunction
        in an isobaric surface, relative to the reference surface.

    """
    interp_methods = {'pchip' : 2, 'linear' : 1}
    if interp_method not in interp_methods:
        raise ValueError('interp_method must be one of %s'
                         % (interp_methods.keys(),))
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
    with np.errstate(invalid='ignore'):
        # The need for this context seems to be a bug in np.ma.any.
        if np.ma.any(np.ma.diff(np.ma.masked_invalid(p), axis=axis) <= 0):
            raise ValueError('p must be increasing along the specified axis')
    p = np.broadcast_to(p, SA.shape)
    goodmask = ~(np.isnan(SA) | np.isnan(CT) | np.isnan(p))
    dh = np.empty(SA.shape, dtype=float)
    dh.fill(np.nan)

    try:
        order = 'F' if SA.flags.fortran else 'C'
    except AttributeError:
        order = 'C'  # e.g., xarray DataArray doesn't have flags
    for ind in indexer(SA.shape, axis, order=order):
        igood = goodmask[ind]
        # If p_ref is below the deepest value, skip the profile.
        pgood = p[ind][igood]
        if  len(pgood) > 1 and pgood[-1] >= p_ref:
            sa = SA[ind][igood]
            ct = CT[ind][igood]
            # Temporarily add a top (typically surface) point and mixed layer
            # if p_ref is above the shallowest pressure.
            if pgood[0] > p_ref:
                ptop = np.arange(p_ref, pgood[0], max_dp)
                ntop = len(ptop)
                sa = np.hstack(([sa[0]]*ntop, sa))
                ct = np.hstack(([ct[0]]*ntop, ct))
                pgood = np.hstack((ptop, pgood))
            else:
                ntop = 0
            dh_all = _gsw_ufuncs.geo_strf_dyn_height_1(
                                         sa, ct, pgood, p_ref, max_dp,
                                         interp_methods[interp_method])
            if ntop > 0:
                dh[ind][igood] = dh_all[ntop:]
            else:
                dh[ind][igood] = dh_all

    return dh


def unwrap(lon, centered=True, copy=True):
    """
    Unwrap a sequence of longitudes or headings in degrees.

    Optionally center it as close to zero as possible

    By default, return a copy; if *copy* is False, avoid a
    copy when possible.

    Returns a masked array only if the input is a masked array.
    """
    # From pycurrents.data.ocean.  It could probably be simplified
    # for use here.

    masked_input = np.ma.isMaskedArray(lon)
    if masked_input:
        fill_value = lon.fill_value
        # masked_invalid loses the original fill_value (ma bug, 2011/01/20)
    lon = np.ma.masked_invalid(lon).astype(float)
    if lon.ndim != 1:
        raise ValueError("Only 1-D sequences are supported")
    if lon.shape[0] < 2:
        return lon
    x = lon.compressed()
    if len(x) < 2:
        return lon
    w = np.zeros(x.shape[0]-1, int)
    ld = np.diff(x)
    np.putmask(w, ld > 180, -1)
    np.putmask(w, ld < -180, 1)
    x[1:] += (w.cumsum() * 360.0)

    if centered:
        x -= 360 * np.round(x.mean() / 360.0)

    if lon.mask is np.ma.nomask:
        lon[:] = x
    else:
        lon[~lon.mask] = x
    if masked_input:
        lon.fill_value = fill_value
        return lon
    else:
        return lon.filled(np.nan)


@match_args_return
def distance(lon, lat, p=0, axis=-1):
    """
    Great-circle distance in m between lon, lat points.

    Parameters
    ----------
    lon, lat : array-like, 1-D or 2-D (shapes must match)
        Longitude, latitude, in degrees.
    p : array-like, scalar, 1-D or 2-D, optional, default is 0
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    axis : int, -1, 0, 1, optional
        The axis or dimension along which *lat and lon* vary.
        This differs from most functions, for which axis is the
        dimension along which p increases.

    Returns
    -------
    distance : 1-D or 2-D array
        distance in meters between adjacent points.

    """
    earth_radius = 6371e3

    if not lon.shape == lat.shape:
        raise ValueError('lon, lat shapes must match; found %s, %s'
                          % (lon.shape, lat.shape))
    if not (lon.ndim in (1, 2) and lon.shape[axis] > 1):
        raise ValueError('lon, lat must be 1-D or 2-D with more than one point'
                         ' along axis; found shape %s and axis %s'
                          % (lon.shape, axis))
    if lon.ndim == 1:
        one_d = True
        # xarray requires expand_dims() rather than [newaxis, :]
        lon = np.expand_dims(lon, 0)
        lat = np.expand_dims(lat, 0)
        axis = -1
    else:
        one_d = False

    # Handle scalar default; match_args_return doesn't see it.
    p = np.atleast_1d(p)
    one_d = (one_d and p.ndim == 1)

    if axis == 0:
        indm = (slice(0, -1), slice(None))
        indp = (slice(1, None), slice(None))
    else:
        indm = (slice(None), slice(0, -1))
        indp = (slice(None), slice(1, None))

    if np.all(p == 0):
        z = 0
    else:
        lon, lat, p = np.broadcast_arrays(lon, lat, p)

        p_mid = 0.5 * (p[indm] + p[indp])
        lat_mid = 0.5 * (lat[indm] + lat[indp])

        z = z_from_p(p_mid, lat_mid)

    lon = np.radians(lon)
    lat = np.radians(lat)

    dlon = np.diff(lon, axis=axis)
    dlat = np.diff(lat, axis=axis)

    a = ((np.sin(dlat / 2)) ** 2 + np.cos(lat[indm]) *
         np.cos(lat[indp]) * (np.sin(dlon / 2)) ** 2)

    angles = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = (earth_radius + z) * angles

    if one_d:
        distance = distance[0]

    return distance


@match_args_return
def f(lat):
    """
    Coriolis parameter in 1/s for latitude in degrees.
    """
    omega = 7.292115e-5  # (1/s)   (Groten, 2004).
    f = 2 * omega * np.sin(np.radians(lat))
    return f

@match_args_return
def geostrophic_velocity(geo_strf, lon, lat, p=0, axis=0):
    """
    Calculate geostrophic velocity from a streamfunction.

    Calculates geostrophic velocity relative to a reference pressure,
    given a geostrophic streamfunction and the position of each station
    in sequence along an ocean section.  The data can be from a single
    isobaric or "density" surface, or from a series of such surfaces.

    Parameters
    ----------
    geo_strf : array-like, 1-D or 2-D
        geostrophic streamfunction; see Notes below.
    lon : array-like, 1-D
        Longitude, -360 to 360 degrees
    lat : array-like, 1-D
        Latitude, degrees
    p : float or array-like, optional
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar.
        This used only for a tiny correction in the distance calculation;
        it is safe to omit it.
    axis : int, 0 or 1, optional
        The axis or dimension along which pressure increases in geo_strf.
        If geo_strf is 1-D, it is ignored.

    Returns
    -------
    velocity : array, 2-D or 1-D
        Geostrophic velocity in m/s relative to the sea surface,
        averaged between each successive pair of positions.
    mid_lon, mid_lat : array, 1-D
        Midpoints of input lon and lat.

    Notes
    -----
    The geostrophic streamfunction can be:

    - geo_strf_dyn_height (in an isobaric surface)
    - geo_strf_Montgomery (in a specific volume anomaly surface)
    - geo_strf_Cunninhgam (in an approximately neutral surface
      such as a potential density surface).
    - geo_strf_isopycnal (in an approximately neutral surface
      such as a potential density surface, a Neutral Density
      surface, or an omega surface (Klocker et al., 2009)).

    Only :func:`geo_strf_dyn_height` is presently implemented
    in GSW-Python.

    """
    lon = unwrap(lon)

    if lon.shape != lat.shape or lon.ndim != 1:
        raise ValueError('lon, lat must be 1-D and matching; found shapes'
                         ' %s and %s' % (lon.shape, lat.shape))

    if geo_strf.ndim not in (1, 2):
        raise ValueError('geo_strf must be 1-D or 2-d; found shape %s'
                         % (geo_strf.shape,))

    laxis = 0 if axis else -1

    ds = distance(lon, lat, p)

    mid_lon = 0.5 * (lon[:-1] + lon[1:])
    mid_lat = 0.5 * (lat[:-1] + lat[1:])

    u = np.diff(geo_strf, axis=laxis) / (ds * f(mid_lat))

    return u, mid_lon, mid_lat
