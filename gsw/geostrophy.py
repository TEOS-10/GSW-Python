"""
Functions for calculating geostrophic currents.
"""

import numpy as np

from . import _gsw_ufuncs
from ._utilities import match_args_return, indexer


__all__ = ['geo_strf_dyn_height',
           'distance',
           'f',
           'geostrophic_velocity',
           ]

@match_args_return
def geo_strf_dyn_height(SA, CT, p, p_ref=0, axis=0):
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
    axis : int, optional
        The index of the pressure dimension in SA and CT.

    Returns
    -------
    dynamic_height : array
        This is the integral of specific volume anomaly with respect
        to pressure, from each pressure in p to the specified
        reference pressure.  It is the geostrophic streamfunction
        in an isobaric surface, relative to the reference surface.

    """
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

    order = 'F' if SA.flags.fortran else 'C'
    for ind in indexer(SA.shape, axis, order=order):
        igood = goodmask[ind]
        # If p_ref is below the deepest value, skip the profile.
        pgood = p[ind][igood]
        # The C function calls the rr68 interpolation, which
        # requires at least 4 "bottles"; but the C function is
        # not checking this, so we need to do so.
        if pgood[-1] >= p_ref and len(pgood) > 3:
            dh[ind][igood] = _gsw_ufuncs.geo_strf_dyn_height(
                                         SA[ind][igood],
                                         CT[ind][igood],
                                         pgood, p_ref)
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

    if lon.mask is ma.nomask:
        lon[:] = x
    else:
        lon[~lon.mask] = x
    if masked_input:
        lon.fill_value = fill_value
        return lon
    else:
        return lon.filled(np.nan)


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


@match_args_return
def f(lat):
    """
    Coriolis parameter in 1/s for latitude in degrees.
    """
    omega = 7.292115e-5;                              #(1/s)   (Groten, 2004)
    f = 2 * omega * np.sin(np.radians(lat))
    return f

@match_args_return
def geostrophic_velocity(geo_strf, lon, lat, p=0, axis=0):
    """
    Calculate geostrophic velocity from a streamfunction.

    Calculates geostrophic velocity relative to the sea surface, given a
    geostrophic streamfunction and the position of each station in
    sequence along an ocean section.  The data can be from a single
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
    npts = len(lon)

    if geo_strf.ndim not in (1, 2):
        raise ValueError('geo_strf must be 1-D or 2-d; found shape %s'
                         % (geo_strf.shape,))

    laxis = 0 if axis else -1

    ds = distance(lon, lat, p)
    u = np.diff(geo_strf, axis=laxis) / ds

    mid_lon = 0.5 * (lon[:-1] + lon[1:])
    mid_lat = 0.5 * (lat[:-1] + lon[1:])

    return u, mid_lon, mid_lat
