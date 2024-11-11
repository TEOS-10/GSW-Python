"""
Functions for vertical interpolation.
"""

import numpy as np

from . import _gsw_ufuncs
from ._utilities import indexer, match_args_return

__all__ = ['sa_ct_interp',
           'tracer_ct_interp',
           ]

@match_args_return
def sa_ct_interp(SA, CT, p, p_i, axis=0):
    """
    Interpolates vertical casts of values of Absolute Salinity
    and Conservative Temperature to the arbitrary pressures p_i.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    p_i : array-like
        Sea pressure to interpolate on, dbar
    axis : int, optional, default is 0
        The index of the pressure dimension in SA and CT.


    Returns
    -------
    SA_i : array
        Values of SA interpolated to p_i along the specified axis.
    CT_i : array
        Values of CT interpolated to p_i along the specified axis.

    """
    if SA.shape != CT.shape:
        raise ValueError(f'Shapes of SA and CT must match; found {SA.shape} and {CT.shape}')
    if p.ndim != p_i.ndim:
        raise ValueError(f'p and p_i must have the same number of dimensions;\n'
                         f' found {p.ndim} versus {p_i.ndim}')
    if p.ndim == 1 and SA.ndim > 1:
        if len(p) != SA.shape[axis]:
            raise ValueError(
                f'With 1-D p, len(p) must be SA.shape[axis];\n'
                f' found {len(p)} versus {SA.shape[axis]} on specified axis, {axis}'
                )
        ind = [np.newaxis] * SA.ndim
        ind[axis] = slice(None)
        p = p[tuple(ind)]
        p_i = p_i[tuple(ind)]
    elif p.ndim > 1:
        if p.shape != SA.shape:
            raise ValueError(f'With {p.ndim}-D p, shapes of p and SA must match;\n'
                             f'found {p.shape} and {SA.shape}')
        if any(p.shape[i] != p_i.shape[i] for i in range(p.ndim) if i != axis):
            raise ValueError(f'With {p.ndim}-D p, p and p_i must have the same dimensions outside of axis {axis};\n'
                             f' found {p.shape} versus {p_i.shape}')
    with np.errstate(invalid='ignore'):
        # The need for this context seems to be a bug in np.ma.any.
        if np.ma.any(np.ma.diff(np.ma.masked_invalid(p_i), axis=axis) <= 0) \
                or np.ma.any(np.ma.diff(np.ma.masked_invalid(p), axis=axis) <= 0):
            raise ValueError('p and p_i must be increasing along the specified axis')
    p = np.broadcast_to(p, SA.shape)
    goodmask = ~(np.isnan(SA) | np.isnan(CT) | np.isnan(p))
    SA_i = np.empty(p_i.shape, dtype=float)
    CT_i = np.empty(p_i.shape, dtype=float)
    SA_i.fill(np.nan)
    CT_i.fill(np.nan)

    try:
        order = 'F' if SA.flags.fortran else 'C'
    except AttributeError:
        order = 'C'  # e.g., xarray DataArray doesn't have flags
    for ind in indexer(SA.shape, axis, order=order):
        # this is needed to support xarray inputs for numpy < 1.23
        igood = np.asarray(goodmask[ind])
        pgood = p[ind][igood]
        pi = p_i[ind]
        # There must be at least 2 non-NaN values for interpolation
        if len(pgood) > 2:
            sa = SA[ind][igood]
            ct = CT[ind][igood]
            sai, cti = _gsw_ufuncs.sa_ct_interp(sa, ct, pgood, pi)
            SA_i[ind] = sai
            CT_i[ind] = cti

    return (SA_i, CT_i)

@match_args_return
def tracer_ct_interp(tracer, CT, p, p_i, factor=9., axis=0):
    """
    Interpolates vertical casts of values of a tracer
    and Conservative Temperature to the arbitrary pressures p_i.

    Parameters
    ----------
    tracer : array-like
        tracer
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    p_i : array-like
        Sea pressure to interpolate on, dbar
    factor: float, optional, default is 9.
        Ratio between the ranges of Conservative Temperature
        and tracer in the world ocean.
    axis : int, optional, default is 0
        The index of the pressure dimension in tracer and CT.


    Returns
    -------
    tracer_i : array
        Values of tracer interpolated to p_i along the specified axis.
    CT_i : array
        Values of CT interpolated to p_i along the specified axis.

    """
    if tracer.shape != CT.shape:
        raise ValueError(f'Shapes of tracer and CT must match; found {tracer.shape} and {CT.shape}')
    if p.ndim != p_i.ndim:
        raise ValueError(f'p and p_i must have the same number of dimensions;\n'
                         f' found {p.ndim} versus {p_i.ndim}')
    if p.ndim == 1 and tracer.ndim > 1:
        if len(p) != tracer.shape[axis]:
            raise ValueError(
                f'With 1-D p, len(p) must be tracer.shape[axis];\n'
                f' found {len(p)} versus {tracer.shape[axis]} on specified axis, {axis}'
                )
        ind = [np.newaxis] * tracer.ndim
        ind[axis] = slice(None)
        p = p[tuple(ind)]
        p_i = p_i[tuple(ind)]
    elif p.ndim > 1:
        if p.shape != tracer.shape:
            raise ValueError(f'With {p.ndim}-D p, shapes of p and tracer must match;\n'
                             f'found {p.shape} and {tracer.shape}')
        if any(p.shape[i] != p_i.shape[i] for i in range(p.ndim) if i != axis):
            raise ValueError(f'With {p.ndim}-D p, p and p_i must have the same dimensions outside of axis {axis};\n'
                             f' found {p.shape} versus {p_i.shape}')
    with np.errstate(invalid='ignore'):
        # The need for this context seems to be a bug in np.ma.any.
        if np.ma.any(np.ma.diff(np.ma.masked_invalid(p_i), axis=axis) <= 0) \
                or np.ma.any(np.ma.diff(np.ma.masked_invalid(p), axis=axis) <= 0):
            raise ValueError('p and p_i must be increasing along the specified axis')
    p = np.broadcast_to(p, tracer.shape)
    goodmask = ~(np.isnan(tracer) | np.isnan(CT) | np.isnan(p))
    tracer_i = np.empty(p_i.shape, dtype=float)
    CT_i = np.empty(p_i.shape, dtype=float)
    tracer_i.fill(np.nan)
    CT_i.fill(np.nan)

    try:
        order = 'F' if tracer.flags.fortran else 'C'
    except AttributeError:
        order = 'C'  # e.g., xarray DataArray doesn't have flags
    for ind in indexer(tracer.shape, axis, order=order):
        # this is needed to support xarray inputs for numpy < 1.23
        igood = np.asarray(goodmask[ind])
        pgood = p[ind][igood]
        pi = p_i[ind]
        # There must be at least 2 non-NaN values for interpolation
        if len(pgood) > 2:
            tr = tracer[ind][igood]
            ct = CT[ind][igood]
            tri, cti = _gsw_ufuncs.tracer_ct_interp(tr, ct, pgood, pi, factor)
            tracer_i[ind] = tri
            CT_i[ind] = cti

    return (tracer_i, CT_i)
