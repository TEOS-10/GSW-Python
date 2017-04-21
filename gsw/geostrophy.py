
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
