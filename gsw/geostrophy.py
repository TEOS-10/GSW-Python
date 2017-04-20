
import numpy as np

from . import _gsw_ufuncs
from ._utilities import match_args_return, indexer


@match_args_return
def geo_strf_dyn_height(SA, CT, p, p_ref=0, axis=0):
    p_ref = float(p_ref)
    if (np.diff(p, axis=axis) <= 0).any():
        raise ValueError('p must be increasing along the specified axis')
    SA, CT, p = np.broadcast_arrays(SA, CT, p)
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
