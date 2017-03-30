import numpy as np

from ._utilities import match_args_return, axis_slicer
from ._gsw_ufuncs import grav, specvol_alpha_beta

__all__ = ['Nsquared',]

@match_args_return
def Nsquared(SA, CT, p, lat=None, axis=-1):
    if lat is not None:
        if np.any((lat < -90) | (lat > 90)):
            raise ValueError('lat is out of range')
        SA, CT, p, lat = np.broadcast_arrays(SA, CT, p, lat)
        g = grav(lat, p)
    else:
        SA, CT, p = np.broadcast_arrays(SA, CT, p)
        g = 9.7963  # (Griffies, 2004)

    db_to_pa = 1e4
    shallow = axis_slicer(SA.ndim, slice(-1), axis)
    deep = axis_slicer(SA.ndim, slice(1, None), axis)
    if lat is not None:
        g_local = 0.5 * (g[shallow] + g[deep])
    else:
        g_local = g

    dSA = SA[deep] - SA[shallow]
    dCT = CT[deep] - CT[shallow]
    dp = p[deep] - p[shallow]
    SA_mid = 0.5 * (SA[shallow] + SA[deep])
    CT_mid = 0.5 * (CT[shallow] + CT[deep])
    p_mid = 0.5 * (p[shallow] + p[deep])

    specvol_mid, alpha_mid, beta_mid = specvol_alpha_beta(SA_mid,
                                                          CT_mid, p_mid)

    N2 = ((g_local**2) / (specvol_mid * db_to_pa * dp))
    N2 *= (beta_mid*dSA - alpha_mid*dCT)

    return N2, p_mid
