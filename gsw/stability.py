"""
Vertical stability functions.

These work with ndarrays of profiles; use the `axis` keyword
argument to specify the axis along which pressure varies.
For example, the default, following the Matlab versions, is
`axis=0`, meaning the pressure varies along the first dimension.
Use `axis=-1` if pressure varies along the last dimension--that
is, along a row, as the column index increases, in the 2-D case.

Docstrings will be added later, either manually or via
an automated mechanism.

"""


import numpy as np

from ._utilities import match_args_return, axis_slicer
from ._gsw_ufuncs import grav, specvol_alpha_beta

__all__ = ['Nsquared',
           'Turner_Rsubrho',
           'IPV_vs_fNsquared_ratio',
           ]

# In the following, axis=0 matches the Matlab behavior.

@match_args_return
def Nsquared(SA, CT, p, lat=None, axis=0):
    """
    Calculate the square of the buoyancy frequency.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lat : array-like, 1-D, optional
        Latitude, degrees.
    axis : int, optional
        The dimension along which pressure increases.

    Returns
    -------
    N2 : array
        Buoyancy frequency-squared at pressure midpoints, 1/s^2.
        The shape along the pressure axis dimension is one
        less than that of the inputs.
        (Frequency N is in radians per second.)
    p_mid : array
        Pressure at midpoints of p, dbar.
        The array shape matches N2.

    """
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


@match_args_return
def Turner_Rsubrho(SA, CT, p, axis=0):
    """
    Calculate the Turner Angle and the Stability Ratio.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    axis : int, optional
        The dimension along which pressure increases.

    Returns
    -------
    Tu : array
        Turner Angle at pressure midpoints, degrees.
        The shape along the pressure axis dimension is one
        less than that of the inputs.
    Rsubrho : array
        Stability Ratio, dimensionless.
        The shape matches Tu.
    p_mid : array
        Pressure at midpoints of p, dbar.
        The array shape matches Tu.

    """

    SA = np.clip(SA, 0, 50)
    SA, CT, p = np.broadcast_arrays(SA, CT, p)
    shallow = axis_slicer(SA.ndim, slice(-1), axis)
    deep = axis_slicer(SA.ndim, slice(1, None), axis)

    dSA = -SA[deep] + SA[shallow]
    dCT = -CT[deep] + CT[shallow]

    SA_mid = 0.5 * (SA[shallow] + SA[deep])
    CT_mid = 0.5 * (CT[shallow] + CT[deep])
    p_mid = 0.5 * (p[shallow] + p[deep])

    _, alpha, beta = specvol_alpha_beta(SA_mid, CT_mid, p_mid)

    Tu = np.arctan2((alpha*dCT + beta*dSA), (alpha*dCT - beta*dSA))
    Tu = np.degrees(Tu)

    igood = (dSA != 0)
    Rsubrho = np.zeros_like(dSA)
    Rsubrho.fill(np.nan)
    Rsubrho[igood] = (alpha[igood]*dCT[igood])/(beta[igood]*dSA[igood])

    return Tu, Rsubrho, p_mid


@match_args_return
def IPV_vs_fNsquared_ratio(SA, CT, p, p_ref=0, axis=0):
    """
    Calculates the ratio of the vertical gradient of potential density to
    the vertical gradient of locally-referenced potential density.  This
    is also the ratio of the planetary Isopycnal Potential Vorticity
    (IPV) to f times N^2, hence the name for this variable,
    IPV_vs_fNsquared_ratio (see Eqn. (3.20.17) of IOC et al. (2010)).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    p_ref : float
        Reference pressure, dbar

    Returns
    -------
    IPV_vs_fNsquared_ratio : array
        The ratio of the vertical gradient of
        potential density referenced to p_ref, to the vertical
        gradient of locally-referenced potential density, dimensionless.
    p_mid : array
        Pressure at midpoints of p, dbar.
        The array shape matches IPV_vs_fNsquared_ratio.

    """

    SA = np.clip(SA, 0, 50)
    SA, CT, p = np.broadcast_arrays(SA, CT, p)
    shallow = axis_slicer(SA.ndim, slice(-1), axis)
    deep = axis_slicer(SA.ndim, slice(1, None), axis)

    dSA = -SA[deep] + SA[shallow]
    dCT = -CT[deep] + CT[shallow]

    SA_mid = 0.5 * (SA[shallow] + SA[deep])
    CT_mid = 0.5 * (CT[shallow] + CT[deep])
    p_mid = 0.5 * (p[shallow] + p[deep])

    _, alpha, beta = specvol_alpha_beta(SA_mid, CT_mid, p_mid)
    _, alpha_pref, beta_pref = specvol_alpha_beta(SA_mid, CT_mid, p_ref)

    num = dCT*alpha_pref - dSA*beta_pref
    den = dCT*alpha - dSA*beta

    igood = (den != 0)
    IPV_vs_fNsquared_ratio = np.zeros_like(dSA)
    IPV_vs_fNsquared_ratio.fill(np.nan)
    IPV_vs_fNsquared_ratio[igood] = num[igood] / den[igood]

    return IPV_vs_fNsquared_ratio, p_mid
