"""
Functions not specific to the TEOS-10 realm of variables.
"""

import numpy as np

from . import _gsw_ufuncs
from ._utilities import match_args_return, indexer

@match_args_return
def pchip_interp(x, y, xi, axis=0):
    """
    Interpolate using Piecewise Cubic Hermite Interpolating Polynomial

    This is a shape-preserving algorithm; it does not introduce new local
    extrema.  The implementation in C that is wrapped here is largely taken
    from the scipy implementation,
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html.

    Points outside the range of the interpolation table are filled using the
    end values in the table.  (In contrast,
    scipy.interpolate.pchip_interpolate() extrapolates using the end
    polynomials.)

    Parameters
    ----------
    x, y : array-like
        Interpolation table x and y; n-dimensional, must be broadcastable to
        the same dimensions.
    xi : array-like
        One-dimensional array of new x values.
    axis : int, optional, default is 0
        Axis along which xi is taken.

    Returns
    -------
    yi : array
        Values of y interpolated to xi along the specified axis.

    """

    xi = np.array(xi, dtype=float, copy=False, order='C', ndmin=1)
    if xi.ndim > 1:
        raise ValueError('xi must be no more than 1-dimensional')
    nxi = xi.size
    x, y = np.broadcast_arrays(x, y)
    shape0 = x.shape
    out_shape = list(x.shape)
    out_shape[axis] = nxi
    yi = np.empty(out_shape, dtype=float)
    yi.fill(np.nan)

    goodmask = ~(np.isnan(x) | np.isnan(y))

    order = 'F' if y.flags.fortran else 'C'
    for ind in indexer(y.shape, axis, order=order):
        igood = goodmask[ind]
        # If p_ref is below the deepest value, skip the profile.
        xgood = x[ind][igood]
        ygood = y[ind][igood]

        yi[ind] = _gsw_ufuncs.util_pchip_interp(xgood, ygood, xi)

    return yi
