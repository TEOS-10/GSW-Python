import pytest

import numpy as np

import gsw

nx, ny, nz = 2, 3, 10
y = np.arange(nx*ny*nz, dtype=float).reshape((nx, ny, nz))
y += y**1.5
z = np.arange(nz, dtype=float)
z = np.broadcast_to(z, y.shape)
zn = z.copy()
zn[:, :, [0, -1]] = np.nan

xi_arraylist = [[0.5, 1.5], np.linspace(-1, z.max() + 10, 50)]

# Initial smoke test with small and large xi arrays.
@pytest.mark.parametrize("xi", xi_arraylist)
def test_in_range(xi):
    yi = gsw.pchip_interp(z, y, xi, axis=-1)
    assert yi.shape == (nx, ny, len(xi))


# Try with NaNs.
@pytest.mark.parametrize("xi", xi_arraylist)
def test_in_range_nan(xi):
    yi = gsw.pchip_interp(zn, y, xi, axis=-1)
    assert yi.shape == (nx, ny, len(xi))
