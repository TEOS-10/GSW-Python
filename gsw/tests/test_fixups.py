"""
Tests of function modifications by _fixed_wrapped_ufuncs.py.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import gsw

# In the following pairs of z, p, the expected values for p are simply the
# values returned by z_from_p on an M4 Mac with gsw version
# 3.6.22.dev3+g2bad68c16 (prior to the commit in which this test is added).

zpvals_ok = [
    (-5500, 5609.875428946266),
    (
        np.linspace(-100, 0, 11),
        np.array(
            [
                100.70968879,
                90.6365065,
                80.56381587,
                70.49161697,
                60.41990988,
                50.34869469,
                40.27797148,
                30.20774032,
                20.1380013,
                10.0687545,
                -0.0,
            ]
        ),
    ),
    (np.nan, np.nan),
    ([np.nan, -100], np.array([np.nan, 100.70968879])),
    (np.ma.masked_invalid([np.nan, -100]), np.ma.masked_invalid([np.nan, 100.70968879])),
]

zvals_bad = [
    5500,
    np.linspace(0, 100, 11),
    [np.nan, 100],
    np.ma.masked_invalid([np.nan, 100]),
]


@pytest.mark.parametrize("zp", zpvals_ok)
def test_p_from_z_ok(zp):
    z, expected = zp
    p = gsw.p_from_z(z, 30)
    assert_allclose(p, expected)


@pytest.mark.parametrize("z", zvals_bad)
def test_p_from_z_bad(z):
    with pytest.raises(ValueError):
        gsw.p_from_z(z, 30)
