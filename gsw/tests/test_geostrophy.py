import numpy as np
from numpy.testing import assert_array_equal

import gsw

lon = [1, 2]
lat = [45, 45]
expected = 78626.18767687


def test_list():
    value = gsw.distance(lon, lat, p=0, axis=-1)
    np.testing.assert_almost_equal(expected, value)

def test_1darray():
    value = gsw.distance(np.array(lon), np.array(lat), p=0, axis=-1)
    np.testing.assert_almost_equal(expected, value)


def test_2dlist():
    value = gsw.distance(np.atleast_2d(lon), np.atleast_2d(lat), p=0, axis=1)
    np.testing.assert_almost_equal(expected, value)


def test_strf_no_good():
    # geo_strf_dyn_height requires more than 3 valid points
    shape = (5,)
    SA = np.ma.masked_all(shape, dtype=float)
    CT = np.ma.masked_all(shape, dtype=float)
    p = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    # No valid points.
    strf = gsw.geo_strf_dyn_height(SA, CT, p, p_ref=0, axis=0)
    expected = np.zeros(shape, float) + np.nan
    assert_array_equal(strf.filled(np.nan), expected)

    # 3 valid points: still not enough.
    SA[:3] = 35
    CT[:3] = [5, 4, 3]
    strf = gsw.geo_strf_dyn_height(SA, CT, p, p_ref=0, axis=0)
    expected = np.zeros(shape, float) + np.nan
    assert_array_equal(strf.filled(np.nan), expected)

    # 4 valid points: enough for the calculation to proceed.
    SA[:4] = 35
    CT[:4] = [5, 4, 3, 2]
    strf = gsw.geo_strf_dyn_height(SA, CT, p, p_ref=0, axis=0)
    assert strf.count() == 4
