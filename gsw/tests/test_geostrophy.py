import numpy as np

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
