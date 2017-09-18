import os

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

import gsw
from gsw._utilities import Bunch

root_path = os.path.abspath(os.path.dirname(__file__))

cv = Bunch(np.load(os.path.join(root_path, 'gsw_cv_v3_0.npz')))
# Override the original with what we calculate using pchip interp
cv.geo_strf_velocity = np.load(os.path.join(root_path,'geo_strf_velocity.npy'))

lon = [1, 2]
lat = [45, 45]
expected = 78626.18767687

# distance tests

def test_list():
    value = gsw.distance(lon, lat, p=0, axis=-1)
    assert_almost_equal(expected, value)

def test_1darray():
    value = gsw.distance(np.array(lon), np.array(lat), p=0, axis=-1)
    assert_almost_equal(expected, value)


def test_2dlist():
    value = gsw.distance(np.atleast_2d(lon), np.atleast_2d(lat), p=0, axis=1)
    assert_almost_equal(expected, value)

# geostrophic streamfunction tests

def test_strf_no_good():
    # revised geo_strf_dyn_height requires 2 valid points
    shape = (5,)
    SA = np.ma.masked_all(shape, dtype=float)
    CT = np.ma.masked_all(shape, dtype=float)
    p = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    # No valid points.
    strf = gsw.geo_strf_dyn_height(SA, CT, p, p_ref=0, axis=0)
    expected = np.zeros(shape, float) + np.nan
    assert_array_equal(strf.filled(np.nan), expected)

    # 1 valid point: not enough.
    SA[:1] = 35
    CT[:1] = 5
    strf = gsw.geo_strf_dyn_height(SA, CT, p, p_ref=0, axis=0)
    expected = np.zeros(shape, float) + np.nan
    assert_array_equal(strf.filled(np.nan), expected)

    # 2 valid points: enough for the calculation to proceed.
    SA[:2] = 35
    CT[:2] = [5, 4]
    strf = gsw.geo_strf_dyn_height(SA, CT, p, p_ref=0, axis=0)
    assert strf.count() == 2

def test_geostrophy():
    lon = cv.long_chck_cast
    lat = cv.lat_chck_cast
    p = cv.p_chck_cast
    CT = cv.CT_chck_cast
    SA = cv.SA_chck_cast
    strf = gsw.geo_strf_dyn_height(SA, CT, p)
    geovel, midlon, midlat = gsw.geostrophic_velocity(strf, lon, lat)
    assert_almost_equal(geovel, cv.geo_strf_velocity)
    assert_almost_equal(midlon, cv.geo_strf_velocity_mid_long[0])
    assert_almost_equal(midlat, cv.geo_strf_velocity_mid_lat[0])
