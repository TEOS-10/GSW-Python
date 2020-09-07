"""
Tests functions with xarray inputs.

This version is a copy of the original test_check_functions but with
an import of xarray, and conversion of the 3 main check cast arrays
into DataArray objects.

An additional xarray-dask test is added.
"""

import os
import pytest

import numpy as np
from numpy.testing import assert_allclose

import gsw
from gsw._utilities import Bunch
from check_functions import parse_check_functions

xr = pytest.importorskip('xarray')

# Most of the tests have some nan values, so we need to suppress the warning.
# Any more careful fix would likely require considerable effort.
np.seterr(invalid='ignore')

root_path = os.path.abspath(os.path.dirname(__file__))

# Function checks that we can't handle automatically yet.
blacklist = ['deltaSA_atlas',  # the test is complicated; doesn't fit the pattern.
             'geostrophic_velocity',  # test elsewhere; we changed the API
             #'CT_from_entropy', # needs prior entropy_from_CT; don't have it in C
             #'CT_first_derivatives', # passes, but has trouble in "details";
                                      # see check_functions.py
             #'entropy_second_derivatives', # OK now; handling extra parens.
             #'melting_ice_into_seawater',  # OK now; fixed nargs mismatch.
             ]

# We get an overflow from ct_from_enthalpy_exact, but the test passes.
cv = Bunch(np.load(os.path.join(root_path, 'gsw_cv_v3_0.npz')))

# Substitute new check values for the pchip interpolation version.
cv.geo_strf_dyn_height = np.load(os.path.join(root_path,'geo_strf_dyn_height.npy'))
cv.geo_strf_velocity = np.load(os.path.join(root_path,'geo_strf_velocity.npy'))

for name in ['SA_chck_cast', 't_chck_cast', 'p_chck_cast']:
    cv[name] = xr.DataArray(cv[name])

cf = Bunch()

d = dir(gsw)
funcnames = [name for name in d if '__' not in name]

mfuncs = parse_check_functions(os.path.join(root_path, 'gsw_check_functions_save.m'))
mfuncs = [mf for mf in mfuncs if mf.name in d and mf.name not in blacklist]
mfuncnames = [mf.name for mf in mfuncs]


@pytest.fixture(scope='session', params=mfuncs)
def cfcf(request):
    return cv, cf, request.param


def test_check_function(cfcf):
    cv, cf, mfunc = cfcf
    mfunc.run(locals())
    if mfunc.exception is not None or not mfunc.passed:
        print('\n', mfunc.name)
        print('  ', mfunc.runline)
        print('  ', mfunc.testline)
        if mfunc.exception is None:
            mfunc.exception = ValueError('Calculated values are different from the expected matlab results.')
        raise mfunc.exception
    else:
        print(mfunc.name)
        assert mfunc.passed


def test_dask_chunking():
    dsa = pytest.importorskip('dask.array')

    # define some input data
    shape = (100, 1000)
    chunks = (100, 200)
    sp = xr.DataArray(dsa.full(shape, 35., chunks=chunks), dims=['time', 'depth'])
    p = xr.DataArray(np.arange(shape[1]), dims=['depth'])
    lon = 0
    lat = 45

    sa = gsw.SA_from_SP(sp, p, lon, lat)
    sa_dask = sa.compute()

    sa_numpy = gsw.SA_from_SP(np.full(shape, 35.0), p.values, lon, lat)
    assert_allclose(sa_dask, sa_numpy)
