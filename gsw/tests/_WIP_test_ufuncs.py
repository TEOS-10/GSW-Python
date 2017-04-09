"""
Tests for the unwrapped ufuncs.

This is a WIP; it doesn't work yet for all cases, and might not be a good
approach anyway.  For now, test_check_functions is adequate, handling the
wrapped ufuncs via check_functions "eval" and "exec" machinery.
"""
import pytest

import numpy as np
from numpy.testing import assert_allclose

import gsw
from gsw._utilities import Bunch
from gsw.tests.check_functions import parse_check_functions

cv = Bunch(np.load('gsw_cv_v3_0.npz'))
cf = Bunch()

d = dir(gsw._gsw_ufuncs)
funcnames = [name for name in d if '__' not in name]

mfuncs = parse_check_functions('gsw_check_functions_save.m')
mfuncs = [mf for mf in mfuncs if mf.name in d]
mfuncnames = [mf.name for mf in mfuncs]

@pytest.fixture(scope='session', params=mfuncs)
def cfcf(request):
    return cv, cf, request.param

def test_mechanism(cfcf):
    cv, cf, mfunc = cfcf
    print("<%s>" % mfunc.name)
    def value_from_name(vname):
        b, name = vname.split('.')
        if b == 'cf':
            return cf[name]
        elif b == 'cv':
            return cv[name]
        else:
            raise ValueError("Can't find cf. or cv. in %s" % vname)
    def set_from_name(vname, value):
        b, name = vname.split('.')
        if b == 'cf':
            cf[name] = value
        else:
            raise ValueError("attempting to set value in %s" % (b,))

    func = getattr(gsw._gsw_ufuncs, mfunc.name)
    args = [eval(a) for a in mfunc.argstrings]
    #print("<<%s>>" % (args,))
    out = func(*args)
    #print("<<<%s>>>" % (out,))
    if isinstance(out, tuple):
        nout = len(out)
    else:
        nout = 1
        out = (out,)
    n = min(nout, len(mfunc.outstrings))
    for i, s in enumerate(mfunc.outstrings[:n]):
        set_from_name(s, out[i])
    if mfunc.test_varstrings is not None:
        ntests = (len(mfunc.test_varstrings) - 1) // 3
        for i in range(ntests):
            expected = value_from_name(mfunc.test_varstrings[3*i+1])
            found = value_from_name(mfunc.test_varstrings[3*i+2])
            tolerance = value_from_name(mfunc.test_varstrings[3*i+3])
            #print(expected)
            #print(found)
            print(tolerance)
            try:
                assert_allclose(expected, found, atol=tolerance)
            except TypeError:
                print(mfunc.test_varstrings[3*i+3], tolerance.shape)
                print(mfunc.test_varstrings)
        # The following is not right, but this step is unimportant.
        #set_from_name(mfunc.test_varstrings[0], expected - found)

    else:
        print(">>%s<<" % mfunc.testline)
        print("missing mfunc.test_varstrings")
        mfunc.run()
        if hasattr(mfunc, 'exception'):
            print(">>>%s<<<", mfunc.exception)
        else:
            assert mfunc.passed
