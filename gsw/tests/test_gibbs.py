import numpy as np
from numpy.testing import assert_almost_equal
from gsw import gibbs

saref = 35.
ctref = 15.
pref = 0.


def test_gibbs():
    value = -1624.8303316
    assert_almost_equal(gibbs(0, 0, 0, saref, ctref, pref), value)


def test_array():
    value = -1624.8303316
    shape = (4, 5, 3)
    sa = saref*np.ones(shape)
    ct = ctref*np.ones(shape)
    p = pref*np.ones(shape)

    res = gibbs(1, 1, 0, sa, ct, p)
    assert res.shape == shape


def test_gibbsderivative():
    value = 0.590312110989
    assert_almost_equal(gibbs(1, 1, 0, saref, ctref, pref), value)
