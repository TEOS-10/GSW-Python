import numpy as np
from numpy.testing import assert_almost_equal

import gsw


def test_gibbs_0():
    SA = np.array([35, 34])[:, np.newaxis]
    p = np.array([0, 1000])[np.newaxis, :]
    out = gsw.gibbs(0, 0, 0, SA, 15, p)
    expected = np.array([[-1624.830331610998, 8102.300078026374],
                         [-1694.143599586675, 8040.219363411558]])
    assert_almost_equal(out, expected)


def test_gibbs_1():
    ns = np.array([0, 1])[:, np.newaxis, np.newaxis]
    nt = np.array([0, 1])[np.newaxis, :, np.newaxis]
    np_ = np.array([0, 1])[np.newaxis, np.newaxis, :]
    out = gsw.gibbs(ns, nt, np_, 35, 15, 0)
    expected = np.array([[[-1624.830331610998, 9.748019262990162e-04],
                          [-213.3508284006898, 2.083180196723266e-07]],
                          [[70.39427245333731, -7.288938619915955e-07],
                           [0.590312110989411, 2.083180196723266e-07]]])
    expected[1, 1, 1] = np.nan  # Unless we add this case to GSW-C.
    print(out)
    assert_almost_equal(out, expected)


def test_gibbs_2():
    params = [
        (2, 0, 0, 35, 15, 0, 2.144088568168594),
        (0, 2, 0, 35, 15, 0, -13.86057508638656),
        (0, 0, 2, 35, 15, 0, -4.218331910346273e-13)
        ]
    for p in params:
        assert_almost_equal(gsw.gibbs(*p[:6]), p[6])


def test_gibbs_ice():
    out = gsw.gibbs_ice(1, 0, 0, [0, 100])
    expected = np.array([1220.788661299953, 1220.962914882458])
    assert_almost_equal(out, expected)


# Source, on an Intel Mac:
# octave:3> gsw_gibbs(0, 0, 0, 35, 15, 0)
# ans = -1624.830331610998
# octave:4> gsw_gibbs(0, 0, 0, 35, 15, 1000)
# ans = 8102.300078026374
# octave:5> gsw_gibbs(0, 0, 0, 34, 15, 0)
# ans = -1694.143599586675
# octave:6> gsw_gibbs(0, 0, 0, 34, 15, 1000)
# ans = 8040.219363411558


# octave:7> gsw_gibbs(1, 0, 0, 35, 15, 0)
# ans = 70.39427245333731
# octave:8> gsw_gibbs(0, 1, 0, 35, 15, 0)
# ans = -213.3508284006898
# octave:9> gsw_gibbs(0, 0, 1, 35, 15, 0)
# ans = 9.748019262990162e-04


# octave:10> gsw_gibbs(2, 0, 0, 35, 15, 0)
# ans = 2.144088568168594
# octave:11> gsw_gibbs(0, 2, 0, 35, 15, 0)
# ans = -13.86057508638656
# octave:12> gsw_gibbs(0, 0, 2, 35, 15, 0)
# ans = -4.218331910346273e-13

# octave:13> gsw_gibbs(1, 0, 1, 35, 15, 0)
# ans = -7.288938619915955e-07
# octave:14> gsw_gibbs(1, 1, 0, 35, 15, 0)
# ans = 0.590312110989411
# octave:15> gsw_gibbs(0, 1, 1, 35, 15, 0)
# ans = 2.083180196723266e-07
# octave:16>

# octave:16> gsw_gibbs(1, 1, 1, 35, 15, 0)
# ans = 1.420449745181019e-09

# octave:7> gsw_gibbs_ice(1, 0, 0, 0)
# ans = 1220.788661299953
# octave:8> gsw_gibbs_ice(1, 0, 0, 100)
# ans = 1220.962914882458