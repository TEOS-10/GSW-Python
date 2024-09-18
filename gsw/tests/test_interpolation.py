import os

import numpy as np
from numpy.testing import assert_allclose

import gsw
from gsw._utilities import Bunch

root_path = os.path.abspath(os.path.dirname(__file__))

cv = Bunch(np.load(os.path.join(root_path, 'gsw_cv_v3_0.npz')))

def test_sa_ct_interp():
    p = cv.p_chck_cast
    CT = cv.CT_chck_cast
    SA = cv.SA_chck_cast
    p_i = np.repeat(cv.p_i[:, np.newaxis], p.shape[1], axis=1)
    SA_i, CT_i = gsw.sa_ct_interp(SA, CT, p, p_i)
    assert_allclose(SA_i, cv.SAi_SACTinterp, rtol=0, atol=cv.SAi_SACTinterp_ca)
    assert_allclose(CT_i, cv.CTi_SACTinterp, rtol=0, atol=cv.CTi_SACTinterp_ca)

def test_tracer_ct_interp():
    p = cv.p_chck_cast
    CT = cv.CT_chck_cast
    tracer = cv.SA_chck_cast
    p_i = np.repeat(cv.p_i[:, np.newaxis], p.shape[1], axis=1)
    tracer_i, CT_i = gsw.tracer_ct_interp(tracer, CT, p, p_i)
    assert_allclose(tracer_i, cv.traceri_tracerCTinterp, rtol=0, atol=cv.traceri_tracerCTinterp_ca)
    assert_allclose(CT_i, cv.CTi_SACTinterp, rtol=0, atol=cv.CTi_SACTinterp_ca)
