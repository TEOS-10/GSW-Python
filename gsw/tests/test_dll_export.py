import ctypes

import gsw


def test_ctypes_access():
    dllname = gsw._gsw_ufuncs.__file__
    gswlib = ctypes.cdll.LoadLibrary(dllname)
    rho_gsw_ctypes = gswlib.gsw_rho  # In-situ density.
    rho_gsw_ctypes.argtypes = [ctypes.c_double] * 3
    rho_gsw_ctypes.restype = ctypes.c_double
    stp = (35.0, 10.0, 0.0)
    assert rho_gsw_ctypes(*stp) == gsw.rho(*stp)
