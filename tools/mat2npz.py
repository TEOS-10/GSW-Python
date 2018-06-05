#!/usr/bin/env python

from pathlib import Path

import numpy as np

import scipy.io as sio


def _structured_to_dict(arr):
    if arr.dtype.kind == 'V' and arr.shape == (1, 1):
        b = {}
        x = arr[0, 0]
        for name in x.dtype.names:
            b[name] = _structured_to_dict(x[name])
        return b
    return _crunch(arr)


def _crunch(arr):
    if arr.size == 1:
        arr = arr.item()
        return arr
    arr = arr.squeeze()
    return np.array(arr)


def loadmatdict(fname):
    out = {}
    with fname.open('rb') as fobj:
        xx = sio.loadmat(fobj)
        keys = [k for k in xx.keys() if not k.startswith('__')]
        for k in keys:
            out[k] = _structured_to_dict(xx[k])
    return out


# We can add our own data version later; the problem is that
# the original matfile from TEOS-10 has the same file name even
# as its contents change.
data_ver = 'v3_0'

gsw_data_file = Path('..', '..', 'GSW-Matlab/Toolbox/library/gsw_data_v3_0.mat')
gsw_data = loadmatdict(gsw_data_file)

# Save compare values `gsw_cv` in a separate file.
cv_vars = gsw_data['gsw_cv']
cv_vars['gsw_data_file'] = str(gsw_data_file)
fname = Path('..', 'gsw', 'tests', 'gsw_cv_%s' % data_ver)
np.savez(str(fname), **cv_vars)
