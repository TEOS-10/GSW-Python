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

# This is the data version designation used in the file name; but it
# is not a true version, because the file contents changes from one
# matlab release to another.
data_ver = 'v3_0'

# This is the version of the matlab zipfile from which we are getting
# the data file.
mat_zip_ver = 'v3_06_16'

# The following relative path will depend on the directory layout for
# whoever is running this utility.
basedir = Path(__file__).parent.parent
gsw_data_file = Path(
    basedir.parent,
    "GSW-Matlab",
    "Toolbox",
    "library",
    f"gsw_data_{data_ver}.mat",
    )
print(gsw_data_file)

gsw_data = loadmatdict(gsw_data_file)

# Save compare values `gsw_cv` in a separate file.
cv_vars = gsw_data['gsw_cv']
cv_vars['gsw_data_file'] = str(gsw_data_file)
cv_vars['mat_zip_ver'] = mat_zip_ver
fname = Path(basedir, "gsw", "tests", f"gsw_cv_{data_ver}")
np.savez(str(fname), **cv_vars)
