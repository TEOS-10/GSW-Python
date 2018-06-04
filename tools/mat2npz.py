#!/usr/bin/env python

from pathlib import Path

import numpy as np

from pycurrents.file.matfile import loadmatbunch

# We can add our own data version later; the problem is that
# the original matfile from TEOS-10 has the same file name even
# as its contents change.
data_ver = 'v3_0'

gsw_data_file = str(Path('..', '..', 'GSW-Matlab/Toolbox/library/gsw_data_v3_0.mat'))

gsw_data = loadmatbunch(gsw_data_file, masked=False)

# Save compare values `gsw_cv` in a separate file.
cv_vars = gsw_data['gsw_cv']
cv_vars.gsw_data_file = gsw_data_file
fname = Path('..', 'gsw', 'tests', 'gsw_cv_%s' % data_ver)
np.savez(fname, **cv_vars)
