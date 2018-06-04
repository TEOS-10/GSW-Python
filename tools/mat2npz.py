#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# mat2npz.py
#
# purpose:  Convert matlab file from TEOS-10 group to a npz file
# author:   Filipe P. A. Fernandes
# e-mail:   ocefpaf@gmail
# web:      http://ocefpaf.tiddlyspot.com/
# created:  06-Jun-2011
# modified: Mon 16 Sep 2013 01:40:54 PM BRT
#
# obs:
#

import os

import numpy as np

from pycurrents.file.matfile import loadmatbunch

# We can add our own data version later; the problem is that
# the original matfile from TEOS-10 has the same file name even
# as its contents change.
data_ver = 'v3_0'

# gsw_data_file = '../../../gsw_matlab_v3_05_8/library/gsw_data_v3_0.mat'
gsw_data_file = '../../GSW-Matlab/Toolbox/library/gsw_data_v3_0.mat'

gsw_data = loadmatbunch(gsw_data_file, masked=False)

# Save compare values `gsw_cv` in a separate file.
cv_vars = gsw_data['gsw_cv']
cv_vars.gsw_data_file = gsw_data_file
np.savez("../gsw/tests/gsw_cv_%s" % data_ver, **cv_vars)
