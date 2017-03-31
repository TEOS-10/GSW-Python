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

from __future__ import print_function
import numpy as np

from pycurrents.file.matfile import loadmatbunch

# We can add our own data version later; the problem is that
# the original matfile from TEOS-10 has the same file name even
# as its contents change.
data_ver = 'v3_0'
gsw_data = loadmatbunch('gsw_cv_cf.mat', masked=False)

# Save compare values `gsw_cv` in a separate file.
cv_vars = gsw_data['gsw_cv']
np.savez("gsw_cv_%s" % data_ver, **cv_vars)
cf_vars = gsw_data['gsw_cf']
np.savez("gsw_cf_%s" % data_ver, **cv_vars)
