#!/usr/bin/env python
"""
Copy all relevant .c and .h files from Python-C, if they are newer.

This is a simple utility, to be run from this directory.  It assumes that
an up-to-date GSW-C github repo and the present GSW-Python repo are
siblings in the directory tree.
"""

import shutil
from pathlib import Path


fnames = ['gsw_oceanographic_toolbox.c',
          'gsw_saar.c',
          'gsw_saar_data.c',
          'gsw_internal_const.h',
          'gswteos-10.h']

srcdir = Path('..', '..', 'GSW-C')
destdir = Path('..', 'src', 'c_gsw')

for fname in fnames:
    src = srcdir.joinpath(fname)
    dest = destdir.joinpath(fname)
    if src.stat().st_mtime > dest.stat().st_mtime:
        shutil.copyfile(str(src), str(dest))
        print('copied %s to %s' % (src, dest))
