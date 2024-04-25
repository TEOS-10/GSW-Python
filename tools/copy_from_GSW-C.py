#!/usr/bin/env python
"""
Copy all relevant .c and .h files from Python-C, if they are newer.

This is a simple utility, to be run from this directory.  It assumes that
an up-to-date GSW-C GitHub repo and the present GSW-Python repo are
siblings in the directory tree.
"""

import sys
import shutil
from pathlib import Path

current = Path(__file__).parent
srcdir = Path(current.parent.parent, "GSW-C")

destdir = Path(current.parent, "src", "c_gsw")

fnames = [
    "gsw_oceanographic_toolbox.c",
    "gsw_saar.c",
    "gsw_saar_data.h",
    "gsw_internal_const.h",
    "gswteos-10.h",
    ]


if not srcdir.exists():
    raise IOError(
        f"Could not find the GSW-C source code in {srcdir}. "
        "Please read the development notes to find how to setup your GSW-Python development environment."
        )

for fname in fnames:
    src = srcdir.joinpath(fname)
    dest = destdir.joinpath(fname)
    if src.stat().st_mtime > dest.stat().st_mtime:
        shutil.copyfile(str(src), str(dest))
        print(f"copied {src} to {dest}")
