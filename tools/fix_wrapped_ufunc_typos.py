#!/usr/bin/env python
"""
Run this script from this directory after running 'make_wrapped_ufuncs.py'.
It will correct common typos in the docstrings.  The original
_wrapped_ufuncs.py is copied to _wrapped_ufuncs.orig, which can be deleted.

"""

from pathlib import Path
import shutil

basedir = Path('..').resolve()
wrapmod = basedir.joinpath('gsw', '_wrapped_ufuncs.py')

orig = wrapmod.with_suffix('.orig')
shutil.copyfile(wrapmod, orig)

subs = [
    (' the the ', ' the '),
    ('fomula', 'formula'),
    (' caclulated ', ' calculated '),
    (' occuring ', ' occurring '),
    (' thoughout ', ' throughout '),
    (' orignal ', ' original '),
    ('proceedure', 'procedure' ),
    (' appropiate ', ' appropriate '),
    (' subracted ', ' subtracted '),
    (' anomally ', ' anomaly '),
    (' frist ', ' first '),
    (' calulated ', ' calculated '),
    (' outout ', ' output '),
    (' degress ', ' degrees ')
]

with open(wrapmod) as f:
    bigstring = f.read()

for bad, good in subs:
    bigstring = bigstring.replace(bad, good)

with open(wrapmod, "w") as f:
    f.write(bigstring)

