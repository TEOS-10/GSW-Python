#!/usr/bin/env python
"""
Run this script from this directory after running 'make_wrapped_ufuncs.py'.
It will correct common typos in the docstrings.  The original
_wrapped_ufuncs.py is copied to _wrapped_ufuncs.orig, which can be deleted.

"""

from pathlib import Path
import shutil

basedir = Path(__file__).parent.parent
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
    (' degress ', ' degrees '),
    (' specifc ', ' specific '),
    (' avaialble ', ' available '),
    (' equlibrium ', ' equilibrium '),
    ('equlibrium', 'equilibrium'),
    (' apendix ', ' appendix '),
    (' slighty ', ' slightly '),
    ('rho : array-like, kg/m', 'rho : array-like, kg/m^3'),
    ('http://www.TEOS-10.org', 'https://www.teos-10.org/'),
    ('http://www.ocean-sci.net/8/1117/2012/os-8-1117-2012.pdf', 'https://os.copernicus.org/articles/8/1117/2012/os-8-1117-2012.pdf'),
    ('http://www.ocean-sci.net/6/3/2010/os-6-3-2010.pdf', 'https://os.copernicus.org/articles/6/3/2010/os-6-3-2010.pdf'),
    ('http://www.ocean-sci.net/7/363/2011/os-7-363-2011.pdf', 'https://os.copernicus.org/articles/7/363/2011/os-7-363-2011.pdf'),
    ('http://www.ocean-sci.net/8/1123/2012/os-8-1123-2012.pdf', 'https://os.copernicus.org/articles/8/1123/2012/os-8-1123-2012.pdf'),
    ('http://www.iapws.org', 'https://iapws.org/'),

]

with open(wrapmod) as f:
    bigstring = f.read()

for bad, good in subs:
    bigstring = bigstring.replace(bad, good)

with open(wrapmod, "w") as f:
    f.write(bigstring)

