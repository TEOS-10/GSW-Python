"""
Lists functions checked by test_check_functions, and functions
that are in gsw_check_functions but are not in gsw.
"""

import os

import numpy as np

import gsw
from check_functions import parse_check_functions

root_path = os.path.abspath(os.path.dirname(__file__))

# Function checks that we can't handle automatically yet.
blacklist = ['deltaSA_atlas',  # the test is complicated; doesn't fit the pattern.
             ]

d = dir(gsw)
funcnames = [name for name in d if '__' not in name]

mfuncs_all = parse_check_functions(os.path.join(root_path,
                                        'gsw_check_functions_save.m'))
mfuncs = [mf for mf in mfuncs_all if mf.name in d and mf.name not in blacklist]
mfuncnames = sorted([mf.name for mf in mfuncs])

missingnames = [mf for mf in mfuncs_all if mf.name not in d]
missingnames = sorted([mf.name for mf in missingnames])

print('Functions being checked:')
for i, name in enumerate(mfuncnames):
    print(i, name)

print('Functions not in gsw:')
for i, name in enumerate(missingnames):
    print(i, name)

