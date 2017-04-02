"""
Script to generate tests directly from our local version of
gsw_check_functions.m.

Usage (run from this test directory):
    python check_functions.py

A primary use for this script is to see which functions are
missing from python-gsw.

The results are summarized at the end.  The NameError category
includes functions that are missing entirely from python-gsw;
the TypeError category can include functions that are incomplete
or otherwise not functioning.

For functions that run but yield results that fail the check,
the error arrays are printed.

This can be improved--we should get less information from the
matlab script and more from our own functions.  We probably
should not need the matlab script at all, or maybe use it only
to extract the list of functions being tested in matlab.
"""

from __future__ import print_function
import os
import sys

import numpy as np

#import gsw
#from gsw.gibbs import *
#from gsw.utilities import Bunch

from gsw import *
from gsw._utilities import Bunch

# If we switch to using the logging module, uncomment:
# import logging
# log = logging.getLogger()
# logging.basicConfig()

# There is probably a better way to handle the "invalid value"
# problem, but for now we will ignore it.  (Nans in the test arrays...)
np.seterr(invalid='ignore')

# There are also divide-by-zero warnings, which we will leave in
# place. (library.py lines 1238, 1239; maybe something should be
# done to block this, so we don't get inf values.)


def find(x):
    """
    Numpy equivalent to Matlab find.
    """
    return np.nonzero(x.flatten())[0]


def group_or(line):
    """
    Translate matlab 'find' functions including logical or operators.

    Example: the test for gsw_rho_alpha_beta

    Numpy wart: using bitwise or as a fake elementwise logical or,
    we need to add parentheses.
    """
    if not ('find(' in line and '|' in line):
        return line
    i0 = line.index('find(') + 5
    head = line[:i0]
    tail = line[i0:]
    parts = tail.replace('|', ') | (')
    new = head + '(' + parts + ')'
    return new


class FunctionCheck(object):
    """
    Parse the line-pair for checks in gsw_check_functions.
    """
    def __init__(self, linepair):
        """
        *linepair* is the sequence of two lines; the first runs
        the function and assigns the output, and the second
        generates an array of indices where the output error
        exceeds a tolerance.
        """

        self.linepair = linepair
        self.runline = linepair[0]
        self.testline = linepair[1]

        # parse the line that runs the function
        head, tail = self.runline.split('=')
        self.outstrings = [s.strip() for s in head.split(',')]
        self.outstr = ','.join(self.outstrings)
        funcstr, argpart = tail.split('(', 1)
        self.name = funcstr.strip()
        self.argstrings = [s.strip() for s in argpart[:-1].split(',')]
        self.argstr = ','.join(self.argstrings)

        # parse the line that checks the results
        head, tail = self.testline.split('=', 1)
        self.resultstr = head.strip()  # cv.I*
        head, tail = tail.split('(', 1)
        self.teststr = tail.strip()[:-1]   # argument of "find()"
        self.teststr = self.teststr.replace('abs(', 'np.abs(')

        # To be set when run() is successful
        self.outlist = None
        self.result = None   # will be a reference to the cv.I* array
        self.passed = None   # will be set to True or False

        # To be set if run() is not successful
        self.exception = None

    def __str__(self):
        return self.runline

    def record_details(self):
        tline = self.testline
        i0 = 5 + tline.index('find(')
        tline = tline[i0:-1]
        checks = tline.split('|')
        parts = []
        for check in checks:
            check = check.strip()
            if check.startswith('('):
                check = check[1:-1].strip()
            part = Bunch(check=check)
            LHS, RHS = check.split('>=')
            part.tolerance = eval(RHS)
            LHS = LHS.strip()[4:-1]  # chop off abs(...)
            target, calculated = LHS.split('-')
            part.checkval = eval(target)
            part.val = eval(calculated)
            parts.append(part)

        self.details = parts

    def run(self):
        try:
            exec(self.runline)
            # In Matlab, the number of output arguments varies
            # depending on the LHS of the assignment, but Python
            # always returns the full set.  Here we handle the
            # case where Python is returning 2 (or more) but
            # the LHS is assigning only the first.
            if len(self.outstrings) == 1:
                if isinstance(eval(self.outstr), tuple):
                    exec("%s = %s[0]" % (self.outstr, self.outstr))
            self.outlist = [eval(s) for s in self.outstrings]
            exec(self.testline)
            self.result = eval(self.resultstr)
            self.passed = len(self.result) == 0
            self.record_details()

        except Exception as e:
            self.exception = e


def find_arguments(checks):
    """
    For a sequence of FunctionCheck instances, return the
    set of unique arguments as a sorted list.
    """
    argset = set()
    for c in checks:
        argset.update(c.argstrings)
    argsetlist = list(argset)
    argsetlist.sort()
    return argsetlist


def find_arglists(checks):
    """
    For a sequence of FunctionCheck instances, return the
    set of unique argument lists as a sorted list.
    """
    alset = set()
    for c in checks:
        alset.update([c.argstr])
    arglists = list(alset)
    arglists.sort()
    return arglists

def parse_check_functions(mfile):
    """
    Return a list of FunctionCheck instances from gsw_check_functions.m
    """

    with open(mfile, 'rt') as fid:
        mfilelines = fid.readlines()

    first_pass = []

    concat = False
    for line in mfilelines:
        line = line.strip()
        if concat:
            if line.endswith('...'):
                line = line[:-3]
            first_pass[-1] += line
            if line.endswith(';'):
                concat = False
            continue
        if '=' in line and (line.startswith('gsw_') or line.startswith('[gsw_')):
            if line.endswith('...'):
                line = line[:-3]
                concat = True
            first_pass.append(line)

    second_pass = []

    for line in first_pass:
        if not '(' in line:
            continue
        if 'which' in line:
            continue
        line = line.replace('gsw_', '')
        if line.startswith('['):
            line = line[1:].replace(']', '')
        if line.endswith(';'):
            line = line[:-1]
        line = line.replace('(I)', '')  # For deltaSA_atlas.
        second_pass.append(line)

    pairs = []

    for i in range(len(second_pass)):
        if 'find(' in second_pass[i] and not 'find(' in second_pass[i-1]:
            pairs.extend(second_pass[i-1:i+1])

    final = [group_or(line) for line in pairs]

    checks = []
    for i in range(0, len(final), 2):
        pair = final[i:i+2]
        checks.append(FunctionCheck(pair))

    return checks

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                description='Run checks from gsw_check_functions.m')

    parser.add_argument('--path', dest='mfiledir',
                        default="",
                       help='path to external gsw_check_functions.m')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='print output mismatch arrays')
    parser.add_argument('--find',
                        help='run functions with this substring')

    args = parser.parse_args()

    if args.mfiledir:
        mfile = os.path.join(args.mfiledir, "gsw_check_functions.m")
    else:
        mfile = "gsw_check_functions_save.m"
    checks = parse_check_functions(mfile)

    #datadir = os.path.join(os.path.dirname(gsw.utilities.__file__), 'data')
    datadir = './'
    cv = Bunch(np.load(os.path.join(datadir, 'gsw_cv_v3_0.npz')))
    cf = Bunch()

    if args.find:
        checks = [c for c in checks if args.find in c.runline]

    for fc in checks:
        fc.run()

    passes = [f for f in checks if f.passed]
    failures = [f for f in checks if f.passed is False]

    run_problems = [f for f in checks if f.exception is not None]

    etypes = [NameError, UnboundLocalError, TypeError, AttributeError]
    ex_dict = dict()
    for exc in etypes:
        elist = [(f.name, f.exception) for f in checks if
                 isinstance(f.exception, exc)]
        ex_dict[exc] = elist

    print("\n%s tests were translated from gsw_check_functions.m" % len(checks))
    print("\n%s tests ran with no error and with correct output" % len(passes))
    if args.verbose:
        for f in passes:
            print(f.name)

    print("\n%s tests had an output mismatch:" % len(failures))
    for f in failures:
        print(f.name)
        print(f.runline)
        print(f.testline)
        if args.verbose:
            print(f.result)
            for part in f.details:
                print("tolerance: ", part.tolerance)
                print("error:")
                print(part.checkval - part.val)
                print('')

        print('')

    print("\n%s exceptions were raised as follows:" % len(run_problems))
    for exc in etypes:
        print("  ", exc.__name__)
        strings = ["     %s : %s" % e for e in ex_dict[exc]]
        print("\n".join(strings))
        print("")

    checkbunch = Bunch([(c.name, c) for c in checks])
