"""
Module with functions and script to generate tests directly
from our local version of gsw_check_functions.m.

Usage (run from this test directory):
    python check_functions.py

A primary use for this script is to see which functions are
missing from GSW-Python; they appear in the NameError category.

TypeError category can include functions that are incomplete
or otherwise not working correctly.

For functions that run but yield results that fail the check,
the error arrays are printed.

This can be improved--we should get less information from the
matlab script and more from our own functions.  We probably
should not need the matlab script at all, or maybe use it only
to extract the list of functions being tested in matlab.

This module is also imported by test_check_functions.py, which
is run by py.test.

"""

import os
import sys
import re

import numpy as np

from gsw import *
from gsw._utilities import Bunch

# If we switch to using the logging module, uncomment:
# import logging
# log = logging.getLogger()
# logging.basicConfig()

#  The following re patterns are for the "alternative" parsing of
#  the test line to support using numpy assert_allclose.  This is
#  not presently in use, but aspects of this method, here and in
#  _WIP_test_ufuncs.py, might replace some of the original code here.
#
# pattern for a single test line after it has been pre-processed to
# remove spaces, the square brackets, and semicolon
testlinepat = r"(\w+\.\w+)=find\(\w*\((\w+\.\w+)-(\w+\.\w+)\)>=(\w+\.\w+)\)"
#
# pattern for the inner test when there is a sequence separated by '|'
testpat = r"\(+\w*\((\w+\.\w+)-(\w+\.\w+)\)+>=(\w+\.\w+)\)"


def find(x):
    """
    Numpy equivalent to Matlab find.
    """
    return np.nonzero(np.asarray(x).flatten())[0]


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

        # alternative parsing of testline
        testline = self.testline.replace(' ', '')
        if '|' in testline:
            diffstr, test_str = testline.split('=', 1)
            # Chop off the leading 'find('.
            tests = test_str[5:].split('|')
            self.test_varstrings = [diffstr]
            for test in tests:
                m = re.match(testpat, test)
                if m is None:
                    print(self.name, testpat, test, m)
                if m is not None:
                    self.test_varstrings.extend(list(m.groups()))
        else:
            m = re.match(testlinepat, testline)
            if m is not None:
                self.test_varstrings =  m.groups()
            else:
                print("no match")
                self.test_varstrings = None


        # To be set when run() is successful
        self.outlist = None
        self.result = None   # will be a reference to the cv.I* array
        self.passed = None   # will be set to True or False

        # To be set if run() is not successful
        self.exception = None

    def __str__(self):
        return self.runline

    def record_details(self, evalargs):
        tline = self.testline
        i0 = 5 + tline.index('find(')
        tline = tline[i0:-1]
        checks = tline.split('|')
        parts = []
        for check in checks:
            check = check.replace(' ', '')
            if check.startswith('('):
                check = check[1:-1]
            part = Bunch(check=check)
            LHS, RHS = check.split('>=')
            part.tolerance = eval(RHS, *evalargs)
            # Sometimes there is an extra set of ().
            if LHS.startswith('('):
                LHS = LHS[1:-1]
            LHS = LHS[4:-1]  # chop off abs(...)
            target, calculated = LHS.split('-')
            part.checkval = eval(target, *evalargs)
            part.val = eval(calculated, *evalargs)
            parts.append(part)

        self.details = parts

    def run(self, locals=None):
        try:
            if locals is not None:
                _globals = globals() #dict(**globals())
                _globals.update(locals)
                evalargs = (_globals,)
            else:
                evalargs = tuple()

            # The following is needed for melting_ice_into_seawater.
            if len(self.outstrings) > 1:
                rl_ind = '[:%d]' % len(self.outstrings)
            else:
                rl_ind = ''

            exec(self.runline + rl_ind, *evalargs)
            if len(self.outstrings) == 1:
                if isinstance(eval(self.outstr, *evalargs), tuple):
                    exec("%s = %s[0]" % (self.outstr, self.outstr), *evalargs)
            self.outlist = [eval(s, *evalargs) for s in self.outstrings]

            exec(self.testline, *evalargs)
            self.result = eval(self.resultstr, *evalargs)

            self.passed = (len(self.result) == 0)
            # The following has trouble with CT_first_derivatives
            if self.name not in ['CT_first_derivatives',]:
                self.record_details(evalargs)
            # print("%s passed? %s" % (self.name, self.passed))

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
