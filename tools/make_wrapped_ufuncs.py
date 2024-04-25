"""
Script that generates _wrapped_ufuncs.py based on the output
of make_ufuncs.py.
"""

import sys
import re
from pathlib import Path

from _utilities import Bunch

from matlab_parser import get_complete_sigdict, get_helpdict
from c_header_parser import get_signatures, parse_signatures
from docstring_parts import parameters, return_overrides
from docstring_utils import (paragraphs,
                             fix_outputs_doc,
                             docstring_from_sections)

basedir = Path(__file__).parent.parent


# Functions that are Matlab subroutines, or exclusive to
# the C and not needed; we don't need to expose them.
blacklist = {'ct_freezing_exact',
'pt0_cold_ice_poly',
'pt_from_pot_enthalpy_ice_poly_dh',
't_freezing_exact',
}

wrapper_head = '''
"""
Auto-generated wrapper for C ufunc extension; do not edit!
"""

from . import _gsw_ufuncs
from ._utilities import match_args_return

'''

## Alternatives: The first was the original, but it did not provide a way to
# tell the decorator about the signature of the ufunc. The second solved that
# problem, but failed to provide the argument names for the signature in the
# help function and the ipython "?" functionality.

# wrapper_template = '''
# @match_args_return
# def %(funcname)s(%(args)s):
#     """%(doc)s
#     """
#     return _gsw_ufuncs.%(ufuncname)s(%(args)s)
# '''

# wrapper_template = """
# %(funcname)s = match_args_return(_gsw_ufuncs.%(ufuncname)s)
# %(funcname)s.__doc__ = '''%(doc)s
# '''
# """

# Make a Python function with the proper list of arguments; add the 'types'
# attribute for the use of the decorator; then use the decorator in its
# function form.
wrapper_template = '''
def %(funcname)s(%(args)s):
    """%(doc)s
    """
    return _gsw_ufuncs.%(ufuncname)s(%(args)s)
%(funcname)s.types = _gsw_ufuncs.%(ufuncname)s.types
%(funcname)s = match_args_return(%(funcname)s)
'''




def get_argnames(ufname):
    try:
        msig = Bunch(msigdict[ufname])
        csig = Bunch(csigdict[ufname])
    except KeyError:
        return None
    cnames = csig.argnames[:]
    mnames = msig.argnames[:]
    nc, nm = len(cnames), len(mnames)
    if nc < nm:
        print('%s: truncating argument list, %s, %s' % (
                ufname, cnames, mnames))
        mnames = mnames[:nc]

    argnames = []
    for ac, am in zip(cnames, mnames):
        if am == 'long':
            am = 'lon'
        if ac == am.lower():
            argnames.append(am)
        else:
            raise RuntimeError("arg mismatch: %s, %s" % (
                                csig.argnames, msig.argnames))
    return argnames

def get_argname_set():
    # This is not currently used internally.
    argset = set()
    for ufname in ufunclist:
        args = get_argnames(ufname)
        if args is not None:
            argset.update(args)
    return argset

def get_ufnames_by_arg():
    # This is not currently used internally.
    argdict = dict()
    for ufname in ufunclist:
        args = get_argnames(ufname)
        if args is None:
            continue
        for arg in args:
            if arg in argdict:
                argdict[arg].append(ufname)
            else:
                argdict[arg] = [ufname]
    return argdict


def get_outnames(ufname):
    # This is currently used only in get_outname_set, which is not used internally.
    try:
        msig = Bunch(msigdict[ufname])
    except KeyError:
        return None
    mnames = msig.outnames[:]

    outnames = []
    for am in mnames:
        if am == 'long':
            am = 'lon'
        outnames.append(am)
    return outnames

def get_outname_set():
    # This is not currently used internally.
    argset = set()
    for ufname in ufunclist:
        args = get_outnames(ufname)
        if args is not None:
            argset.update(args)
    return argset

def get_help_output_dict():
    # This is not currently used internally.
    out = Bunch()
    for ufname in ufunclist:
        msig = msigdict[ufname]
        helpdict = get_helpdict(msig['path'])

        if 'OUTPUT' in helpdict:
            raw = helpdict['OUTPUT']
            outdoc = fix_outputs_doc(raw)
        else:
            raw = ''
            outdoc = ['']
        if ufname in return_overrides:
            outdoc = return_overrides[ufname]
        out[ufname] = Bunch(raw=raw, outdoc=outdoc)
    return out


def uf_wrapper(ufname):
    argnames = get_argnames(ufname)
    argstr = ', '.join(argnames)
    msig = Bunch(msigdict[ufname])

    subs = dict(ufuncname=ufname,
                funcname=msig['name'],
                args=argstr,
                )
    helpdict = get_helpdict(msig['path'])

    # Filter out minimally documented library functions.
    if 'DESCRIPTION' not in helpdict:
        return None

    try:
        desclist = paragraphs(helpdict['DESCRIPTION'])[0]
        sections = dict(Head=desclist)
        plist = []
        for arg in argnames:
            plist.append('%s : array-like' % arg)
            for line in parameters[arg].split('\n'):
                plist.append("    %s" % line)
        sections['Parameters'] = plist

        # I think we can assume OUTPUT will be present, but just
        # in case, we check for it.  Maybe remove this later.
        if 'OUTPUT' in helpdict:
            outdoc = fix_outputs_doc(helpdict['OUTPUT'])
        else:
            outdoc = ['None']
        if ufname in return_overrides:
            outdoc = return_overrides[ufname]
        sections['Returns'] = outdoc
        doc = docstring_from_sections(sections)
    except KeyError as e:
        print("KeyError for %s, %s" % (ufname, e))
        doc = "(no description available)"
    subs['doc'] = doc
    return wrapper_template % subs

if __name__ == '__main__':
    srcdir = basedir.joinpath('src')
    with open(srcdir.joinpath('_ufuncs.list')) as f:
        ufunclist = [name.strip() for name in f.readlines()]
        ufunclist = [name for name in ufunclist if name not in blacklist]

    wrapmod = basedir.joinpath('gsw', '_wrapped_ufuncs.py')

    msigdict = get_complete_sigdict()
    csigdict = parse_signatures(get_signatures(srcdir=srcdir))

    wrapped_ufnames = []

    with wrapmod.open('w') as f:
        f.write(wrapper_head)
        for ufname in ufunclist:
            try:
                wrapped = uf_wrapper(ufname)
                if wrapped is None:
                    continue
            except RuntimeError as err:
                print(ufname, err)
            if wrapped is None:
                print("failed:", ufname)
            else:
                f.write(wrapped)
                wrapped_ufnames.append(ufname)
    wrapped_ufnames.sort()
    with open(srcdir.joinpath('_wrapped_ufuncs.list'), 'w') as f:
        f.write('\n'.join(wrapped_ufnames) + '\n')
