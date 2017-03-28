import re

from pycurrents.system import Bunch

from matlab_parser import get_complete_sigdict, get_helpdict
from parse_declarations import get_signatures, parse_signatures
from docstring_parts import parameters

wrapmod = 'gsw/_wrapped_ufuncs.py'

# Functions that are Matlab subroutines; we don't need to expose them.
blacklist = {'ct_freezing_exact',
'pt0_cold_ice_poly',
'pt_from_pot_enthalpy_ice_poly_dh',
't_freezing_exact',
}

with open('ufuncs.list') as f:
    ufunclist = [name.strip() for name in f.readlines()]
    ufunclist = [name for name in ufunclist if name not in blacklist]

msigdict = get_complete_sigdict()
csigdict = parse_signatures(get_signatures())

wrapper_head = '''
"""
Auto-generated wrapper for C ufunc extension; do not edit!
"""

#from ._wrapped_ufuncs import *
from . import _gsw_ufuncs
from ._utilities import match_args_return

'''

wrapper_template = '''
@match_args_return
def %(funcname)s(%(args)s):
    """
    %(doc)s
    """
    return _gsw_ufuncs.%(ufuncname)s(%(args)s)
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
    argset = set()
    for ufname in ufunclist:
        args = get_argnames(ufname)
        if args is not None:
            argset.update(args)
    return argset

def get_ufnames_by_arg():
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
    argset = set()
    for ufname in ufunclist:
        args = get_outnames(ufname)
        if args is not None:
            argset.update(args)
    return argset

def paragraphs(linelist):
    plist = []
    newlinelist = []
    for line in linelist:
        line = line.strip()
        if line:
            newlinelist.append(line)
        elif newlinelist:
            plist.append(newlinelist)
            newlinelist = []
    if newlinelist:
        plist.append(newlinelist)
    return plist

def fix_outputs_doc(lines):
    for i, line in enumerate(list(lines)):
        match = re.search(r'(.*)\[(.*)\]', line)
        if match is not None:
            units = match.group(2).strip()
            remainder = match.group(1).strip()
            lines[i] = remainder
        else:
            lines[i] = line.strip()

    outname, remainder = lines[0].split('=')
    outlines = ['%s : array-like, %s' % (outname.strip(), units)]
    # FIXME: we need to assemble the rest into a single line
    # and then break it into lines of appropriate length.
    for line in lines[1:]:
        outlines.append('    ' + line)
    return outlines

def uf_wrapper(ufname):
    argnames = get_argnames(ufname)
    argstr = ', '.join(argnames)
    msig = Bunch(msigdict[ufname])

    subs = dict(ufuncname=ufname,
                funcname=msig['name'],
                args=argstr,
                )
    helpdict = get_helpdict(msig['path'])
    try:
        desclist = paragraphs(helpdict['DESCRIPTION'])[0]
        doc = '\n    '.join(desclist)

        pdoclist = ['\n\n    Parameters\n    ----------']
        for arg in argnames:
            pdoclist.append('    %s : array-like' % arg)
            for line in parameters[arg].split('\n'):
                pdoclist.append("        %s" % line)
        doc = doc + '\n'.join(pdoclist) + '\n'

        if 'OUTPUT' in helpdict:
            outdoc = helpdict['OUTPUT']
            doc = doc + '\n    Returns\n    -------'
            # Temporary Q&D:
            outdoc = fix_outputs_doc(outdoc)
            for line in outdoc:
                doc = doc + '\n    %s' % line
            doc = doc + '\n'

    except KeyError:
        doc = "(no description available)"
    subs['doc'] = doc
    return wrapper_template % subs

if __name__ == '__main__':
    with open(wrapmod, 'w') as f:
        f.write(wrapper_head)
        for ufname in ufunclist:
            try:
                wrapped = uf_wrapper(ufname)
            except RuntimeError as err:
                print(ufname, err)
            if wrapped is None:
                print("failed:", ufname)
            else:
                f.write(wrapped)
