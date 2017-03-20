
from pycurrents.system import Bunch

from matlab_parser import get_complete_sigdict, get_helpdict
from parse_declarations import get_signatures, parse_signatures

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

'''

wrapper_template = '''
# Maybe a decorator will go here...
def %(funcname)s(%(args)s):
    """
    %(doc)s
    """
    return _gsw_ufuncs.%(ufuncname)s(%(args)s)
'''


def uf_wrapper(ufname):
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
    argstr = ', '.join(argnames)
    subs = dict(ufuncname=ufname,
                funcname=msig['name'],
                args=argstr,
                )
    helpdict = get_helpdict(msig['path'])
    try:
        desclist = helpdict['DESCRIPTION']
        subs['doc'] = '\n    '.join(desclist)
    except KeyError:
        subs['doc'] = "(no description available)"
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
