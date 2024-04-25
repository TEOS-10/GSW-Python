"""
Functions for taking apart the function declarations in gswteos-10.h.
"""
from collections import ChainMap
from pathlib import Path
import re

import numpy as np


basedir = Path(__file__).parent.parent

def get_signatures(strip_extern=True, srcdir='src'):
    """
    Return a list of C function declarations.
    """
    fname = basedir.joinpath(srcdir, "c_gsw/gswteos-10.h")

    with fname.open() as f:
        for line in f:
            if 'Prototypes' in line:
                break
        sigs = []
        started = False
        for line in f:
            line = line.strip()
            if line.startswith('extern'):
                sigs.append(line)
                if not line.endswith(';'):
                    started = True
            elif started:
                sigs[-1] += line
                if line.endswith(';'):
                    started = False
    if strip_extern:
        sigs = [s[7:].strip() for s in sigs]  # probably don't need strip()
    return sigs


def parse_signature(sig):
    # grab the part inside parentheses: the arguments (single group)
    arglistpat = re.compile(r'.*\((.*)\);')

    # the return type and asterisk if any, and name (3 groups)
    retpat = re.compile(r'^(\w+)\s+(\**)gsw_(\w+)')

    # in an argument specification, get the type, asterisk if any, name (3)
    argpat = re.compile(r'(\w+)\s+(\**)(\w+)')

    # Get the full argument list string.
    argstr = arglistpat.match(sig).groups()[0]

    # Make a tuple with an entry for each argument, e.g., 'double p'.
    argtup = tuple([a.strip() for a in argstr.split(',')])

    argtypes = []
    argnames = []
    for arg in argtup:
        parts = argpat.match(arg).groups()
        argtypes.append(parts[0] + parts[1])
        argnames.append(parts[2])

    retgroups = retpat.match(sig).groups()
    ret = retgroups[0] + retgroups[1]
    funcname = retgroups[2]

    return dict(name=funcname,
                returntype=ret,
                argtypes=tuple(argtypes),
                argnames=tuple(argnames),
                argstring=argstr,
                argtuple=argtup,
                )

def parse_signatures(sigs):
    """
    Given the default list of signatures from get_signatures,
    return a dictionary with function names as keys, and with
    each entry being the (dictionary) output of parse_signature.
    """
    sigdict = {}
    for sig in sigs:
        psig = parse_signature(sig)
        sigdict[psig['name']] = psig
    return sigdict

def get_sigdict(srcdir="src"):
    return parse_signatures(get_signatures(srcdir=srcdir))

# Note: some "sigdict" structures below do *not* use the name as the key.

def simple_sigs(sigdict):
    """
    Given the dict output of parse_signatures, return a dict
    with the *number of inputs as key*, and a list of names as the value.
    Only functions with double arguments and return value are included.
    """
    simple = {}
    for psig in sigdict.values():
        if (psig['returntype'] == 'double' and
                all([t == 'double' for t in psig['argtypes']])):
            n = len(psig['argtypes'])
            if n in simple:
                simple[n].append(psig['name'])
            else:
                simple[n] = [psig['name']]
    for value in simple.values():
        value.sort()
    return simple

def get_simple_sig_dict(srcdir='src'):
    return simple_sigs(get_sigdict(srcdir="src"))

def complex_sigdict(sigdict):
    """
    This is a name-keyed sigdict with everything that is *not* in "simple".
    """
    out = {}
    for key, psig in sigdict.items():
        if (psig['returntype'] == 'double' and
                all([t == 'double' for t in psig['argtypes']])):
            continue
        out[key] = psig
    return out

def get_complex_sigdict(srcdir='src'):
    return complex_sigdict(get_sigdict(srcdir=srcdir))


def mixed_sigdict(sigdict):
    """
    This should find gibbs and gibbs_ice, with their leading int arguments.
    It is keyed by name.
    """
    out1 = {k: psig for k, psig in sigdict.items() if psig['returntype'] == 'double'}
    out = {}
    for k, psig in out1.items():
        n_int = np.array([arg == "int" for arg in psig["argtypes"]]).sum()
        n_double = np.array([arg == "double" for arg in psig["argtypes"]]).sum()
        if n_int > 0 and n_int + n_double == len(psig["argtypes"]):
            out[k] = psig
            psig["letter_sig"] = f"{''.join([a[0] for a in psig['argtypes']])}_d"
    return out

def get_mixed_sigdict(srcdir="src"):
    return mixed_sigdict(get_sigdict(srcdir=srcdir))

def get_complex_scalar_sigdict(srcdir='src'):
    """
    Return a name-keyed sigdict for functions with more than one return but
    with scalar arguments and return values.
    """
    # This works with the current set of functions, but it is not using a fully
    # general criterion.  It would fail if a scalar function were added with
    # more than one output and with integer arguments.
    cd = get_complex_sigdict(srcdir=srcdir)
    scalar_dict = {}
    for k, v in cd.items():
        if v['returntype'] == 'void' and 'int' not in v['argtypes']:
            scalar_dict[k] = v
    return scalar_dict

def get_complex_scalar_dict_by_nargs_nreturns(srcdir='src'):
    sd = get_complex_scalar_sigdict(srcdir=srcdir)
    names_by_sigtup = {}
    for k, v in sd.items():
        nargs = 0
        nrets = 0
        for arg in v['argtuple']:
            if '*' in arg:
                nrets += 1
            else:
                nargs += 1
        sigtup = (nargs, nrets)
        if sigtup in names_by_sigtup:
            names_by_sigtup[sigtup].append(k)
        else:
            names_by_sigtup[sigtup] = [k]
    return names_by_sigtup

def print_complex_names_by_nargs_nreturns(srcdir='src'):
    d = get_complex_scalar_dict_by_nargs_nreturns(srcdir=srcdir)
    for k, v in d.items():
        print(k, len(v))
        for name in v:
            print('    %s' % name)

def print_non_wrappable(srcdir='src'):
    sigdict = get_sigdict(srcdir=srcdir)  # everything
    csd = complex_sigdict(sigdict)        # some we wrap, some we don't
    scd = get_complex_scalar_sigdict(srcdir=srcdir)  # we wrap these
    mixed = mixed_sigdict(sigdict)  # and these
    # Find the names of functions we don't wrap.
    others = [k for k in csd if k not in ChainMap(scd, mixed)]
    othersd = {k : csd[k] for k in others}
    for k, v in othersd.items():
        print(k, v['argstring'], v['returntype'])
