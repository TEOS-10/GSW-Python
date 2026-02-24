"""
Functions for taking apart the function declarations in gswteos-10.h.
"""
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
            if line.startswith('DECLSPEC extern'):
                sigs.append(line)
                if not line.endswith(';'):
                    started = True
            elif started:
                sigs[-1] += line
                if line.endswith(';'):
                    started = False
    if strip_extern:
        sigs = [s[16:].strip() for s in sigs]  # probably don't need strip()
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

    try:
        retgroups = retpat.match(sig).groups()
    except AttributeError:
        # For example, name doesn't start with "gsw_".
        return None
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
        if psig is not None:
            sigdict[psig['name']] = psig
    return sigdict

def get_sigdict(srcdir="src"):
    return parse_signatures(get_signatures(srcdir=srcdir))


def get_simple_name_nin_returntype(sigdict):
    """
    Return a list of (name, nin, returntype) tuples.
    Include only functions with double arguments and a single return.
    Return may be double or int.
    """
    tups = []
    for name, sig in sigdict.items():
        if all([t == 'double' for t in sig['argtypes']]):
            nin = len(sig['argtypes'])
            if sig['returntype'] in ('double', 'int'):
                tups.append((name, nin, sig['returntype']))
    return tups


def get_complex_name_nin_nout(sigdict):
    """
    Return a list of (name, nin, nout) tuples.
    Include only functions with multiple outputs, double only.
    This not bullet-proof, but it works with the current set of functions.
    """
    tups = []
    simple = [tup[0] for tup in get_simple_name_nin_returntype(sigdict)]
    for name, sig in sigdict.items():
        if name in simple:
            continue
        if sig['returntype'] == 'void' and 'int' not in sig['argtypes']:
            nin = 0
            nout = 0
            for arg in sig['argtuple']:
                if '*' in arg:
                    nout += 1
                else:
                    nin += 1
            tups.append((name, nin, nout))
    return tups

def mixed_sigdict(sigdict):
    """
    This should find gibbs and gibbs_ice, with their leading int arguments.
    It is keyed by name.
    Returns a subset of sigdict, with a "letter_sig" entry added to each
    signature.
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

