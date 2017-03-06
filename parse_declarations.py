"""
Functions for taking apart the function declarations in gswteos-10.h.
"""

import re

def get_signatures(strip_extern=True):
    """
    Return a list of C function declarations.
    """
    fname = "src/c_gsw/gswteos-10.h"

    with open(fname) as f:
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

def simple_sigs(sigdict):
    """
    Given the dict output of parse_signatures, return a dict
    with the number of inputs as key, and a list of names as the value.
    Only functions returning a double, and with double arguments
    are included.
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
    for key, value in simple.items():
        value.sort()

    return simple

def get_simple_sig_dict():
    sigs = get_signatures()
    sigdict = parse_signatures(sigs)
    simple = simple_sigs(sigdict)
    return simple


