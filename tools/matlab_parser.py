"""
It may be necessary to edit the location of the GSW-Matlab directory.
"""

import re
from pathlib import Path

basedir = Path(__file__).parent.parent

gsw_matlab_dir = basedir.joinpath('..', 'GSW-Matlab', 'Toolbox').resolve()
if not gsw_matlab_dir.exists():
    raise IOError(
        f"Could not find the GSW-Matlab source code in {gsw_matlab_dir}."
        "Please read the development notes to find how to setup your GSW-Python development environment."
        )

gsw_matlab_subdirs = ['library', 'thermodynamics_from_t']

# pattern for functions returning one variable
mfunc_topline1 = re.compile(r"^function (?P<output>\S+)\s*=\s*"
                            r"gsw_(?P<funcname>\S+)"
                            r"\((?P<input>.*)\)")

# pattern for multiple returns
mfunc_topline2 = re.compile(r"^function \[(?P<output>.*)\]\s*=\s*"
                            r"gsw_(?P<funcname>\S+)"
                            r"\((?P<input>.*)\)")

# mis-spellings: key is bad in Matlab; replace with value
arg_fixups = dict(sea_surface_geopotental='sea_surface_geopotential',)

def list_functions(matdir=gsw_matlab_dir, subdir=''):
    rawlist = matdir.glob('*.m')
    signatures = []
    rejects = []
    for m in rawlist:
        with m.open(encoding='latin-1') as f:
            line = f.readline()
        _match = mfunc_topline1.match(line)
        if _match is None:
            _match = mfunc_topline2.match(line)
        if _match is None:
            rejects.append(m)
        else:
            _input = [s.strip() for s in _match.group('input').split(',')]
            _input = [arg_fixups.get(n, n) for n in _input]
            _output = [s.strip() for s in _match.group('output').split(',')]
            _funcname = _match.group('funcname')
            signatures.append((_funcname, _input, _output, m))

    return signatures, rejects

def get_all_signatures():
    signatures, _ = list_functions()
    for subdir in gsw_matlab_subdirs:
        path = gsw_matlab_dir.joinpath(subdir)
        s, _ = list_functions(path)
        signatures.extend(s)
    return signatures

def to_sigdict(signatures):
    sigdict = dict()
    for s in signatures:
        _funcname, _input, _output, _m = s
        sdict = dict(name=_funcname,
                     argnames=tuple(_input),
                     outnames=tuple(_output),
                     path=_m)
        sigdict[_funcname.lower()] = sdict
    return sigdict

def get_complete_sigdict():
    return to_sigdict(get_all_signatures())


def get_sigdicts_by_subdir():
    out = dict(toolbox=to_sigdict(list_functions()[0]))
    for subdir in gsw_matlab_subdirs:
        out[subdir] = to_sigdict(list_functions(subdir=subdir)[0])
    return out


def variables_from_signatures(signatures):
    inputs = set()
    outputs = set()
    for sig in signatures:
        inputs.update(sig[1])
        outputs.update(sig[2])
    return inputs, outputs

def input_groups_from_signatures(signatures):
    groups = set()
    for sig in signatures:
        groups.add(tuple(sig[1]))
    return groups

def get_help_text(fname):
    with fname.open(encoding='latin-1') as f:
        lines = f.readlines()
        help = []
        started = False
        for line in lines:
            if not line.startswith('%'):
                if not started:
                    continue
                else:
                    break
            started = True
            help.append(line[2:])
        return help

def help_text_to_dict(help):
    """
    Divide the help text into blocks, using headings as delimiters, and return
    them as a dictionary with the headings as keys.
    """
    # Headings ('USAGE:', 'DESCRIPTION:', etc.) start with all caps and a colon.
    keypat = r"^([A-Z ]+):(.*)"
    hdict = dict()
    topline = help[0][2:].strip()
    parts = topline.split(maxsplit=1)
    if len(parts) == 2:
        hdict["summary"] = parts[1:]
    else:
        hdict["summary"] = ["no summary"]
    started = False
    for line in help[1:]:
        keyline = re.match(keypat, line)
        if keyline:
            # We found a new heading.
            if started:
                # End the previous block.
                hdict[key] = blocklines
            # Save the name of the block.
            key = keyline.groups()[0]
            blocklines = []
            started = True
            # If there is anything else on the heading line, start the block
            # with it.
            _s = keyline.groups()[1].strip()
            if _s:
                blocklines.append(_s)
        elif started:
            _s = line.rstrip()
            _s_ljust = _s.lstrip()
            if (_s_ljust.startswith('The software is') or
                    _s_ljust.startswith('=======')):
                continue
            blocklines.append(_s)
    if started and blocklines:
        hdict[key] = blocklines
    # Library functions don't have sections; we can use the whole thing instead.
    block = []
    started = False
    for line in help:
        if line.startswith("=========="):
            started = True
            continue
        block.append(line)
        if line.startswith("VERSION"):
            break
    hdict['all'] = block
    return hdict


def get_helpdict(fname):
    return help_text_to_dict(get_help_text(fname))
