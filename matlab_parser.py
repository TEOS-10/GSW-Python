import os
import glob
import re

gsw_matlab_dir = '../GSW-Matlab/Toolbox'
gsw_matlab_subdirs = ['library', 'thermodynamics_from_t']

# pattern for functions returning one variable
mfunc_topline1 = re.compile(r"^function (?P<output>\S+)\s*=\s*"
                            r"gsw_(?P<funcname>\S+)"
                            r"\((?P<input>.*)\)")

# pattern for multiple returns
mfunc_topline2 = re.compile(r"^function \[(?P<output>.*)\]\s*=\s*"
                            r"gsw_(?P<funcname>\S+)"
                            r"\((?P<input>.*)\)")


def list_functions(matdir=gsw_matlab_dir):
    rawlist = glob.glob(os.path.join(matdir, '*.m'))
    signatures = []
    rejects = []
    for m in rawlist:
        with open(m, encoding='latin-1') as f:
            line = f.readline()
        _match = mfunc_topline1.match(line)
        if _match is None:
            _match = mfunc_topline2.match(line)
        if _match is None:
            rejects.append(m)
        else:
            _input = [s.strip() for s in _match.group('input').split(',')]
            _output = [s.strip() for s in _match.group('output').split(',')]
            _funcname = _match.group('funcname')
            signatures.append((_funcname, _input, _output, m))

    return signatures, rejects

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
    with open(fname, encoding='latin-1') as f:
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
    keypat = '^([A-Z ]+):(.*)'
    hdict = dict()
    started = False
    for line in help:
        keyline = re.match(keypat, line)
        if keyline:
            if started:
                hdict[key] = blocklines
            key = keyline.groups()[0]
            blocklines = []
            started = True
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
    return hdict
