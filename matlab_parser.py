import os
import glob
import re

gsw_matlab_dir = '../../TEOS-10/gsw_matlab_v3_05_8'

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
        with open(m, errors='ignore') as f:
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
            signatures.append((_funcname, _input, _output))

    return signatures, rejects

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
