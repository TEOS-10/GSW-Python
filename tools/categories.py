"""
Use 'src_wrapped_ufuncs.list' and the function names from parsing
Matlab to generate lists of wrapped functions in categories.
"""

from matlab_parser import get_sigdicts_by_subdir

sigdicts = get_sigdicts_by_subdir()
with open('src_wrapped_ufuncs.list') as f:
    lines = f.readlines()
    uflist = [name.strip() for name in lines]

def write_basic_conversions():
    out = []
    nlist = [n for n in uflist if 'from' in n]
    for name in nlist:
        if not('ice' in name or 'freezing' in name or 'exact' in name):
            try:
                out.append(sigdicts['toolbox'][name]['name'])
            except KeyError:
                pass
    out.append('')
    with open('basic_conversions.list', 'w') as f:
        f.write(',\n'.join(out))

def write_freezing():
    out = []
    nlist = [n for n in uflist if 'freezing' in n]
    for name in nlist:
        try:
            out.append(sigdicts['toolbox'][name]['name'])
        except KeyError:
            pass
    out.append('')
    with open('freezing.list', 'w') as f:
        f.write(',\n'.join(out))

def write_ice_not_freezing():
    out = []
    nlist = [n for n in uflist if 'ice' in n and 'freezing' not in n]
    for name in nlist:
        try:
            out.append(sigdicts['toolbox'][name]['name'])
        except KeyError:
            pass
    out.append('')
    with open('ice.list', 'w') as f:
        f.write(',\n'.join(out))


def write_exact():
    out = []
    nlist = [n for n in uflist if 'exact' in n]
    for name in nlist:
        try:
            out.append(sigdicts['toolbox'][name]['name'])
        except KeyError:
            pass
    out.append('')
    with open('exact.list', 'w') as f:
        f.write(',\n'.join(out))

