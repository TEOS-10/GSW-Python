"""
Generate the src/_ufuncs.c file to turn the scalar C functions
into numpy ufuncs.  Also writes ufuncs.list as a record of the
ufunc names.

"""
from pathlib import Path

from c_header_parser import (
    get_sigdict,
    get_simple_name_nin_returntype,
    get_complex_name_nin_nout,
    mixed_sigdict,
)

blacklist = ['add_barrier', 'add_mean']

basedir = Path(__file__).parent.parent

modfile_head_top = """
/*
This file is auto-generated--do not edit it.

*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include "gswteos-10.h"

/* possible hack for MSVC: */
#ifndef NAN
    static double NAN = 0.0/0.0;
#endif

#ifndef isnan
#   define isnan(x) ((x) != (x))
#endif

#define CONVERT_INVALID(x) ((x == GSW_INVALID_VALUE)? NAN: x)

"""

# Loops will be generated by calls to modfile_loop_entry.


modfile_middle = """

#include "method_bodies.c"

static PyMethodDef GswMethods[] = {
# include "method_def_entries.c"
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    GswMethods,
    NULL,
    NULL,
    NULL,
    NULL
};


PyMODINIT_FUNC PyInit__gsw_ufuncs(void)
{
    PyObject *m, *d;

    PyObject *ufunc_ptr;

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    d = PyModule_GetDict(m);

    import_array();
    import_umath();
"""

modfile_tail = """

    return m;
}
"""


def modfile_loop_entry(nin, nout, out_type):
    if out_type == 'd':
        out_return = 'double'
        npy_out_type = 'NPY_DOUBLE'
    else:
        out_return = 'int'
        npy_out_type = 'NPY_INT'  # maybe change to NPY_BOOL
    ndin = 'd'*nin
    ndout = out_type*nout
    loop_id = '%s_%s' % (ndin, ndout)

    linelist = ['/* %d in, %d out */' % (nin, nout)]
    linelist.extend([
    'static void loop1d_%s(char **args, npy_intp const *dimensions,' % loop_id,
    '                          npy_intp const* steps, void* data)',
    '{',
    '    npy_intp i;',
    '    npy_intp n = dimensions[0];'])
    for i in range(nin):
        linelist.append('    char *in%d = args[%d];' % (i, i))
        linelist.append('    npy_intp in_step%d = steps[%d];' % (i, i))
    for i in range(nout):
        linelist.append('    char *out%d = args[%d];' % (i, i+nin))
        linelist.append('    npy_intp out_step%d = steps[%d];' % (i, i+nin))
    intypes = ', '.join(['double'] * nin)
    if nout == 1:
        linelist.append(f'    {out_return} (*func)(%s);' % (intypes,))
    else:
        # Multiple outputs: only double is supported here.
        outtypes = ', '.join(['double *'] * nout)
        linelist.append('    void (*func)(%s, %s);' % (intypes, outtypes))

    # Declare local variables for outputs.
    douts = []
    for i in range(nout):
        douts.append('outd%d' % (i,))
    linelist.append(f'    {out_return} %s;' % ', '.join(douts))
    linelist.extend([
    '    func = data;',
    '',  # End of declarations, start the loop.
    '    for (i = 0; i < n; i++) {'])
    tests = []
    args = []
    for i in range(nin):
        tests.append('isnan(*(double *)in%d)' % i)
        args.append('*(double *)in%d' % i)
    linelist.append('        if (%s) {' % '||'.join(tests))
    outs = []
    for i in range(nout):
        if out_type == 'd':
            outs.append('*((double *)out%d) = NAN;' % i)
        else:  # integer for infunnel
            outs.append('*((int *)out0) = 0;')
    linelist.append('            %s' % ''.join(outs))
    linelist.append('        } else {')
    if nout > 1:
        for i in range(nout):
            args.append('&outd%d' % i)
        linelist.append('            func(%s);' % ', '.join(args))
    else:
        linelist.append('            outd0 = func(%s);' % ', '.join(args))
    if out_type == 'd':
        for i in range(nout):
            linelist.append('            *((double *)out%d)' % (i,)
                            + ' = CONVERT_INVALID(outd%d);' % (i,))
    else:
        for i in range(nout):
            linelist.append('            *((int *)out%d)' % (i,)
                            + ' = outd%d;' % (i,))

    linelist.append('        }')
    for i in range(nin):
        linelist.append('        in%d += in_step%d;' % (i, i))
    for i in range(nout):
        linelist.append('        out%d += out_step%d;' % (i, i))

    linelist.extend(['    }', '}', ''])
    linelist.append('static PyUFuncGenericFunction'
                    ' funcs_%s[] = {&loop1d_%s};' % (loop_id, loop_id))
    linelist.append('')
    linelist.append('static char types_%s[] = {' % (loop_id,))

    linelist.append('        ' + 'NPY_DOUBLE, ' * nin)
    linelist.append('        ' + f'{npy_out_type}, ' * nout)
    linelist.extend(['};', ''])

    return '\n'.join(linelist)


def modfile_loop_entry_from_sig(sig):
    """
    Special case for gibbs, gibbs_ice.
    Assume the first half of the args are int, the remainder are double.
    Return is a double.
    This could all be generalized, but there is probably no need to do so.
    It could also be simplified by stripping out the handling of nout > 1.
    """
    nin = len(sig["argtypes"])
    nout = 1
    # loop_id = f"{'i' * (nin//2)}{'d' * (nin//2)}_{'d' * nout}"
    loop_id = sig["letter_sig"]
    linelist = ['/* %d int in, %d double in, %d out */' % (nin//2, nin//2, nout)]
    linelist.extend([
    'static void loop1d_%s(char **args, npy_intp const *dimensions,' % loop_id,
    '                          npy_intp const* steps, void* data)',
    '{',
    '    npy_intp i;',
    '    npy_intp n = dimensions[0];'])
    for i in range(nin):
        linelist.append('    char *in%d = args[%d];' % (i, i))
        linelist.append('    npy_intp in_step%d = steps[%d];' % (i, i))
    for i in range(nout):
        linelist.append('    char *out%d = args[%d];' % (i, i+nin))
        linelist.append('    npy_intp out_step%d = steps[%d];' % (i, i+nin))
    intypes = ', '.join(['int'] * (nin//2) + ['double'] * (nin//2))
    if nout == 1:
        linelist.append('    double (*func)(%s);' % (intypes,))
    else:
        outtypes = ', '.join(['double *'] * nout)
        linelist.append('    void (*func)(%s, %s);' % (intypes, outtypes))

    douts = []
    for i in range(nout):
        douts.append('outd%d' % (i,))
    linelist.append('    double %s;' % ', '.join(douts))
    linelist.extend([
    '    func = data;',
    '',
    '    for (i = 0; i < n; i++) {'])
    tests = []
    args = []
    for i in range(nin//2, nin):
        tests.append('isnan(*(double *)in%d)' % i)
    for i in range(nin//2):
        args.append('(int)*(long long *)in%d' % i)
    for i in range(nin//2, nin):
        args.append('*(double *)in%d' % i)
    linelist.append('        if (%s) {' % '||'.join(tests))
    outs = []
    for i in range(nout):
        outs.append('*((double *)out%d) = NAN;' % i)
    linelist.append('            %s' % ''.join(outs))
    linelist.append('        } else {')
    if nout > 1:
        for i in range(nout):
            args.append('&outd%d' % i)
        linelist.append('            func(%s);' % ', '.join(args))
    else:
        linelist.append('            outd0 = func(%s);' % ', '.join(args))
    for i in range(nout):
        linelist.append('            *((double *)out%d)' % (i,)
                        + ' = CONVERT_INVALID(outd%d);' % (i,))
    linelist.append('        }')
    for i in range(nin):
        linelist.append('        in%d += in_step%d;' % (i, i))
    for i in range(nout):
        linelist.append('        out%d += out_step%d;' % (i, i))

    linelist.extend(['    }', '}', ''])
    linelist.append('static PyUFuncGenericFunction'
                    ' funcs_%s[] = {&loop1d_%s};' % (loop_id, loop_id))
    linelist.append('')
    linelist.append('static char types_%s[] = {' % (loop_id,))

    linelist.append('        ' + 'NPY_INT64, ' * (nin//2))
    linelist.append('        ' + 'NPY_DOUBLE, ' * (nin//2))
    linelist.append('        ' + 'NPY_DOUBLE, ' * nout)
    linelist.extend(['};', ''])

    return '\n'.join(linelist)


def modfile_array_entry(funcname):
    return "static void *data_%s[] = {&gsw_%s};\n" % (funcname, funcname)


_init_entry = """
    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_%(ndin)s_%(ndout)s,
                                    data_%(funcname)s,
                                    types_%(ndin)s_%(ndout)s,
                                    1, %(nin)d, %(nout)d,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "%(funcname)s",
                                    "%(funcname)s_docstring",
                                    0);

    PyDict_SetItemString(d, "%(funcname)s", ufunc_ptr);
    Py_DECREF(ufunc_ptr);
"""


def modfile_init_entry(funcname, nin, nout, out_type='d'):
    return _init_entry % dict(funcname=funcname, nin=nin, nout=nout,
                              ndin='d'*nin, ndout=out_type*nout)

def modfile_init_entry_from_sig(sig):
    # Specialized for the gibbs functions.
    funcname = sig["name"]
    nin = len(sig["argtypes"])
    nout = 1
    letter_sig = sig["letter_sig"]
    entry = f"""
    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_{letter_sig:s},
                                    data_{funcname:s},
                                    types_{letter_sig:s},
                                    1, {nin:d}, {nout:d},  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "{funcname:s}",
                                    "{funcname:s}_docstring",
                                    0);

    PyDict_SetItemString(d, "{funcname:s}", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    """
    return entry % vars()

def write_modfile(modfile_name, srcdir):
    raw_sigdict = get_sigdict(srcdir=srcdir)
    sigdict = {name: sig for name, sig in raw_sigdict.items() if name not in blacklist}
    simple_tups = get_simple_name_nin_returntype(sigdict)
    complex_tups = get_complex_name_nin_nout(sigdict)
    mixed_sigs = mixed_sigdict(sigdict)

    modfile_head_parts = [modfile_head_top]
    simple_artups = {(nin, 1, returntype[0]) for _, nin, returntype in simple_tups}
    for artup in sorted(simple_artups):
        modfile_head_parts.append(modfile_loop_entry(*artup))

    complex_artups = {tup[1:] for tup in complex_tups}
    for artup in sorted(complex_artups):
        modfile_head_parts.append(modfile_loop_entry(*artup, 'd'))
    modfile_head = '\n'.join(modfile_head_parts)

    chunks = [modfile_head]

    for sig in mixed_sigs.values():
        chunks.append(modfile_loop_entry_from_sig(sig))

    # Array entries
    for name, _, _ in simple_tups:
        chunks.append(modfile_array_entry(name))

    for name, _, _ in complex_tups:
        chunks.append(modfile_array_entry(name))

    for name in mixed_sigs.keys():
        chunks.append(modfile_array_entry(name))

    chunks.append(modfile_middle)

    for name, nin, returntype in simple_tups:
        chunks.append(modfile_init_entry(name, nin, 1, returntype[0]))

    for name, nin, nout in complex_tups:
        chunks.append(modfile_init_entry(name, nin, nout, 'd'))

    for sig in mixed_sigs.values():
        chunks.append(modfile_init_entry_from_sig(sig))

    chunks.append(modfile_tail)

    with modfile_name.open('w') as f:
        f.write(''.join(chunks))

    funcnamelist1 = sorted([tup[0] for tup in simple_tups])
    with open(srcdir.joinpath('_ufuncs1.list'), 'w') as f:
        f.write('\n'.join(funcnamelist1))

    funcnamelist2 = sorted([tup[0] for tup in complex_tups])
    with open(srcdir.joinpath('_ufuncs2.list'), 'w') as f:
        f.write('\n'.join(funcnamelist2))

    funcnamelist = funcnamelist1 + funcnamelist2 + list(mixed_sigs.keys())
    funcnamelist.sort()
    with open(srcdir.joinpath('_ufuncs.list'), 'w') as f:
        f.write('\n'.join(funcnamelist))

if __name__ == '__main__':
    srcdir = basedir.joinpath('src')
    modfile_name = basedir.joinpath(srcdir, '_ufuncs.c')
    write_modfile(modfile_name, srcdir=srcdir)
