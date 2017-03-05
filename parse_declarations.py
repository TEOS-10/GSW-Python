import re

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

#print('\n'.join(sigs))

argsets = dict()

argpat = re.compile(r'.*\((.*)\);')

retpat = re.compile(r'extern\s+(\w+)\s+(\**)gsw_(\w+)')

for sig in sigs:
    argstr = argpat.match(sig).groups()[0]
    argtup = tuple([a.strip() for a in argstr.split(',')])
    retgroups = retpat.match(sig).groups()
    ret = retgroups[0] + retgroups[1]
    ret_argtup = (ret, argtup)
    #print(retgroups[2], ret_argtup)
    if ret_argtup in argsets:
        argsets[ret_argtup].append(retgroups[2])
    else:
        argsets[ret_argtup] = [retgroups[2]]

#for key, value in argsets.items():
#    print(key, value)

# argcategories will hold cases where some number of double arguments
# return a double.  These are the easiest to deal with in bulk.
#  (I need to make a modified version that gives all arg categories,
#  not just the simple ones.  Then we can see how many other cases
#  must be handled.)
argcategories = dict()

skip = False
for arginfo, names in argsets.items():
    if arginfo[0] != 'double':
        continue
    for arg in arginfo[1]:
        a0, a1 = arg.split()
        print(a0, a1)
        if a0 != 'double' or a1.startswith('*'):
            print('not double')
            skip = True
            break
    if skip:
        skip = False
        continue
    key = len(arginfo[1])
    if key in argcategories:
        argcategories[key].extend(names)
    else:
        argcategories[key] = names

for key, value in argcategories.items():
    print(key, value)

for key, value in argcategories.items():
    print(key, len(value))

simplefuncs = []
for value in argcategories.values():
    simplefuncs.extend(value)

allfuncs = set()
for value in argsets.values():
    allfuncs.update(value)

complexfuncs = [f for f in allfuncs if f not in simplefuncs]



modfile_head = """
/* This is python 3-only (for simplicity).
*/

#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"
#include "gswteos-10.h"

static PyMethodDef GswMethods[] = {
        {NULL, NULL, 0, NULL}
};


static void loop1d_dd_d(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *in2 = args[1];
    char *out = args[2];
    npy_intp in_step1 = steps[0];
    npy_intp in_step2 = steps[1];
    npy_intp out_step = steps[2];
    double (*func)(double, double);
    func = data;

    for (i = 0; i < n; i++) {
        *((double *)out) = func(*(double *)in1,
                                *(double *)in2);

        in1 += in_step1;
        in2 += in_step2;
        out += out_step;
    }
}

static void loop1d_ddd_d(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *in2 = args[1];
    char *in3 = args[2];
    char *out = args[3];
    npy_intp in_step1 = steps[0];
    npy_intp in_step2 = steps[1];
    npy_intp in_step3 = steps[2];
    npy_intp out_step = steps[3];
    double (*func)(double, double, double);
    func = data;

    for (i = 0; i < n; i++) {
        *((double *)out) = func(*(double *)in1,
                                *(double *)in2,
                                *(double *)in3);

        in1 += in_step1;
        in2 += in_step2;
        in3 += in_step3;
        out += out_step;
    }
}

static void loop1d_dddd_d(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *in2 = args[1];
    char *in3 = args[2];
    char *in4 = args[3];
    char *out = args[4];
    npy_intp in_step1 = steps[0];
    npy_intp in_step2 = steps[1];
    npy_intp in_step3 = steps[2];
    npy_intp in_step4 = steps[3];
    npy_intp out_step = steps[4];
    double (*func)(double, double, double, double);
    func = data;

    for (i = 0; i < n; i++) {
        *((double *)out) = func(*(double *)in1,
                                *(double *)in2,
                                *(double *)in3,
                                *(double *)in4);

        in1 += in_step1;
        in2 += in_step2;
        in3 += in_step3;
        in4 += in_step4;
        out += out_step;
    }
}




static PyUFuncGenericFunction funcs_dd_d[] = {&loop1d_dd_d};
static PyUFuncGenericFunction funcs_ddd_d[] = {&loop1d_ddd_d};
static PyUFuncGenericFunction funcs_dddd_d[] = {&loop1d_dddd_d};

/* These are the input and return dtypes.*/
static char types_dd_d[] = {
                       NPY_DOUBLE, NPY_DOUBLE,
                       NPY_DOUBLE,
};

static char types_ddd_d[] = {
                       NPY_DOUBLE, NPY_DOUBLE,
                       NPY_DOUBLE, NPY_DOUBLE,
};

static char types_dddd_d[] = {
                       NPY_DOUBLE, NPY_DOUBLE,
                       NPY_DOUBLE, NPY_DOUBLE,
                       NPY_DOUBLE,
};

/* The next thing is generic: */

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
"""    # modfile_head

modfile_tail = """

    return m;
}
"""

modfile_middle = """
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

def modfile_array_entry(funcname):
    return "static void *data_%s[] = {&gsw_%s};\n" % (funcname, funcname)

_init_entry = """
    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_%(nd)s_d,
                                    data_%(funcname)s,
                                    types_%(nd)s_d,
                                    1, %(nin)d, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "%(funcname)s",
                                    "%(funcname)s_docstring",
                                    0);

    PyDict_SetItemString(d, "%(funcname)s", ufunc_ptr);
    Py_DECREF(ufunc_ptr);
"""

def modfile_init_entry(funcname, nin):
    return _init_entry % dict(funcname=funcname, nin=nin, nd='d'*nin)

def write_modfile():
    chunks = [modfile_head]
    for nin in [2, 3, 4]:
        for funcname in argcategories[nin]:
            chunks.append(modfile_array_entry(funcname))

    chunks.append(modfile_middle)

    for nin in [2, 3, 4]:
        for funcname in argcategories[nin]:
            chunks.append(modfile_init_entry(funcname, nin))

    chunks.append(modfile_tail)

    with open('src/_ufuncs.c', 'w') as f:
        f.write(''.join(chunks))



