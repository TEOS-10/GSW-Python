"""
Generate the _ufuncs.c file to turn the scalar C functions
into numpy ufuncs.
"""

from parse_declarations import get_simple_sig_dict

modfile_head = """
/*
This file is auto-generated--do not edit it.

This is python 3-only (for simplicity) to begin with.
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


static void loop1d_d_d(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out = args[1];
    npy_intp in_step1 = steps[0];
    npy_intp out_step = steps[1];
    double (*func)(double);
    func = data;

    for (i = 0; i < n; i++) {
        *((double *)out) = func(*(double *)in1);

        in1 += in_step1;
        out += out_step;
    }
}



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


static void loop1d_ddddd_d(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *in2 = args[1];
    char *in3 = args[2];
    char *in4 = args[3];
    char *in5 = args[4];
    char *out = args[5];
    npy_intp in_step1 = steps[0];
    npy_intp in_step2 = steps[1];
    npy_intp in_step3 = steps[2];
    npy_intp in_step4 = steps[3];
    npy_intp in_step5 = steps[4];
    npy_intp out_step = steps[5];
    double (*func)(double, double, double, double, double);
    func = data;

    for (i = 0; i < n; i++) {
        *((double *)out) = func(*(double *)in1,
                                *(double *)in2,
                                *(double *)in3,
                                *(double *)in4,
                                *(double *)in5);

        in1 += in_step1;
        in2 += in_step2;
        in3 += in_step3;
        in4 += in_step4;
        in5 += in_step5;
        out += out_step;
    }
}



static PyUFuncGenericFunction funcs_d_d[] = {&loop1d_d_d};
static PyUFuncGenericFunction funcs_dd_d[] = {&loop1d_dd_d};
static PyUFuncGenericFunction funcs_ddd_d[] = {&loop1d_ddd_d};
static PyUFuncGenericFunction funcs_dddd_d[] = {&loop1d_dddd_d};
static PyUFuncGenericFunction funcs_ddddd_d[] = {&loop1d_ddddd_d};

/* These are the input and return dtypes.*/
static char types_d_d[] = {
                       NPY_DOUBLE, NPY_DOUBLE,
};

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

static char types_ddddd_d[] = {
                       NPY_DOUBLE, NPY_DOUBLE,
                       NPY_DOUBLE, NPY_DOUBLE,
                       NPY_DOUBLE, NPY_DOUBLE,
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
    argcategories = get_simple_sig_dict()
    chunks = [modfile_head]
    nins = range(1, 6)
    for nin in nins:
        for funcname in argcategories[nin]:
            chunks.append(modfile_array_entry(funcname))

    chunks.append(modfile_middle)

    for nin in nins:
        for funcname in argcategories[nin]:
            chunks.append(modfile_init_entry(funcname, nin))

    chunks.append(modfile_tail)

    with open('src/_ufuncs.c', 'w') as f:
        f.write(''.join(chunks))

if __name__ == '__main__':
    write_modfile()

