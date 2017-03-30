"""
Generate the src/_ufuncs.c file to turn the scalar C functions
into numpy ufuncs.  Also writes ufuncs.list as a record of the
ufunc names.
"""

from c_header_parser import (get_simple_sig_dict,
                            get_complex_scalar_dict_by_nargs_nreturns)

modfile_name = 'src/_ufuncs.c'

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

/* possible hack for MSVC: */
#ifndef NAN
#   static double NAN = 0.0/0.0;
#endif

#ifndef isnan
#   define isnan(x) ((x) != (x))
#endif

#define CONVERT_INVALID(x) ((x == GSW_INVALID_VALUE)? NAN: x)

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
    double outd;
    func = data;

    for (i = 0; i < n; i++) {
        outd = func(*(double *)in1);
        *((double *)out) = CONVERT_INVALID(outd);

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
    double outd;
    func = data;

    for (i = 0; i < n; i++) {
        outd = func(*(double *)in1,
                    *(double *)in2);
        *((double *)out) = CONVERT_INVALID(outd);

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
    double outd;
    func = data;

    for (i = 0; i < n; i++) {
        outd = func(*(double *)in1,
                    *(double *)in2,
                    *(double *)in3);
        *((double *)out) = CONVERT_INVALID(outd);

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
    double outd;
    func = data;

    for (i = 0; i < n; i++) {
        outd = func(*(double *)in1,
                    *(double *)in2,
                    *(double *)in3,
                    *(double *)in4);
        *((double *)out) = CONVERT_INVALID(outd);

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
    double outd;
    func = data;

    for (i = 0; i < n; i++) {
        outd = func(*(double *)in1,
                    *(double *)in2,
                    *(double *)in3,
                    *(double *)in4,
                    *(double *)in5);
        *((double *)out) = CONVERT_INVALID(outd);

        in1 += in_step1;
        in2 += in_step2;
        in3 += in_step3;
        in4 += in_step4;
        in5 += in_step5;
        out += out_step;
    }
}

static void loop1d_ddd_ddd(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *in2 = args[1];
    char *in3 = args[2];
    char *out1 = args[3];
    char *out2 = args[4];
    char *out3 = args[5];
    npy_intp in_step1 = steps[0];
    npy_intp in_step2 = steps[1];
    npy_intp in_step3 = steps[2];
    npy_intp out_step1 = steps[3];
    npy_intp out_step2 = steps[4];
    npy_intp out_step3 = steps[4];
    void (*func)(double, double, double, double *, double *, double *);
    double outd1, outd2, outd3;
    func = data;

    for (i = 0; i < n; i++) {
        func(*(double *)in1,
             *(double *)in2,
             *(double *)in3,
             &outd1, &outd2, &outd3
             );
        *((double *)out1) = CONVERT_INVALID(outd1);
        *((double *)out2) = CONVERT_INVALID(outd2);
        *((double *)out3) = CONVERT_INVALID(outd3);

        in1 += in_step1;
        in2 += in_step2;
        in3 += in_step3;
        out1 += out_step1;
        out2 += out_step2;
        out3 += out_step3;
    }
}




static PyUFuncGenericFunction funcs_d_d[] = {&loop1d_d_d};
static PyUFuncGenericFunction funcs_dd_d[] = {&loop1d_dd_d};
static PyUFuncGenericFunction funcs_ddd_d[] = {&loop1d_ddd_d};
static PyUFuncGenericFunction funcs_dddd_d[] = {&loop1d_dddd_d};
static PyUFuncGenericFunction funcs_ddddd_d[] = {&loop1d_ddddd_d};

static PyUFuncGenericFunction funcs_ddd_ddd[] = {&loop1d_ddd_ddd};


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


static char types_ddd_ddd[] = {
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


def modfile_init_entry(funcname, nin, nout):
    return _init_entry % dict(funcname=funcname, nin=nin, nout=nout,
                              ndin='d'*nin, ndout='d'*nout)


def write_modfile(modfile_name):
    argcategories1 = get_simple_sig_dict()
    chunks = [modfile_head]
    funcnamelist1 = []

    nins = range(1, 6)
    for nin in nins:
        for funcname in argcategories1[nin]:
            chunks.append(modfile_array_entry(funcname))
            funcnamelist1.append(funcname)

    argcategories2 = get_complex_scalar_dict_by_nargs_nreturns()
    funcnamelist2 = []
    for artup in [(3, 3),]:
        for funcname in argcategories2[artup]:
            chunks.append(modfile_array_entry(funcname))
            funcnamelist2.append(funcname)

    chunks.append(modfile_middle)

    for nin in nins:
        for funcname in argcategories1[nin]:
            chunks.append(modfile_init_entry(funcname, nin, 1))

    for artup in [(3, 3),]:
        for funcname in argcategories2[artup]:
            chunks.append(modfile_init_entry(funcname, *artup))


    chunks.append(modfile_tail)

    with open(modfile_name, 'w') as f:
        f.write(''.join(chunks))

    funcnamelist1.sort()
    with open('ufuncs1.list', 'w') as f:
        f.write('\n'.join(funcnamelist1))

    funcnamelist2.sort()
    with open('ufuncs2.list', 'w') as f:
        f.write('\n'.join(funcnamelist2))


if __name__ == '__main__':
    write_modfile(modfile_name)
