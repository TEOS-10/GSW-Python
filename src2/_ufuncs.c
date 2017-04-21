
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
    static double NAN = 0.0/0.0;
#endif

#ifndef isnan
#   define isnan(x) ((x) != (x))
#endif

#define CONVERT_INVALID(x) ((x == GSW_INVALID_VALUE)? NAN: x)


/* 1 in, 1 out */
static void loop1d_d_d(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    npy_intp in_step0 = steps[0];
    char *out0 = args[1];
    npy_intp out_step0 = steps[1];
    double (*func)(double);
    double outd0;
    func = data;

    for (i = 0; i < n; i++) {
        if (isnan(*(double *)in0)) {
            *((double *)out0) = NAN;
        } else {
            outd0 = func(*(double *)in0);
            *((double *)out0) = CONVERT_INVALID(outd0);
        }
        in0 += in_step0;
        out0 += out_step0;
    }
}

static PyUFuncGenericFunction funcs_d_d[] = {&loop1d_d_d};

static char types_d_d[] = {
        NPY_DOUBLE, 
        NPY_DOUBLE, 
};

/* 2 in, 1 out */
static void loop1d_dd_d(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    npy_intp in_step0 = steps[0];
    char *in1 = args[1];
    npy_intp in_step1 = steps[1];
    char *out0 = args[2];
    npy_intp out_step0 = steps[2];
    double (*func)(double, double);
    double outd0;
    func = data;

    for (i = 0; i < n; i++) {
        if (isnan(*(double *)in0)||isnan(*(double *)in1)) {
            *((double *)out0) = NAN;
        } else {
            outd0 = func(*(double *)in0, *(double *)in1);
            *((double *)out0) = CONVERT_INVALID(outd0);
        }
        in0 += in_step0;
        in1 += in_step1;
        out0 += out_step0;
    }
}

static PyUFuncGenericFunction funcs_dd_d[] = {&loop1d_dd_d};

static char types_dd_d[] = {
        NPY_DOUBLE, NPY_DOUBLE, 
        NPY_DOUBLE, 
};

/* 3 in, 1 out */
static void loop1d_ddd_d(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    npy_intp in_step0 = steps[0];
    char *in1 = args[1];
    npy_intp in_step1 = steps[1];
    char *in2 = args[2];
    npy_intp in_step2 = steps[2];
    char *out0 = args[3];
    npy_intp out_step0 = steps[3];
    double (*func)(double, double, double);
    double outd0;
    func = data;

    for (i = 0; i < n; i++) {
        if (isnan(*(double *)in0)||isnan(*(double *)in1)||isnan(*(double *)in2)) {
            *((double *)out0) = NAN;
        } else {
            outd0 = func(*(double *)in0, *(double *)in1, *(double *)in2);
            *((double *)out0) = CONVERT_INVALID(outd0);
        }
        in0 += in_step0;
        in1 += in_step1;
        in2 += in_step2;
        out0 += out_step0;
    }
}

static PyUFuncGenericFunction funcs_ddd_d[] = {&loop1d_ddd_d};

static char types_ddd_d[] = {
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
        NPY_DOUBLE, 
};

/* 4 in, 1 out */
static void loop1d_dddd_d(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    npy_intp in_step0 = steps[0];
    char *in1 = args[1];
    npy_intp in_step1 = steps[1];
    char *in2 = args[2];
    npy_intp in_step2 = steps[2];
    char *in3 = args[3];
    npy_intp in_step3 = steps[3];
    char *out0 = args[4];
    npy_intp out_step0 = steps[4];
    double (*func)(double, double, double, double);
    double outd0;
    func = data;

    for (i = 0; i < n; i++) {
        if (isnan(*(double *)in0)||isnan(*(double *)in1)||isnan(*(double *)in2)||isnan(*(double *)in3)) {
            *((double *)out0) = NAN;
        } else {
            outd0 = func(*(double *)in0, *(double *)in1, *(double *)in2, *(double *)in3);
            *((double *)out0) = CONVERT_INVALID(outd0);
        }
        in0 += in_step0;
        in1 += in_step1;
        in2 += in_step2;
        in3 += in_step3;
        out0 += out_step0;
    }
}

static PyUFuncGenericFunction funcs_dddd_d[] = {&loop1d_dddd_d};

static char types_dddd_d[] = {
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
        NPY_DOUBLE, 
};

/* 5 in, 1 out */
static void loop1d_ddddd_d(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    npy_intp in_step0 = steps[0];
    char *in1 = args[1];
    npy_intp in_step1 = steps[1];
    char *in2 = args[2];
    npy_intp in_step2 = steps[2];
    char *in3 = args[3];
    npy_intp in_step3 = steps[3];
    char *in4 = args[4];
    npy_intp in_step4 = steps[4];
    char *out0 = args[5];
    npy_intp out_step0 = steps[5];
    double (*func)(double, double, double, double, double);
    double outd0;
    func = data;

    for (i = 0; i < n; i++) {
        if (isnan(*(double *)in0)||isnan(*(double *)in1)||isnan(*(double *)in2)||isnan(*(double *)in3)||isnan(*(double *)in4)) {
            *((double *)out0) = NAN;
        } else {
            outd0 = func(*(double *)in0, *(double *)in1, *(double *)in2, *(double *)in3, *(double *)in4);
            *((double *)out0) = CONVERT_INVALID(outd0);
        }
        in0 += in_step0;
        in1 += in_step1;
        in2 += in_step2;
        in3 += in_step3;
        in4 += in_step4;
        out0 += out_step0;
    }
}

static PyUFuncGenericFunction funcs_ddddd_d[] = {&loop1d_ddddd_d};

static char types_ddddd_d[] = {
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
        NPY_DOUBLE, 
};

/* 2 in, 2 out */
static void loop1d_dd_dd(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    npy_intp in_step0 = steps[0];
    char *in1 = args[1];
    npy_intp in_step1 = steps[1];
    char *out0 = args[2];
    npy_intp out_step0 = steps[2];
    char *out1 = args[3];
    npy_intp out_step1 = steps[3];
    void (*func)(double, double, double *, double *);
    double outd0, outd1;
    func = data;

    for (i = 0; i < n; i++) {
        if (isnan(*(double *)in0)||isnan(*(double *)in1)) {
            *((double *)out0) = NAN;*((double *)out1) = NAN;
        } else {
            func(*(double *)in0, *(double *)in1, &outd0, &outd1);
            *((double *)out0) = CONVERT_INVALID(outd0);
            *((double *)out1) = CONVERT_INVALID(outd1);
        }
        in0 += in_step0;
        in1 += in_step1;
        out0 += out_step0;
        out1 += out_step1;
    }
}

static PyUFuncGenericFunction funcs_dd_dd[] = {&loop1d_dd_dd};

static char types_dd_dd[] = {
        NPY_DOUBLE, NPY_DOUBLE, 
        NPY_DOUBLE, NPY_DOUBLE, 
};

/* 3 in, 2 out */
static void loop1d_ddd_dd(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    npy_intp in_step0 = steps[0];
    char *in1 = args[1];
    npy_intp in_step1 = steps[1];
    char *in2 = args[2];
    npy_intp in_step2 = steps[2];
    char *out0 = args[3];
    npy_intp out_step0 = steps[3];
    char *out1 = args[4];
    npy_intp out_step1 = steps[4];
    void (*func)(double, double, double, double *, double *);
    double outd0, outd1;
    func = data;

    for (i = 0; i < n; i++) {
        if (isnan(*(double *)in0)||isnan(*(double *)in1)||isnan(*(double *)in2)) {
            *((double *)out0) = NAN;*((double *)out1) = NAN;
        } else {
            func(*(double *)in0, *(double *)in1, *(double *)in2, &outd0, &outd1);
            *((double *)out0) = CONVERT_INVALID(outd0);
            *((double *)out1) = CONVERT_INVALID(outd1);
        }
        in0 += in_step0;
        in1 += in_step1;
        in2 += in_step2;
        out0 += out_step0;
        out1 += out_step1;
    }
}

static PyUFuncGenericFunction funcs_ddd_dd[] = {&loop1d_ddd_dd};

static char types_ddd_dd[] = {
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
        NPY_DOUBLE, NPY_DOUBLE, 
};

/* 3 in, 3 out */
static void loop1d_ddd_ddd(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    npy_intp in_step0 = steps[0];
    char *in1 = args[1];
    npy_intp in_step1 = steps[1];
    char *in2 = args[2];
    npy_intp in_step2 = steps[2];
    char *out0 = args[3];
    npy_intp out_step0 = steps[3];
    char *out1 = args[4];
    npy_intp out_step1 = steps[4];
    char *out2 = args[5];
    npy_intp out_step2 = steps[5];
    void (*func)(double, double, double, double *, double *, double *);
    double outd0, outd1, outd2;
    func = data;

    for (i = 0; i < n; i++) {
        if (isnan(*(double *)in0)||isnan(*(double *)in1)||isnan(*(double *)in2)) {
            *((double *)out0) = NAN;*((double *)out1) = NAN;*((double *)out2) = NAN;
        } else {
            func(*(double *)in0, *(double *)in1, *(double *)in2, &outd0, &outd1, &outd2);
            *((double *)out0) = CONVERT_INVALID(outd0);
            *((double *)out1) = CONVERT_INVALID(outd1);
            *((double *)out2) = CONVERT_INVALID(outd2);
        }
        in0 += in_step0;
        in1 += in_step1;
        in2 += in_step2;
        out0 += out_step0;
        out1 += out_step1;
        out2 += out_step2;
    }
}

static PyUFuncGenericFunction funcs_ddd_ddd[] = {&loop1d_ddd_ddd};

static char types_ddd_ddd[] = {
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
};

/* 6 in, 2 out */
static void loop1d_dddddd_dd(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    npy_intp in_step0 = steps[0];
    char *in1 = args[1];
    npy_intp in_step1 = steps[1];
    char *in2 = args[2];
    npy_intp in_step2 = steps[2];
    char *in3 = args[3];
    npy_intp in_step3 = steps[3];
    char *in4 = args[4];
    npy_intp in_step4 = steps[4];
    char *in5 = args[5];
    npy_intp in_step5 = steps[5];
    char *out0 = args[6];
    npy_intp out_step0 = steps[6];
    char *out1 = args[7];
    npy_intp out_step1 = steps[7];
    void (*func)(double, double, double, double, double, double, double *, double *);
    double outd0, outd1;
    func = data;

    for (i = 0; i < n; i++) {
        if (isnan(*(double *)in0)||isnan(*(double *)in1)||isnan(*(double *)in2)||isnan(*(double *)in3)||isnan(*(double *)in4)||isnan(*(double *)in5)) {
            *((double *)out0) = NAN;*((double *)out1) = NAN;
        } else {
            func(*(double *)in0, *(double *)in1, *(double *)in2, *(double *)in3, *(double *)in4, *(double *)in5, &outd0, &outd1);
            *((double *)out0) = CONVERT_INVALID(outd0);
            *((double *)out1) = CONVERT_INVALID(outd1);
        }
        in0 += in_step0;
        in1 += in_step1;
        in2 += in_step2;
        in3 += in_step3;
        in4 += in_step4;
        in5 += in_step5;
        out0 += out_step0;
        out1 += out_step1;
    }
}

static PyUFuncGenericFunction funcs_dddddd_dd[] = {&loop1d_dddddd_dd};

static char types_dddddd_dd[] = {
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
        NPY_DOUBLE, NPY_DOUBLE, 
};

/* 2 in, 3 out */
static void loop1d_dd_ddd(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    npy_intp in_step0 = steps[0];
    char *in1 = args[1];
    npy_intp in_step1 = steps[1];
    char *out0 = args[2];
    npy_intp out_step0 = steps[2];
    char *out1 = args[3];
    npy_intp out_step1 = steps[3];
    char *out2 = args[4];
    npy_intp out_step2 = steps[4];
    void (*func)(double, double, double *, double *, double *);
    double outd0, outd1, outd2;
    func = data;

    for (i = 0; i < n; i++) {
        if (isnan(*(double *)in0)||isnan(*(double *)in1)) {
            *((double *)out0) = NAN;*((double *)out1) = NAN;*((double *)out2) = NAN;
        } else {
            func(*(double *)in0, *(double *)in1, &outd0, &outd1, &outd2);
            *((double *)out0) = CONVERT_INVALID(outd0);
            *((double *)out1) = CONVERT_INVALID(outd1);
            *((double *)out2) = CONVERT_INVALID(outd2);
        }
        in0 += in_step0;
        in1 += in_step1;
        out0 += out_step0;
        out1 += out_step1;
        out2 += out_step2;
    }
}

static PyUFuncGenericFunction funcs_dd_ddd[] = {&loop1d_dd_ddd};

static char types_dd_ddd[] = {
        NPY_DOUBLE, NPY_DOUBLE, 
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
};

/* 4 in, 3 out */
static void loop1d_dddd_ddd(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    npy_intp in_step0 = steps[0];
    char *in1 = args[1];
    npy_intp in_step1 = steps[1];
    char *in2 = args[2];
    npy_intp in_step2 = steps[2];
    char *in3 = args[3];
    npy_intp in_step3 = steps[3];
    char *out0 = args[4];
    npy_intp out_step0 = steps[4];
    char *out1 = args[5];
    npy_intp out_step1 = steps[5];
    char *out2 = args[6];
    npy_intp out_step2 = steps[6];
    void (*func)(double, double, double, double, double *, double *, double *);
    double outd0, outd1, outd2;
    func = data;

    for (i = 0; i < n; i++) {
        if (isnan(*(double *)in0)||isnan(*(double *)in1)||isnan(*(double *)in2)||isnan(*(double *)in3)) {
            *((double *)out0) = NAN;*((double *)out1) = NAN;*((double *)out2) = NAN;
        } else {
            func(*(double *)in0, *(double *)in1, *(double *)in2, *(double *)in3, &outd0, &outd1, &outd2);
            *((double *)out0) = CONVERT_INVALID(outd0);
            *((double *)out1) = CONVERT_INVALID(outd1);
            *((double *)out2) = CONVERT_INVALID(outd2);
        }
        in0 += in_step0;
        in1 += in_step1;
        in2 += in_step2;
        in3 += in_step3;
        out0 += out_step0;
        out1 += out_step1;
        out2 += out_step2;
    }
}

static PyUFuncGenericFunction funcs_dddd_ddd[] = {&loop1d_dddd_ddd};

static char types_dddd_ddd[] = {
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
};

/* 5 in, 3 out */
static void loop1d_ddddd_ddd(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    npy_intp in_step0 = steps[0];
    char *in1 = args[1];
    npy_intp in_step1 = steps[1];
    char *in2 = args[2];
    npy_intp in_step2 = steps[2];
    char *in3 = args[3];
    npy_intp in_step3 = steps[3];
    char *in4 = args[4];
    npy_intp in_step4 = steps[4];
    char *out0 = args[5];
    npy_intp out_step0 = steps[5];
    char *out1 = args[6];
    npy_intp out_step1 = steps[6];
    char *out2 = args[7];
    npy_intp out_step2 = steps[7];
    void (*func)(double, double, double, double, double, double *, double *, double *);
    double outd0, outd1, outd2;
    func = data;

    for (i = 0; i < n; i++) {
        if (isnan(*(double *)in0)||isnan(*(double *)in1)||isnan(*(double *)in2)||isnan(*(double *)in3)||isnan(*(double *)in4)) {
            *((double *)out0) = NAN;*((double *)out1) = NAN;*((double *)out2) = NAN;
        } else {
            func(*(double *)in0, *(double *)in1, *(double *)in2, *(double *)in3, *(double *)in4, &outd0, &outd1, &outd2);
            *((double *)out0) = CONVERT_INVALID(outd0);
            *((double *)out1) = CONVERT_INVALID(outd1);
            *((double *)out2) = CONVERT_INVALID(outd2);
        }
        in0 += in_step0;
        in1 += in_step1;
        in2 += in_step2;
        in3 += in_step3;
        in4 += in_step4;
        out0 += out_step0;
        out1 += out_step1;
        out2 += out_step2;
    }
}

static PyUFuncGenericFunction funcs_ddddd_ddd[] = {&loop1d_ddddd_ddd};

static char types_ddddd_ddd[] = {
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
};

/* 3 in, 5 out */
static void loop1d_ddd_ddddd(char **args, npy_intp *dimensions,
                          npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in0 = args[0];
    npy_intp in_step0 = steps[0];
    char *in1 = args[1];
    npy_intp in_step1 = steps[1];
    char *in2 = args[2];
    npy_intp in_step2 = steps[2];
    char *out0 = args[3];
    npy_intp out_step0 = steps[3];
    char *out1 = args[4];
    npy_intp out_step1 = steps[4];
    char *out2 = args[5];
    npy_intp out_step2 = steps[5];
    char *out3 = args[6];
    npy_intp out_step3 = steps[6];
    char *out4 = args[7];
    npy_intp out_step4 = steps[7];
    void (*func)(double, double, double, double *, double *, double *, double *, double *);
    double outd0, outd1, outd2, outd3, outd4;
    func = data;

    for (i = 0; i < n; i++) {
        if (isnan(*(double *)in0)||isnan(*(double *)in1)||isnan(*(double *)in2)) {
            *((double *)out0) = NAN;*((double *)out1) = NAN;*((double *)out2) = NAN;*((double *)out3) = NAN;*((double *)out4) = NAN;
        } else {
            func(*(double *)in0, *(double *)in1, *(double *)in2, &outd0, &outd1, &outd2, &outd3, &outd4);
            *((double *)out0) = CONVERT_INVALID(outd0);
            *((double *)out1) = CONVERT_INVALID(outd1);
            *((double *)out2) = CONVERT_INVALID(outd2);
            *((double *)out3) = CONVERT_INVALID(outd3);
            *((double *)out4) = CONVERT_INVALID(outd4);
        }
        in0 += in_step0;
        in1 += in_step1;
        in2 += in_step2;
        out0 += out_step0;
        out1 += out_step1;
        out2 += out_step2;
        out3 += out_step3;
        out4 += out_step4;
    }
}

static PyUFuncGenericFunction funcs_ddd_ddddd[] = {&loop1d_ddd_ddddd};

static char types_ddd_ddddd[] = {
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
        NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, 
};
static void *data_enthalpy_sso_0[] = {&gsw_enthalpy_sso_0};
static void *data_gibbs_ice_pt0[] = {&gsw_gibbs_ice_pt0};
static void *data_gibbs_ice_pt0_pt0[] = {&gsw_gibbs_ice_pt0_pt0};
static void *data_hill_ratio_at_sp2[] = {&gsw_hill_ratio_at_sp2};
static void *data_pot_enthalpy_from_pt_ice[] = {&gsw_pot_enthalpy_from_pt_ice};
static void *data_pot_enthalpy_from_pt_ice_poly[] = {&gsw_pot_enthalpy_from_pt_ice_poly};
static void *data_pt0_cold_ice_poly[] = {&gsw_pt0_cold_ice_poly};
static void *data_pt_from_pot_enthalpy_ice[] = {&gsw_pt_from_pot_enthalpy_ice};
static void *data_pt_from_pot_enthalpy_ice_poly[] = {&gsw_pt_from_pot_enthalpy_ice_poly};
static void *data_pt_from_pot_enthalpy_ice_poly_dh[] = {&gsw_pt_from_pot_enthalpy_ice_poly_dh};
static void *data_sp_from_sk[] = {&gsw_sp_from_sk};
static void *data_sp_from_sr[] = {&gsw_sp_from_sr};
static void *data_specvol_sso_0[] = {&gsw_specvol_sso_0};
static void *data_sr_from_sp[] = {&gsw_sr_from_sp};
static void *data_adiabatic_lapse_rate_ice[] = {&gsw_adiabatic_lapse_rate_ice};
static void *data_alpha_wrt_t_ice[] = {&gsw_alpha_wrt_t_ice};
static void *data_chem_potential_water_ice[] = {&gsw_chem_potential_water_ice};
static void *data_cp_ice[] = {&gsw_cp_ice};
static void *data_ct_from_entropy[] = {&gsw_ct_from_entropy};
static void *data_ct_from_pt[] = {&gsw_ct_from_pt};
static void *data_ct_maxdensity[] = {&gsw_ct_maxdensity};
static void *data_enthalpy_ice[] = {&gsw_enthalpy_ice};
static void *data_entropy_from_ct[] = {&gsw_entropy_from_ct};
static void *data_entropy_from_pt[] = {&gsw_entropy_from_pt};
static void *data_entropy_ice[] = {&gsw_entropy_ice};
static void *data_entropy_part_zerop[] = {&gsw_entropy_part_zerop};
static void *data_gibbs_ice_part_t[] = {&gsw_gibbs_ice_part_t};
static void *data_gibbs_pt0_pt0[] = {&gsw_gibbs_pt0_pt0};
static void *data_grav[] = {&gsw_grav};
static void *data_helmholtz_energy_ice[] = {&gsw_helmholtz_energy_ice};
static void *data_internal_energy_ice[] = {&gsw_internal_energy_ice};
static void *data_kappa_const_t_ice[] = {&gsw_kappa_const_t_ice};
static void *data_kappa_ice[] = {&gsw_kappa_ice};
static void *data_latentheat_evap_ct[] = {&gsw_latentheat_evap_ct};
static void *data_latentheat_evap_t[] = {&gsw_latentheat_evap_t};
static void *data_latentheat_melting[] = {&gsw_latentheat_melting};
static void *data_melting_ice_equilibrium_sa_ct_ratio[] = {&gsw_melting_ice_equilibrium_sa_ct_ratio};
static void *data_melting_ice_equilibrium_sa_ct_ratio_poly[] = {&gsw_melting_ice_equilibrium_sa_ct_ratio_poly};
static void *data_melting_seaice_equilibrium_sa_ct_ratio[] = {&gsw_melting_seaice_equilibrium_sa_ct_ratio};
static void *data_melting_seaice_equilibrium_sa_ct_ratio_poly[] = {&gsw_melting_seaice_equilibrium_sa_ct_ratio_poly};
static void *data_p_from_z[] = {&gsw_p_from_z};
static void *data_pot_enthalpy_ice_freezing[] = {&gsw_pot_enthalpy_ice_freezing};
static void *data_pot_enthalpy_ice_freezing_poly[] = {&gsw_pot_enthalpy_ice_freezing_poly};
static void *data_pressure_coefficient_ice[] = {&gsw_pressure_coefficient_ice};
static void *data_pt0_from_t_ice[] = {&gsw_pt0_from_t_ice};
static void *data_pt_from_ct[] = {&gsw_pt_from_ct};
static void *data_pt_from_entropy[] = {&gsw_pt_from_entropy};
static void *data_rho_ice[] = {&gsw_rho_ice};
static void *data_sigma0[] = {&gsw_sigma0};
static void *data_sigma1[] = {&gsw_sigma1};
static void *data_sigma2[] = {&gsw_sigma2};
static void *data_sigma3[] = {&gsw_sigma3};
static void *data_sigma4[] = {&gsw_sigma4};
static void *data_sound_speed_ice[] = {&gsw_sound_speed_ice};
static void *data_specvol_ice[] = {&gsw_specvol_ice};
static void *data_spiciness0[] = {&gsw_spiciness0};
static void *data_spiciness1[] = {&gsw_spiciness1};
static void *data_spiciness2[] = {&gsw_spiciness2};
static void *data_t_from_pt0_ice[] = {&gsw_t_from_pt0_ice};
static void *data_z_from_p[] = {&gsw_z_from_p};
static void *data_adiabatic_lapse_rate_from_ct[] = {&gsw_adiabatic_lapse_rate_from_ct};
static void *data_alpha[] = {&gsw_alpha};
static void *data_alpha_on_beta[] = {&gsw_alpha_on_beta};
static void *data_alpha_wrt_t_exact[] = {&gsw_alpha_wrt_t_exact};
static void *data_beta[] = {&gsw_beta};
static void *data_beta_const_t_exact[] = {&gsw_beta_const_t_exact};
static void *data_c_from_sp[] = {&gsw_c_from_sp};
static void *data_cabbeling[] = {&gsw_cabbeling};
static void *data_chem_potential_water_t_exact[] = {&gsw_chem_potential_water_t_exact};
static void *data_cp_t_exact[] = {&gsw_cp_t_exact};
static void *data_ct_freezing[] = {&gsw_ct_freezing};
static void *data_ct_freezing_poly[] = {&gsw_ct_freezing_poly};
static void *data_ct_from_enthalpy[] = {&gsw_ct_from_enthalpy};
static void *data_ct_from_enthalpy_exact[] = {&gsw_ct_from_enthalpy_exact};
static void *data_ct_from_t[] = {&gsw_ct_from_t};
static void *data_deltasa_atlas[] = {&gsw_deltasa_atlas};
static void *data_dilution_coefficient_t_exact[] = {&gsw_dilution_coefficient_t_exact};
static void *data_dynamic_enthalpy[] = {&gsw_dynamic_enthalpy};
static void *data_enthalpy[] = {&gsw_enthalpy};
static void *data_enthalpy_ct_exact[] = {&gsw_enthalpy_ct_exact};
static void *data_enthalpy_t_exact[] = {&gsw_enthalpy_t_exact};
static void *data_entropy_from_t[] = {&gsw_entropy_from_t};
static void *data_entropy_part[] = {&gsw_entropy_part};
static void *data_fdelta[] = {&gsw_fdelta};
static void *data_internal_energy[] = {&gsw_internal_energy};
static void *data_kappa[] = {&gsw_kappa};
static void *data_kappa_t_exact[] = {&gsw_kappa_t_exact};
static void *data_pressure_freezing_ct[] = {&gsw_pressure_freezing_ct};
static void *data_pt0_from_t[] = {&gsw_pt0_from_t};
static void *data_pt_from_t_ice[] = {&gsw_pt_from_t_ice};
static void *data_rho[] = {&gsw_rho};
static void *data_rho_t_exact[] = {&gsw_rho_t_exact};
static void *data_sa_freezing_from_ct[] = {&gsw_sa_freezing_from_ct};
static void *data_sa_freezing_from_ct_poly[] = {&gsw_sa_freezing_from_ct_poly};
static void *data_sa_freezing_from_t[] = {&gsw_sa_freezing_from_t};
static void *data_sa_freezing_from_t_poly[] = {&gsw_sa_freezing_from_t_poly};
static void *data_sa_from_rho[] = {&gsw_sa_from_rho};
static void *data_sa_from_sp_baltic[] = {&gsw_sa_from_sp_baltic};
static void *data_saar[] = {&gsw_saar};
static void *data_sound_speed[] = {&gsw_sound_speed};
static void *data_sound_speed_t_exact[] = {&gsw_sound_speed_t_exact};
static void *data_sp_from_c[] = {&gsw_sp_from_c};
static void *data_sp_from_sa_baltic[] = {&gsw_sp_from_sa_baltic};
static void *data_specvol[] = {&gsw_specvol};
static void *data_specvol_anom_standard[] = {&gsw_specvol_anom_standard};
static void *data_specvol_t_exact[] = {&gsw_specvol_t_exact};
static void *data_t_deriv_chem_potential_water_t_exact[] = {&gsw_t_deriv_chem_potential_water_t_exact};
static void *data_t_freezing[] = {&gsw_t_freezing};
static void *data_t_freezing_poly[] = {&gsw_t_freezing_poly};
static void *data_t_from_ct[] = {&gsw_t_from_ct};
static void *data_thermobaric[] = {&gsw_thermobaric};
static void *data_deltasa_from_sp[] = {&gsw_deltasa_from_sp};
static void *data_enthalpy_diff[] = {&gsw_enthalpy_diff};
static void *data_melting_ice_sa_ct_ratio[] = {&gsw_melting_ice_sa_ct_ratio};
static void *data_melting_ice_sa_ct_ratio_poly[] = {&gsw_melting_ice_sa_ct_ratio_poly};
static void *data_pot_rho_t_exact[] = {&gsw_pot_rho_t_exact};
static void *data_pt_from_t[] = {&gsw_pt_from_t};
static void *data_sa_from_sp[] = {&gsw_sa_from_sp};
static void *data_sa_from_sstar[] = {&gsw_sa_from_sstar};
static void *data_sp_from_sa[] = {&gsw_sp_from_sa};
static void *data_sp_from_sstar[] = {&gsw_sp_from_sstar};
static void *data_sstar_from_sa[] = {&gsw_sstar_from_sa};
static void *data_sstar_from_sp[] = {&gsw_sstar_from_sp};
static void *data_melting_seaice_sa_ct_ratio[] = {&gsw_melting_seaice_sa_ct_ratio};
static void *data_melting_seaice_sa_ct_ratio_poly[] = {&gsw_melting_seaice_sa_ct_ratio_poly};
static void *data_ct_first_derivatives[] = {&gsw_ct_first_derivatives};
static void *data_entropy_first_derivatives[] = {&gsw_entropy_first_derivatives};
static void *data_pot_enthalpy_ice_freezing_first_derivatives[] = {&gsw_pot_enthalpy_ice_freezing_first_derivatives};
static void *data_pot_enthalpy_ice_freezing_first_derivatives_poly[] = {&gsw_pot_enthalpy_ice_freezing_first_derivatives_poly};
static void *data_pt_first_derivatives[] = {&gsw_pt_first_derivatives};
static void *data_ct_freezing_first_derivatives[] = {&gsw_ct_freezing_first_derivatives};
static void *data_ct_freezing_first_derivatives_poly[] = {&gsw_ct_freezing_first_derivatives_poly};
static void *data_ct_from_rho[] = {&gsw_ct_from_rho};
static void *data_enthalpy_first_derivatives[] = {&gsw_enthalpy_first_derivatives};
static void *data_enthalpy_first_derivatives_ct_exact[] = {&gsw_enthalpy_first_derivatives_ct_exact};
static void *data_rho_first_derivatives_wrt_enthalpy[] = {&gsw_rho_first_derivatives_wrt_enthalpy};
static void *data_specvol_first_derivatives_wrt_enthalpy[] = {&gsw_specvol_first_derivatives_wrt_enthalpy};
static void *data_t_freezing_first_derivatives[] = {&gsw_t_freezing_first_derivatives};
static void *data_t_freezing_first_derivatives_poly[] = {&gsw_t_freezing_first_derivatives_poly};
static void *data_ct_first_derivatives_wrt_t_exact[] = {&gsw_ct_first_derivatives_wrt_t_exact};
static void *data_enthalpy_second_derivatives[] = {&gsw_enthalpy_second_derivatives};
static void *data_enthalpy_second_derivatives_ct_exact[] = {&gsw_enthalpy_second_derivatives_ct_exact};
static void *data_frazil_properties[] = {&gsw_frazil_properties};
static void *data_frazil_properties_potential[] = {&gsw_frazil_properties_potential};
static void *data_frazil_properties_potential_poly[] = {&gsw_frazil_properties_potential_poly};
static void *data_frazil_ratios_adiabatic[] = {&gsw_frazil_ratios_adiabatic};
static void *data_frazil_ratios_adiabatic_poly[] = {&gsw_frazil_ratios_adiabatic_poly};
static void *data_rho_alpha_beta[] = {&gsw_rho_alpha_beta};
static void *data_rho_first_derivatives[] = {&gsw_rho_first_derivatives};
static void *data_rho_second_derivatives_wrt_enthalpy[] = {&gsw_rho_second_derivatives_wrt_enthalpy};
static void *data_specvol_alpha_beta[] = {&gsw_specvol_alpha_beta};
static void *data_specvol_first_derivatives[] = {&gsw_specvol_first_derivatives};
static void *data_specvol_second_derivatives_wrt_enthalpy[] = {&gsw_specvol_second_derivatives_wrt_enthalpy};
static void *data_melting_seaice_into_seawater[] = {&gsw_melting_seaice_into_seawater};
static void *data_ct_second_derivatives[] = {&gsw_ct_second_derivatives};
static void *data_entropy_second_derivatives[] = {&gsw_entropy_second_derivatives};
static void *data_pt_second_derivatives[] = {&gsw_pt_second_derivatives};
static void *data_ice_fraction_to_freeze_seawater[] = {&gsw_ice_fraction_to_freeze_seawater};
static void *data_melting_ice_into_seawater[] = {&gsw_melting_ice_into_seawater};
static void *data_seaice_fraction_to_freeze_seawater[] = {&gsw_seaice_fraction_to_freeze_seawater};
static void *data_rho_second_derivatives[] = {&gsw_rho_second_derivatives};
static void *data_specvol_second_derivatives[] = {&gsw_specvol_second_derivatives};


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

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_enthalpy_sso_0,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "enthalpy_sso_0",
                                    "enthalpy_sso_0_docstring",
                                    0);

    PyDict_SetItemString(d, "enthalpy_sso_0", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_gibbs_ice_pt0,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "gibbs_ice_pt0",
                                    "gibbs_ice_pt0_docstring",
                                    0);

    PyDict_SetItemString(d, "gibbs_ice_pt0", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_gibbs_ice_pt0_pt0,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "gibbs_ice_pt0_pt0",
                                    "gibbs_ice_pt0_pt0_docstring",
                                    0);

    PyDict_SetItemString(d, "gibbs_ice_pt0_pt0", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_hill_ratio_at_sp2,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "hill_ratio_at_sp2",
                                    "hill_ratio_at_sp2_docstring",
                                    0);

    PyDict_SetItemString(d, "hill_ratio_at_sp2", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_pot_enthalpy_from_pt_ice,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pot_enthalpy_from_pt_ice",
                                    "pot_enthalpy_from_pt_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "pot_enthalpy_from_pt_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_pot_enthalpy_from_pt_ice_poly,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pot_enthalpy_from_pt_ice_poly",
                                    "pot_enthalpy_from_pt_ice_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "pot_enthalpy_from_pt_ice_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_pt0_cold_ice_poly,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pt0_cold_ice_poly",
                                    "pt0_cold_ice_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "pt0_cold_ice_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_pt_from_pot_enthalpy_ice,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pt_from_pot_enthalpy_ice",
                                    "pt_from_pot_enthalpy_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "pt_from_pot_enthalpy_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_pt_from_pot_enthalpy_ice_poly,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pt_from_pot_enthalpy_ice_poly",
                                    "pt_from_pot_enthalpy_ice_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "pt_from_pot_enthalpy_ice_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_pt_from_pot_enthalpy_ice_poly_dh,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pt_from_pot_enthalpy_ice_poly_dh",
                                    "pt_from_pot_enthalpy_ice_poly_dh_docstring",
                                    0);

    PyDict_SetItemString(d, "pt_from_pot_enthalpy_ice_poly_dh", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_sp_from_sk,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sp_from_sk",
                                    "sp_from_sk_docstring",
                                    0);

    PyDict_SetItemString(d, "sp_from_sk", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_sp_from_sr,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sp_from_sr",
                                    "sp_from_sr_docstring",
                                    0);

    PyDict_SetItemString(d, "sp_from_sr", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_specvol_sso_0,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "specvol_sso_0",
                                    "specvol_sso_0_docstring",
                                    0);

    PyDict_SetItemString(d, "specvol_sso_0", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_d_d,
                                    data_sr_from_sp,
                                    types_d_d,
                                    1, 1, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sr_from_sp",
                                    "sr_from_sp_docstring",
                                    0);

    PyDict_SetItemString(d, "sr_from_sp", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_adiabatic_lapse_rate_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "adiabatic_lapse_rate_ice",
                                    "adiabatic_lapse_rate_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "adiabatic_lapse_rate_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_alpha_wrt_t_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "alpha_wrt_t_ice",
                                    "alpha_wrt_t_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "alpha_wrt_t_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_chem_potential_water_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "chem_potential_water_ice",
                                    "chem_potential_water_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "chem_potential_water_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_cp_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "cp_ice",
                                    "cp_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "cp_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_ct_from_entropy,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_from_entropy",
                                    "ct_from_entropy_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_from_entropy", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_ct_from_pt,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_from_pt",
                                    "ct_from_pt_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_from_pt", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_ct_maxdensity,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_maxdensity",
                                    "ct_maxdensity_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_maxdensity", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_enthalpy_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "enthalpy_ice",
                                    "enthalpy_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "enthalpy_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_entropy_from_ct,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "entropy_from_ct",
                                    "entropy_from_ct_docstring",
                                    0);

    PyDict_SetItemString(d, "entropy_from_ct", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_entropy_from_pt,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "entropy_from_pt",
                                    "entropy_from_pt_docstring",
                                    0);

    PyDict_SetItemString(d, "entropy_from_pt", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_entropy_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "entropy_ice",
                                    "entropy_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "entropy_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_entropy_part_zerop,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "entropy_part_zerop",
                                    "entropy_part_zerop_docstring",
                                    0);

    PyDict_SetItemString(d, "entropy_part_zerop", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_gibbs_ice_part_t,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "gibbs_ice_part_t",
                                    "gibbs_ice_part_t_docstring",
                                    0);

    PyDict_SetItemString(d, "gibbs_ice_part_t", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_gibbs_pt0_pt0,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "gibbs_pt0_pt0",
                                    "gibbs_pt0_pt0_docstring",
                                    0);

    PyDict_SetItemString(d, "gibbs_pt0_pt0", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_grav,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "grav",
                                    "grav_docstring",
                                    0);

    PyDict_SetItemString(d, "grav", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_helmholtz_energy_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "helmholtz_energy_ice",
                                    "helmholtz_energy_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "helmholtz_energy_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_internal_energy_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "internal_energy_ice",
                                    "internal_energy_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "internal_energy_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_kappa_const_t_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "kappa_const_t_ice",
                                    "kappa_const_t_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "kappa_const_t_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_kappa_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "kappa_ice",
                                    "kappa_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "kappa_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_latentheat_evap_ct,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "latentheat_evap_ct",
                                    "latentheat_evap_ct_docstring",
                                    0);

    PyDict_SetItemString(d, "latentheat_evap_ct", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_latentheat_evap_t,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "latentheat_evap_t",
                                    "latentheat_evap_t_docstring",
                                    0);

    PyDict_SetItemString(d, "latentheat_evap_t", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_latentheat_melting,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "latentheat_melting",
                                    "latentheat_melting_docstring",
                                    0);

    PyDict_SetItemString(d, "latentheat_melting", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_melting_ice_equilibrium_sa_ct_ratio,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "melting_ice_equilibrium_sa_ct_ratio",
                                    "melting_ice_equilibrium_sa_ct_ratio_docstring",
                                    0);

    PyDict_SetItemString(d, "melting_ice_equilibrium_sa_ct_ratio", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_melting_ice_equilibrium_sa_ct_ratio_poly,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "melting_ice_equilibrium_sa_ct_ratio_poly",
                                    "melting_ice_equilibrium_sa_ct_ratio_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "melting_ice_equilibrium_sa_ct_ratio_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_melting_seaice_equilibrium_sa_ct_ratio,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "melting_seaice_equilibrium_sa_ct_ratio",
                                    "melting_seaice_equilibrium_sa_ct_ratio_docstring",
                                    0);

    PyDict_SetItemString(d, "melting_seaice_equilibrium_sa_ct_ratio", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_melting_seaice_equilibrium_sa_ct_ratio_poly,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "melting_seaice_equilibrium_sa_ct_ratio_poly",
                                    "melting_seaice_equilibrium_sa_ct_ratio_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "melting_seaice_equilibrium_sa_ct_ratio_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_p_from_z,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "p_from_z",
                                    "p_from_z_docstring",
                                    0);

    PyDict_SetItemString(d, "p_from_z", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_pot_enthalpy_ice_freezing,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pot_enthalpy_ice_freezing",
                                    "pot_enthalpy_ice_freezing_docstring",
                                    0);

    PyDict_SetItemString(d, "pot_enthalpy_ice_freezing", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_pot_enthalpy_ice_freezing_poly,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pot_enthalpy_ice_freezing_poly",
                                    "pot_enthalpy_ice_freezing_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "pot_enthalpy_ice_freezing_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_pressure_coefficient_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pressure_coefficient_ice",
                                    "pressure_coefficient_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "pressure_coefficient_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_pt0_from_t_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pt0_from_t_ice",
                                    "pt0_from_t_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "pt0_from_t_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_pt_from_ct,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pt_from_ct",
                                    "pt_from_ct_docstring",
                                    0);

    PyDict_SetItemString(d, "pt_from_ct", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_pt_from_entropy,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pt_from_entropy",
                                    "pt_from_entropy_docstring",
                                    0);

    PyDict_SetItemString(d, "pt_from_entropy", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_rho_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "rho_ice",
                                    "rho_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "rho_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_sigma0,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sigma0",
                                    "sigma0_docstring",
                                    0);

    PyDict_SetItemString(d, "sigma0", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_sigma1,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sigma1",
                                    "sigma1_docstring",
                                    0);

    PyDict_SetItemString(d, "sigma1", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_sigma2,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sigma2",
                                    "sigma2_docstring",
                                    0);

    PyDict_SetItemString(d, "sigma2", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_sigma3,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sigma3",
                                    "sigma3_docstring",
                                    0);

    PyDict_SetItemString(d, "sigma3", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_sigma4,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sigma4",
                                    "sigma4_docstring",
                                    0);

    PyDict_SetItemString(d, "sigma4", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_sound_speed_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sound_speed_ice",
                                    "sound_speed_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "sound_speed_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_specvol_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "specvol_ice",
                                    "specvol_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "specvol_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_spiciness0,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "spiciness0",
                                    "spiciness0_docstring",
                                    0);

    PyDict_SetItemString(d, "spiciness0", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_spiciness1,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "spiciness1",
                                    "spiciness1_docstring",
                                    0);

    PyDict_SetItemString(d, "spiciness1", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_spiciness2,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "spiciness2",
                                    "spiciness2_docstring",
                                    0);

    PyDict_SetItemString(d, "spiciness2", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_t_from_pt0_ice,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "t_from_pt0_ice",
                                    "t_from_pt0_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "t_from_pt0_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_d,
                                    data_z_from_p,
                                    types_dd_d,
                                    1, 2, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "z_from_p",
                                    "z_from_p_docstring",
                                    0);

    PyDict_SetItemString(d, "z_from_p", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_adiabatic_lapse_rate_from_ct,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "adiabatic_lapse_rate_from_ct",
                                    "adiabatic_lapse_rate_from_ct_docstring",
                                    0);

    PyDict_SetItemString(d, "adiabatic_lapse_rate_from_ct", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_alpha,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "alpha",
                                    "alpha_docstring",
                                    0);

    PyDict_SetItemString(d, "alpha", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_alpha_on_beta,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "alpha_on_beta",
                                    "alpha_on_beta_docstring",
                                    0);

    PyDict_SetItemString(d, "alpha_on_beta", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_alpha_wrt_t_exact,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "alpha_wrt_t_exact",
                                    "alpha_wrt_t_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "alpha_wrt_t_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_beta,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "beta",
                                    "beta_docstring",
                                    0);

    PyDict_SetItemString(d, "beta", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_beta_const_t_exact,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "beta_const_t_exact",
                                    "beta_const_t_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "beta_const_t_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_c_from_sp,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "c_from_sp",
                                    "c_from_sp_docstring",
                                    0);

    PyDict_SetItemString(d, "c_from_sp", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_cabbeling,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "cabbeling",
                                    "cabbeling_docstring",
                                    0);

    PyDict_SetItemString(d, "cabbeling", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_chem_potential_water_t_exact,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "chem_potential_water_t_exact",
                                    "chem_potential_water_t_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "chem_potential_water_t_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_cp_t_exact,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "cp_t_exact",
                                    "cp_t_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "cp_t_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_ct_freezing,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_freezing",
                                    "ct_freezing_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_freezing", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_ct_freezing_poly,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_freezing_poly",
                                    "ct_freezing_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_freezing_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_ct_from_enthalpy,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_from_enthalpy",
                                    "ct_from_enthalpy_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_from_enthalpy", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_ct_from_enthalpy_exact,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_from_enthalpy_exact",
                                    "ct_from_enthalpy_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_from_enthalpy_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_ct_from_t,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_from_t",
                                    "ct_from_t_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_from_t", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_deltasa_atlas,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "deltasa_atlas",
                                    "deltasa_atlas_docstring",
                                    0);

    PyDict_SetItemString(d, "deltasa_atlas", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_dilution_coefficient_t_exact,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "dilution_coefficient_t_exact",
                                    "dilution_coefficient_t_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "dilution_coefficient_t_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_dynamic_enthalpy,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "dynamic_enthalpy",
                                    "dynamic_enthalpy_docstring",
                                    0);

    PyDict_SetItemString(d, "dynamic_enthalpy", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_enthalpy,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "enthalpy",
                                    "enthalpy_docstring",
                                    0);

    PyDict_SetItemString(d, "enthalpy", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_enthalpy_ct_exact,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "enthalpy_ct_exact",
                                    "enthalpy_ct_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "enthalpy_ct_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_enthalpy_t_exact,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "enthalpy_t_exact",
                                    "enthalpy_t_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "enthalpy_t_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_entropy_from_t,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "entropy_from_t",
                                    "entropy_from_t_docstring",
                                    0);

    PyDict_SetItemString(d, "entropy_from_t", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_entropy_part,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "entropy_part",
                                    "entropy_part_docstring",
                                    0);

    PyDict_SetItemString(d, "entropy_part", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_fdelta,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "fdelta",
                                    "fdelta_docstring",
                                    0);

    PyDict_SetItemString(d, "fdelta", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_internal_energy,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "internal_energy",
                                    "internal_energy_docstring",
                                    0);

    PyDict_SetItemString(d, "internal_energy", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_kappa,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "kappa",
                                    "kappa_docstring",
                                    0);

    PyDict_SetItemString(d, "kappa", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_kappa_t_exact,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "kappa_t_exact",
                                    "kappa_t_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "kappa_t_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_pressure_freezing_ct,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pressure_freezing_ct",
                                    "pressure_freezing_ct_docstring",
                                    0);

    PyDict_SetItemString(d, "pressure_freezing_ct", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_pt0_from_t,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pt0_from_t",
                                    "pt0_from_t_docstring",
                                    0);

    PyDict_SetItemString(d, "pt0_from_t", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_pt_from_t_ice,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pt_from_t_ice",
                                    "pt_from_t_ice_docstring",
                                    0);

    PyDict_SetItemString(d, "pt_from_t_ice", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_rho,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "rho",
                                    "rho_docstring",
                                    0);

    PyDict_SetItemString(d, "rho", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_rho_t_exact,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "rho_t_exact",
                                    "rho_t_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "rho_t_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_sa_freezing_from_ct,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sa_freezing_from_ct",
                                    "sa_freezing_from_ct_docstring",
                                    0);

    PyDict_SetItemString(d, "sa_freezing_from_ct", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_sa_freezing_from_ct_poly,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sa_freezing_from_ct_poly",
                                    "sa_freezing_from_ct_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "sa_freezing_from_ct_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_sa_freezing_from_t,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sa_freezing_from_t",
                                    "sa_freezing_from_t_docstring",
                                    0);

    PyDict_SetItemString(d, "sa_freezing_from_t", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_sa_freezing_from_t_poly,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sa_freezing_from_t_poly",
                                    "sa_freezing_from_t_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "sa_freezing_from_t_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_sa_from_rho,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sa_from_rho",
                                    "sa_from_rho_docstring",
                                    0);

    PyDict_SetItemString(d, "sa_from_rho", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_sa_from_sp_baltic,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sa_from_sp_baltic",
                                    "sa_from_sp_baltic_docstring",
                                    0);

    PyDict_SetItemString(d, "sa_from_sp_baltic", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_saar,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "saar",
                                    "saar_docstring",
                                    0);

    PyDict_SetItemString(d, "saar", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_sound_speed,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sound_speed",
                                    "sound_speed_docstring",
                                    0);

    PyDict_SetItemString(d, "sound_speed", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_sound_speed_t_exact,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sound_speed_t_exact",
                                    "sound_speed_t_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "sound_speed_t_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_sp_from_c,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sp_from_c",
                                    "sp_from_c_docstring",
                                    0);

    PyDict_SetItemString(d, "sp_from_c", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_sp_from_sa_baltic,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sp_from_sa_baltic",
                                    "sp_from_sa_baltic_docstring",
                                    0);

    PyDict_SetItemString(d, "sp_from_sa_baltic", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_specvol,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "specvol",
                                    "specvol_docstring",
                                    0);

    PyDict_SetItemString(d, "specvol", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_specvol_anom_standard,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "specvol_anom_standard",
                                    "specvol_anom_standard_docstring",
                                    0);

    PyDict_SetItemString(d, "specvol_anom_standard", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_specvol_t_exact,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "specvol_t_exact",
                                    "specvol_t_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "specvol_t_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_t_deriv_chem_potential_water_t_exact,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "t_deriv_chem_potential_water_t_exact",
                                    "t_deriv_chem_potential_water_t_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "t_deriv_chem_potential_water_t_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_t_freezing,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "t_freezing",
                                    "t_freezing_docstring",
                                    0);

    PyDict_SetItemString(d, "t_freezing", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_t_freezing_poly,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "t_freezing_poly",
                                    "t_freezing_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "t_freezing_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_t_from_ct,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "t_from_ct",
                                    "t_from_ct_docstring",
                                    0);

    PyDict_SetItemString(d, "t_from_ct", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_d,
                                    data_thermobaric,
                                    types_ddd_d,
                                    1, 3, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "thermobaric",
                                    "thermobaric_docstring",
                                    0);

    PyDict_SetItemString(d, "thermobaric", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddd_d,
                                    data_deltasa_from_sp,
                                    types_dddd_d,
                                    1, 4, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "deltasa_from_sp",
                                    "deltasa_from_sp_docstring",
                                    0);

    PyDict_SetItemString(d, "deltasa_from_sp", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddd_d,
                                    data_enthalpy_diff,
                                    types_dddd_d,
                                    1, 4, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "enthalpy_diff",
                                    "enthalpy_diff_docstring",
                                    0);

    PyDict_SetItemString(d, "enthalpy_diff", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddd_d,
                                    data_melting_ice_sa_ct_ratio,
                                    types_dddd_d,
                                    1, 4, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "melting_ice_sa_ct_ratio",
                                    "melting_ice_sa_ct_ratio_docstring",
                                    0);

    PyDict_SetItemString(d, "melting_ice_sa_ct_ratio", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddd_d,
                                    data_melting_ice_sa_ct_ratio_poly,
                                    types_dddd_d,
                                    1, 4, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "melting_ice_sa_ct_ratio_poly",
                                    "melting_ice_sa_ct_ratio_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "melting_ice_sa_ct_ratio_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddd_d,
                                    data_pot_rho_t_exact,
                                    types_dddd_d,
                                    1, 4, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pot_rho_t_exact",
                                    "pot_rho_t_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "pot_rho_t_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddd_d,
                                    data_pt_from_t,
                                    types_dddd_d,
                                    1, 4, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pt_from_t",
                                    "pt_from_t_docstring",
                                    0);

    PyDict_SetItemString(d, "pt_from_t", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddd_d,
                                    data_sa_from_sp,
                                    types_dddd_d,
                                    1, 4, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sa_from_sp",
                                    "sa_from_sp_docstring",
                                    0);

    PyDict_SetItemString(d, "sa_from_sp", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddd_d,
                                    data_sa_from_sstar,
                                    types_dddd_d,
                                    1, 4, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sa_from_sstar",
                                    "sa_from_sstar_docstring",
                                    0);

    PyDict_SetItemString(d, "sa_from_sstar", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddd_d,
                                    data_sp_from_sa,
                                    types_dddd_d,
                                    1, 4, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sp_from_sa",
                                    "sp_from_sa_docstring",
                                    0);

    PyDict_SetItemString(d, "sp_from_sa", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddd_d,
                                    data_sp_from_sstar,
                                    types_dddd_d,
                                    1, 4, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sp_from_sstar",
                                    "sp_from_sstar_docstring",
                                    0);

    PyDict_SetItemString(d, "sp_from_sstar", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddd_d,
                                    data_sstar_from_sa,
                                    types_dddd_d,
                                    1, 4, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sstar_from_sa",
                                    "sstar_from_sa_docstring",
                                    0);

    PyDict_SetItemString(d, "sstar_from_sa", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddd_d,
                                    data_sstar_from_sp,
                                    types_dddd_d,
                                    1, 4, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "sstar_from_sp",
                                    "sstar_from_sp_docstring",
                                    0);

    PyDict_SetItemString(d, "sstar_from_sp", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddddd_d,
                                    data_melting_seaice_sa_ct_ratio,
                                    types_ddddd_d,
                                    1, 5, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "melting_seaice_sa_ct_ratio",
                                    "melting_seaice_sa_ct_ratio_docstring",
                                    0);

    PyDict_SetItemString(d, "melting_seaice_sa_ct_ratio", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddddd_d,
                                    data_melting_seaice_sa_ct_ratio_poly,
                                    types_ddddd_d,
                                    1, 5, 1,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "melting_seaice_sa_ct_ratio_poly",
                                    "melting_seaice_sa_ct_ratio_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "melting_seaice_sa_ct_ratio_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_dd,
                                    data_ct_first_derivatives,
                                    types_dd_dd,
                                    1, 2, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_first_derivatives",
                                    "ct_first_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_first_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_dd,
                                    data_entropy_first_derivatives,
                                    types_dd_dd,
                                    1, 2, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "entropy_first_derivatives",
                                    "entropy_first_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "entropy_first_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_dd,
                                    data_pot_enthalpy_ice_freezing_first_derivatives,
                                    types_dd_dd,
                                    1, 2, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pot_enthalpy_ice_freezing_first_derivatives",
                                    "pot_enthalpy_ice_freezing_first_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "pot_enthalpy_ice_freezing_first_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_dd,
                                    data_pot_enthalpy_ice_freezing_first_derivatives_poly,
                                    types_dd_dd,
                                    1, 2, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pot_enthalpy_ice_freezing_first_derivatives_poly",
                                    "pot_enthalpy_ice_freezing_first_derivatives_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "pot_enthalpy_ice_freezing_first_derivatives_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_dd,
                                    data_pt_first_derivatives,
                                    types_dd_dd,
                                    1, 2, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pt_first_derivatives",
                                    "pt_first_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "pt_first_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_dd,
                                    data_ct_freezing_first_derivatives,
                                    types_ddd_dd,
                                    1, 3, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_freezing_first_derivatives",
                                    "ct_freezing_first_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_freezing_first_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_dd,
                                    data_ct_freezing_first_derivatives_poly,
                                    types_ddd_dd,
                                    1, 3, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_freezing_first_derivatives_poly",
                                    "ct_freezing_first_derivatives_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_freezing_first_derivatives_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_dd,
                                    data_ct_from_rho,
                                    types_ddd_dd,
                                    1, 3, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_from_rho",
                                    "ct_from_rho_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_from_rho", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_dd,
                                    data_enthalpy_first_derivatives,
                                    types_ddd_dd,
                                    1, 3, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "enthalpy_first_derivatives",
                                    "enthalpy_first_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "enthalpy_first_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_dd,
                                    data_enthalpy_first_derivatives_ct_exact,
                                    types_ddd_dd,
                                    1, 3, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "enthalpy_first_derivatives_ct_exact",
                                    "enthalpy_first_derivatives_ct_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "enthalpy_first_derivatives_ct_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_dd,
                                    data_rho_first_derivatives_wrt_enthalpy,
                                    types_ddd_dd,
                                    1, 3, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "rho_first_derivatives_wrt_enthalpy",
                                    "rho_first_derivatives_wrt_enthalpy_docstring",
                                    0);

    PyDict_SetItemString(d, "rho_first_derivatives_wrt_enthalpy", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_dd,
                                    data_specvol_first_derivatives_wrt_enthalpy,
                                    types_ddd_dd,
                                    1, 3, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "specvol_first_derivatives_wrt_enthalpy",
                                    "specvol_first_derivatives_wrt_enthalpy_docstring",
                                    0);

    PyDict_SetItemString(d, "specvol_first_derivatives_wrt_enthalpy", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_dd,
                                    data_t_freezing_first_derivatives,
                                    types_ddd_dd,
                                    1, 3, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "t_freezing_first_derivatives",
                                    "t_freezing_first_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "t_freezing_first_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_dd,
                                    data_t_freezing_first_derivatives_poly,
                                    types_ddd_dd,
                                    1, 3, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "t_freezing_first_derivatives_poly",
                                    "t_freezing_first_derivatives_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "t_freezing_first_derivatives_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_ct_first_derivatives_wrt_t_exact,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_first_derivatives_wrt_t_exact",
                                    "ct_first_derivatives_wrt_t_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_first_derivatives_wrt_t_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_enthalpy_second_derivatives,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "enthalpy_second_derivatives",
                                    "enthalpy_second_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "enthalpy_second_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_enthalpy_second_derivatives_ct_exact,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "enthalpy_second_derivatives_ct_exact",
                                    "enthalpy_second_derivatives_ct_exact_docstring",
                                    0);

    PyDict_SetItemString(d, "enthalpy_second_derivatives_ct_exact", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_frazil_properties,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "frazil_properties",
                                    "frazil_properties_docstring",
                                    0);

    PyDict_SetItemString(d, "frazil_properties", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_frazil_properties_potential,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "frazil_properties_potential",
                                    "frazil_properties_potential_docstring",
                                    0);

    PyDict_SetItemString(d, "frazil_properties_potential", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_frazil_properties_potential_poly,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "frazil_properties_potential_poly",
                                    "frazil_properties_potential_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "frazil_properties_potential_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_frazil_ratios_adiabatic,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "frazil_ratios_adiabatic",
                                    "frazil_ratios_adiabatic_docstring",
                                    0);

    PyDict_SetItemString(d, "frazil_ratios_adiabatic", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_frazil_ratios_adiabatic_poly,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "frazil_ratios_adiabatic_poly",
                                    "frazil_ratios_adiabatic_poly_docstring",
                                    0);

    PyDict_SetItemString(d, "frazil_ratios_adiabatic_poly", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_rho_alpha_beta,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "rho_alpha_beta",
                                    "rho_alpha_beta_docstring",
                                    0);

    PyDict_SetItemString(d, "rho_alpha_beta", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_rho_first_derivatives,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "rho_first_derivatives",
                                    "rho_first_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "rho_first_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_rho_second_derivatives_wrt_enthalpy,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "rho_second_derivatives_wrt_enthalpy",
                                    "rho_second_derivatives_wrt_enthalpy_docstring",
                                    0);

    PyDict_SetItemString(d, "rho_second_derivatives_wrt_enthalpy", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_specvol_alpha_beta,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "specvol_alpha_beta",
                                    "specvol_alpha_beta_docstring",
                                    0);

    PyDict_SetItemString(d, "specvol_alpha_beta", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_specvol_first_derivatives,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "specvol_first_derivatives",
                                    "specvol_first_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "specvol_first_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddd,
                                    data_specvol_second_derivatives_wrt_enthalpy,
                                    types_ddd_ddd,
                                    1, 3, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "specvol_second_derivatives_wrt_enthalpy",
                                    "specvol_second_derivatives_wrt_enthalpy_docstring",
                                    0);

    PyDict_SetItemString(d, "specvol_second_derivatives_wrt_enthalpy", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddddd_dd,
                                    data_melting_seaice_into_seawater,
                                    types_dddddd_dd,
                                    1, 6, 2,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "melting_seaice_into_seawater",
                                    "melting_seaice_into_seawater_docstring",
                                    0);

    PyDict_SetItemString(d, "melting_seaice_into_seawater", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_ddd,
                                    data_ct_second_derivatives,
                                    types_dd_ddd,
                                    1, 2, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ct_second_derivatives",
                                    "ct_second_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "ct_second_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_ddd,
                                    data_entropy_second_derivatives,
                                    types_dd_ddd,
                                    1, 2, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "entropy_second_derivatives",
                                    "entropy_second_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "entropy_second_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dd_ddd,
                                    data_pt_second_derivatives,
                                    types_dd_ddd,
                                    1, 2, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "pt_second_derivatives",
                                    "pt_second_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "pt_second_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_dddd_ddd,
                                    data_ice_fraction_to_freeze_seawater,
                                    types_dddd_ddd,
                                    1, 4, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "ice_fraction_to_freeze_seawater",
                                    "ice_fraction_to_freeze_seawater_docstring",
                                    0);

    PyDict_SetItemString(d, "ice_fraction_to_freeze_seawater", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddddd_ddd,
                                    data_melting_ice_into_seawater,
                                    types_ddddd_ddd,
                                    1, 5, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "melting_ice_into_seawater",
                                    "melting_ice_into_seawater_docstring",
                                    0);

    PyDict_SetItemString(d, "melting_ice_into_seawater", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddddd_ddd,
                                    data_seaice_fraction_to_freeze_seawater,
                                    types_ddddd_ddd,
                                    1, 5, 3,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "seaice_fraction_to_freeze_seawater",
                                    "seaice_fraction_to_freeze_seawater_docstring",
                                    0);

    PyDict_SetItemString(d, "seaice_fraction_to_freeze_seawater", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddddd,
                                    data_rho_second_derivatives,
                                    types_ddd_ddddd,
                                    1, 3, 5,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "rho_second_derivatives",
                                    "rho_second_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "rho_second_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);

    ufunc_ptr = PyUFunc_FromFuncAndData(funcs_ddd_ddddd,
                                    data_specvol_second_derivatives,
                                    types_ddd_ddddd,
                                    1, 3, 5,  // ndatatypes, nin, nout
                                    PyUFunc_None,
                                    "specvol_second_derivatives",
                                    "specvol_second_derivatives_docstring",
                                    0);

    PyDict_SetItemString(d, "specvol_second_derivatives", ufunc_ptr);
    Py_DECREF(ufunc_ptr);


    return m;
}
