/* Custom wrappers for GSW functions that are not suitable for ufuncs.
*/


/*
double  *            Returns NULL on error, dyn_height if okay
gsw_geo_strf_dyn_height(double *sa, double *ct, double *p, double p_ref,
        int n_levels, double *dyn_height)
*/

static PyObject *
geo_strf_dyn_height(PyObject *NPY_UNUSED(self), PyObject *args)
{
    PyObject *sa_o, *ct_o, *p_o, *dh_o;
    double p_ref;
    PyArrayObject *sa_a, *ct_a, *p_a, *dh_a;
    int n_levels;
    double *ret = NULL;  /* NULL on error, dyn_height if OK */

    if (!PyArg_ParseTuple(args, "OOOd", &sa_o, &ct_o, &p_o, &p_ref))
        return NULL;

    sa_a = (PyArrayObject *)PyArray_ContiguousFromAny(sa_o, NPY_DOUBLE, 1, 1);
    if (sa_a == NULL)
        goto error;
    ct_a = (PyArrayObject *)PyArray_ContiguousFromAny(ct_o, NPY_DOUBLE, 1, 1);
    if (ct_a == NULL)
        goto error;
    p_a = (PyArrayObject *)PyArray_ContiguousFromAny(p_o, NPY_DOUBLE, 1, 1);
    if (p_a == NULL)
        goto error;

    n_levels = PyArray_DIM(sa_a, 0);
    if (PyArray_DIM(ct_a, 0) != n_levels || PyArray_DIM(p_a, 0) != n_levels)
    {
        PyErr_SetString(PyExc_ValueError,
            "Arguments SA, CT, and p must have the same dimensions.");
        goto error;
    }

    dh_a = PyArray_NewLikeArray(sa_a, NPY_CORDER, NULL, 0);
    if (dh_a == NULL)
        goto error;

    ret = gsw_geo_strf_dyn_height((double *)PyArray_DATA(sa_a),
                                  (double *)PyArray_DATA(ct_a),
                                  (double *)PyArray_DATA(p_a),
                                  p_ref,
                                  n_levels,
                                  (double *)PyArray_DATA(dh_a));

    if (ret == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,
            "gws_geo_strf_dyn_height failed; check input arguments");
        Py_XDECREF(dh_a);
        goto error;
    }

    error:
    Py_XDECREF(sa_a);
    Py_XDECREF(ct_a);
    Py_XDECREF(p_a);

    if (ret == NULL)
        return NULL;

    return (PyObject *)dh_a;
}
