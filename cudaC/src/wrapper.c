/*  wrapper.c ─ CPython-C 封装（纯 C 语法）*/
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

/* 由 computel.cu 导出的符号（保持 C 名字） */
extern float pycompute_l(float l, float *trans_tau_d, int T);
extern void pyinit_global_XYZEW_V();
extern void pyclean_global_XYZEW_V();
extern void pyreset_Vtp1();
extern void pyinit_global_config(
    int min_X, int max_X, int size_X,
    int min_Y, int max_Y, int size_Y,
    int min_Z, int max_Z, int size_Z,
    int min_E, int max_E, int size_E,
    int min_W, int max_W, int size_W,
    float a1, float a2, float r, float mu, float sigma, int motecalo_nums, float p, float initial_investment
);

/* ---------------- python -> C 桥 ---------------- */
static PyObject *py_compute_l(PyObject *self, PyObject *args)
/*  Python 端签名：compute_l(l : float, tau : 1-D numpy.float32) -> float  */
{
    double  l;          /* Python 的 float -> C double 再 → float  */
    PyObject *tau_obj;  /* 任意可转成 ndarray 的对象               */

    if (!PyArg_ParseTuple(args, "dO", &l, &tau_obj))
        return NULL;

    /* 转成只读、C 连续、float32 ndarray */
    PyArrayObject *tau_arr =
        (PyArrayObject *)PyArray_FROM_OTF(tau_obj,
                                          NPY_FLOAT32,
                                          NPY_ARRAY_IN_ARRAY);
    if (!tau_arr) return NULL;

    int    T   = (int)PyArray_SIZE(tau_arr);
    float *ptr = (float *)PyArray_DATA(tau_arr);

    /* 调 CUDA 实现（它自己负责把数据复制进 GPU） */
    float out = pycompute_l((float)l, ptr, T);

    Py_DECREF(tau_arr);
    return PyFloat_FromDouble((double)out);
}

static PyObject *py_init_global_config(PyObject *self, PyObject *args)
{
    int min_X, max_X, size_X, min_Y, max_Y, size_Y, min_Z, max_Z, size_Z, min_E, max_E, size_E, min_W, max_W, size_W;
    double a1, a2, r, mu, sigma, p, initial_investment;
    int motecalo_nums;
    if (!PyArg_ParseTuple(args, "iiiiiiiiiiiiiiidddddidd", 
        &min_X, &max_X, &size_X, 
        &min_Y, &max_Y, &size_Y, 
        &min_Z, &max_Z, &size_Z, 
        &min_E, &max_E, &size_E,
        &min_W, &max_W, &size_W, 
        &a1, &a2, &r, &mu, &sigma, &motecalo_nums, &p, &initial_investment))
        return NULL;

    printf("min_X: %d, max_X: %d, size_X: %d, min_Y: %d, max_Y: %d, size_Y: %d, min_Z: %d, max_Z: %d, size_Z: %d, min_E: %d, max_E: %d, size_E: %d, min_W: %d, max_W: %d, size_W: %d, a1: %f, a2: %f, r: %f, mu: %f, sigma: %f, motecalo_nums: %d, p: %f, initial_investment: %f\n", 
        min_X, max_X, size_X, min_Y, max_Y, size_Y, min_Z, max_Z, size_Z, min_E, max_E, size_E, min_W, max_W, size_W, a1, a2, r, mu, sigma, motecalo_nums, p, initial_investment);

    pyinit_global_config(
        min_X, max_X, size_X, 
        min_Y, max_Y, size_Y,
        min_Z, max_Z, size_Z, 
        min_E, max_E, size_E, 
        min_W, max_W, size_W, 
        a1, a2, r, mu, sigma, motecalo_nums, p, initial_investment);

    return Py_None;
}

static PyObject *py_init_global_XYZEW_V(PyObject *self, PyObject *args)
/*  Python 端签名：compute_l(l : float, tau : 1-D numpy.float32) -> float  */
{

    /* 调 CUDA 实现（它自己负责把数据复制进 GPU） */
    pyinit_global_XYZEW_V();

    return Py_None;
}

static PyObject *py_clean_global_XYZEW_V(PyObject *self, PyObject *args)
{
    pyclean_global_XYZEW_V();
    return Py_None;
}

static PyObject *py_reset_Vtp1(PyObject *self, PyObject *args)
{
    pyreset_Vtp1();
    return Py_None;
}

/* ---------------- 模块对象 ---------------- */
static PyMethodDef Methods[] = {
    {"compute_l", py_compute_l, METH_VARARGS,
     "compute_l(l, tau) -> float  (CUDA accelerated)"},
    {"init_global_config", py_init_global_config, METH_VARARGS,
     "init_global_config(min_X, max_X, size_X, min_Y, max_Y, size_Y, min_Z, max_Z, size_Z, min_E, max_E, size_E, min_W, max_W, size_W, a1, a2, r, mu, sigma, motecalo_nums, p, initial_investment) -> None"},
    {"init_global_XYZEW_V", py_init_global_XYZEW_V, METH_VARARGS,
     "init_global_XYZEW_V() -> None"},
    {"clean_global_XYZEW_V", py_clean_global_XYZEW_V, METH_VARARGS,
     "clean_global_XYZEW_V() -> None"},
    {"reset_Vtp1", py_reset_Vtp1, METH_VARARGS,
     "reset_Vtp1() -> None"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "compute_l",                /* m_name */
    "CUDA C extension",         /* m_doc  */
    -1,                         /* m_size */
    Methods                     /* m_methods */
};

PyMODINIT_FUNC PyInit_compute_l(void)
{
    import_array();             /* 初始化 NumPy C-API */
    return PyModule_Create(&moduledef);
}
