/*  wrapper.c ─ CPython-C 封装（纯 C 语法）*/
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

/* 由 computel.cu 导出的符号（保持 C 名字） */
extern float pycompute_l(float l, float *trans_tau_d, int T);
extern void pyinit_global_XYZEW_V();
extern void pyclean_global_XYZEW_V();
extern void pyreset_Vtp1();

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

static PyObject *init_global_XYZEW_V(PyObject *self, PyObject *args)
/*  Python 端签名：compute_l(l : float, tau : 1-D numpy.float32) -> float  */
{

    /* 调 CUDA 实现（它自己负责把数据复制进 GPU） */
    pyinit_global_XYZEW_V();

    return Py_None;
}

static PyObject *clean_global_XYZEW_V(PyObject *self, PyObject *args)
{
    pyclean_global_XYZEW_V();
    return Py_None;
}

static PyObject *reset_Vtp1(PyObject *self, PyObject *args)
{
    pyreset_Vtp1();
    return Py_None;
}

/* ---------------- 模块对象 ---------------- */
static PyMethodDef Methods[] = {
    {"compute_l", py_compute_l, METH_VARARGS,
     "compute_l(l, tau) -> float  (CUDA accelerated)"},
    {"init_global_XYZEW_V", init_global_XYZEW_V, METH_VARARGS,
     "init_global_XYZEW_V() -> None"},
    {"clean_global_XYZEW_V", clean_global_XYZEW_V, METH_VARARGS,
     "clean_global_XYZEW_V() -> None"},
    {"reset_Vtp1", reset_Vtp1, METH_VARARGS,
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
