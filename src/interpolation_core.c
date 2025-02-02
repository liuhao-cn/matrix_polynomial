#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#define CLAMP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

static PyObject* run_interpolation_c(PyObject* self, PyObject* args) {
    PyArrayObject *img, *u, *v;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &img, &PyArray_Type, &u, &PyArray_Type, &v))
        return NULL;

    /* 获取数组维度 */
    npy_intp height = PyArray_DIM(img, 0);
    npy_intp width = PyArray_DIM(img, 1);
    
    /* 创建输出数组 */
    npy_intp dims[2] = {height, width};
    PyArrayObject *output = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (!output) {
        return NULL;
    }
    float *out_data = (float*)PyArray_DATA(output);
    
    /* 获取输入数据指针 */
    float *img_data = (float*)PyArray_DATA(img);
    float *u_data = (float*)PyArray_DATA(u);
    float *v_data = (float*)PyArray_DATA(v);

    /* 创建累积权重数组 */
    float *weight_data = (float*)calloc((size_t)(height * width), sizeof(float));
    if (!weight_data) {
        Py_DECREF(output);
        return PyErr_NoMemory();
    }

    /* 反向映射：从源图像到目标图像 */
    npy_intp src_y, src_x;
    for (src_y = 0; src_y < height; src_y++) {
        for (src_x = 0; src_x < width; src_x++) {
            /* 获取该源像素在目标图像中的位置 */
            float dst_x = u_data[src_y * width + src_x];
            float dst_y = v_data[src_y * width + src_x];
            
            /* 计算目标位置的整数部分和小数部分 */
            int x0 = (int)floorf(dst_x);
            int y0 = (int)floorf(dst_y);
            float dx = dst_x - (float)x0;
            float dy = dst_y - (float)y0;
            
            /* 计算四个相邻目标像素的位置 */
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            /* 计算插值权重 */
            float w00 = (1.0f - dx) * (1.0f - dy);
            float w01 = dx * (1.0f - dy);
            float w10 = (1.0f - dx) * dy;
            float w11 = dx * dy;
            
            /* 源像素值 */
            float src_val = img_data[src_y * width + src_x];
            
            /* 将源像素值分配到目标位置 */
            int i, j;
            for (i = 0; i < 2; i++) {
                int y = (i == 0) ? y0 : y1;
                float wy = (i == 0) ? (1.0f - dy) : dy;
                
                if (y >= 0 && y < height) {
                    for (j = 0; j < 2; j++) {
                        int x = (j == 0) ? x0 : x1;
                        float wx = (j == 0) ? (1.0f - dx) : dx;
                        
                        if (x >= 0 && x < width) {
                            float weight = wx * wy;
                            npy_intp idx = (npy_intp)y * width + x;
                            out_data[idx] += src_val * weight;
                            weight_data[idx] += weight;
                        }
                    }
                }
            }
        }
    }
    
    /* 归一化输出 */
    npy_intp i;
    for (i = 0; i < height * width; i++) {
        if (weight_data[i] > 0.0f) {
            out_data[i] /= weight_data[i];
        }
    }
    
    free(weight_data);
    return (PyObject*)output;
}

static PyMethodDef interpolation_methods[] = {
    {"run_interpolation_c", run_interpolation_c, METH_VARARGS, "Bilinear interpolation in C"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef interpolation_module = {
    PyModuleDef_HEAD_INIT,
    "interpolation_core",
    NULL,
    -1,
    interpolation_methods
};

PyMODINIT_FUNC PyInit_interpolation_core(void) {
    import_array();
    return PyModule_Create(&interpolation_module);
} 