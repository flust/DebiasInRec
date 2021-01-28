""" Example of wrapping a C function that takes C double arrays as input using
    the Numpy declarations from Cython """

# cimport the Cython declarations for numpy
cimport numpy as np

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

# cdefine the signature of our c function
cdef extern from "greedy.h":
    void _get_top_k_by_greedy(double *in_array, int num_batch, int num_item, int k, int *out_array);

# create the wrapper code, with numpy type annotations
def get_top_k_by_greedy(np.ndarray[double, ndim=1, mode="c"] in_array not None, num_batch, num_item, k, np.ndarray[int, ndim=1, mode="c"] out_array not None):
    _get_top_k_by_greedy(<double*> np.PyArray_DATA(in_array), num_batch, num_item, k, <int*> np.PyArray_DATA(out_array))
