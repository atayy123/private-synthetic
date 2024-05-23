import cython

import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "lib/privBayes_model.h":
    cdef string c_get_model(const int* data,const string& config, double eps, int max_k, int seed, int mode, int m, int n)




@cython.boundscheck(False)
@cython.wraparound(False)
def py_get_model(np.ndarray[int, ndim=2, mode="c"] input not None, config_str, eps, max_k, seed, mod):

    cdef int m, n

    m, n = input.shape[0], input.shape[1]

    model_str = c_get_model(&input[0,0], config_str, eps, max_k, seed, mod, m, n)

    return model_str
