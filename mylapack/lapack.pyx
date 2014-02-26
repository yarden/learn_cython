cimport numpy as np
import numpy as np
cimport lapack
print "LAPACK interface"
print "Calling dgemm"

#cdef extern from "clapack.h":
#   cdef extern from "f2c.h":
#       pass
#   integer dgemm_(char *transa, char *transb, integer *m, integer *
#                    n, integer *k, doublereal *alpha, doublereal *a, integer *lda,
#                    doublereal *b, integer *ldb, doublereal *beta, doublereal *c__,
#                    integer *ldc)

##
## Define lapack functions to be used
##
cdef extern from "cblas.h":
  enum CBLAS_ORDER:
    CblasRowMajor, CblasColMajor

cdef extern from "f2c.h":
   ctypedef int integer
   ctypedef double doublereal

#cdef extern from "clapack.h":
#   pass

# Import lapack functions.
cdef extern from "clapack.h":
   integer f2c_dgemm(char *transa, char *transb, integer *m,
                     integer *n, integer *k, doublereal *alpha,
                     doublereal *a, integer *lda, doublereal *b,
                     integer *ldb, doublereal *beta, doublereal *c__,
                     integer *ldc)


cdef int main():
    cdef char transa_val = 'N'
    cdef char *transa = &transa_val

    cdef char transb_val = 'N'
    cdef char *transb = &transb_val

    cdef int m = 3
    cdef int n = 3
    cdef int k = 3

    cdef double alpha = 1.0
    cdef np.ndarray[double, ndim=2, mode="c"] a = \
      np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]], dtype=float)

    cdef np.ndarray[double, ndim=2, mode="c"] b = \
      np.array([[1, 0, 0],
                [0, 0, 0],
                [1, 1, 1]], dtype=float)

    # result is a double pointer
    cdef np.ndarray[double, ndim=2, mode="c"] c = \
      np.empty([3, 3], dtype=float)

    cdef int lda = 3
    cdef int ldb = 3

    cdef double beta = 0.0

    cdef int ldc = 0
 
    f2c_dgemm(&transa_val, &transb_val, &m, &n, &k, &alpha, &a[0,0], &lda, &b[0,0], &ldb, &beta, &c[0,0], &ldc)    

    return 0

main()

