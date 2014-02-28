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

cdef extern from "f2c.h":
   ctypedef int integer
   ctypedef double doublereal

# Import lapack functions.
cdef extern from "clapack.h":
   integer f2c_dgemm(char *transa, char *transb, integer *m,
                     integer *n, integer *k, doublereal *alpha,
                     doublereal *a, integer *lda, doublereal *b,
                     integer *ldb, doublereal *beta, doublereal *c__,
                     integer *ldc)

#cdef extern from "blaswarp.h":
#   pass
   

cdef int main():
    cdef char transa_val = 'N'
    cdef char *transa = &transa_val

    cdef char transb_val = 'N'
    cdef char *transb = &transb_val

    cdef integer m = 3
    cdef integer n = 3
    cdef integer k = 3

    cdef double alpha = 1.0
#    cdef np.ndarray[double, ndim=2, mode="c"] a = \
#      np.asarray(np.ascontiguousarray(np.array([[1, 2, 3],
#                                     [4, 5, 6],
#                                     [7, 8, 9]],
#                                     dtype=float)),
#                                     order="c")

#    cdef np.ndarray[double, ndim=2, mode="c"] b = \
#      np.asarray(np.ascontiguousarray(np.array([[1, 0, 0],
#                                     [0, 0, 0],
#                                     [1, 1, 1]],
#                                     dtype=float)), order="c")

    # result is a double pointer
#    cdef np.ndarray[double, ndim=2, mode="c"] c = \
#      np.asarray(np.ascontiguousarray(np.zeros([3, 3], dtype=float)), order="c")

    cdef double a[3][3]
    a[0][0] = 1
    a[0][1] = 2
    a[0][2] = 3
    a[1][0] = 4
    a[1][1] = 5
    a[1][2] = 6
    a[2][0] = 7
    a[2][1] = 8
    a[2][2] = 9
    cdef double b[3][3]
    b[0][0] = 1
    b[0][1] = 0
    b[0][2] = 0
    b[1][0] = 0
    b[1][1] = 0
    b[1][2] = 0
    b[2][0] = 1
    b[2][1] = 1
    b[2][2] = 1
    cdef doublereal c[3][3]

    cdef integer lda = 3
    cdef integer ldb = 3

    cdef doublereal beta = 0.0

    cdef integer ldc = 3

    print "a[0][0]", a[0][0]

    # Pass C arrays
    f2c_dgemm(&transa_val, &transb_val, &m, &n, &k, &alpha, &a[0][0], &lda, &b[0][0], &ldb, &beta, &c[0][0], &ldc)    
    for i in xrange(3):
        for j in xrange(3):
            print "C(%d,%d) is %.2f" %(i, j, C[i][j])

    return 0

main()

