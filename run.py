print "importing mylapack"

import mylapack
print mylapack
import mylapack.matrix_utils as mu
import numpy as np
print mu
#print mylapack
#import mylapack.lapack

A = np.array([[1,2,3],
              [3,4,5],
              [6,7,8]], dtype=float)
B = np.array([[0,0,1],
              [0,0,1],
              [0,0,1]], dtype=float)

print A
print " times "
print B
print "C: "
print A.shape
print B.shape[1]
C = mu.py_mat_times_mat(A, A.shape[0], A.shape[1], B.shape[1], B)
print C

print "NUMPY:"
print np.matrix(A)*np.matrix(B)
