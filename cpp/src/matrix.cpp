//
// Code for dealing with matrices
//
//
// Yarden Katz <yarden@mit.edu>
//
#include <stdio.h>
using namespace std;

namespace matrix {
    // Multiply two double-type matrices together
    // A - (m x n) matrix
    // B - (n x p) matrix
    // C - resulting (m x p) matrix
    void matrix_mult(double *A, int m, int n, int p, double *B, double *C)
    {
	// for each row of A
	for (int i = 0; i < m; i++)
	{
	    printf("%d\n", A[i]);
	    //cout << A[i];
	    // for each column of A (or equivalently row of B)
	    // for (int j = 0; j < n; j++)
	    // {
	    // 	// initialize as zero
	    // 	C[i][j] = 0.0;
	    // 	// for each row of B
	    // 	for (int k = 0; k < p; k++)
	    // 	{
	    // 	    C[i][j] += (A[i][j] * B[j][k]);
	    // 	}
	    // }
	}
    }

    void test_mat(double *A, int m, int n)
    {
	for (int i = 0; i < m; i++)
	{
	    for (int j = 0; j < n; j++)
	    {
		printf("Element %d, %d is %f\n", i, j, A[i*m + j]);
	    }
	}
    }
}
		    
	
    
    
