##
## MISO scoring functions in Cython for MCMC sampler
##
import scipy
import scipy.misc
import numpy as np
from numpy cimport *
cimport numpy as np

np.import_array()

cimport cython
from cython_gsl cimport *

from libc.math cimport log
from libc.stdlib cimport rand


cdef extern from "limits.h":
    int INT_MAX

print "MAXINT: "
print INT_MAX

import sys
print sys.maxint

cdef float MY_MAX_INT = float(10000)

#DTYPE = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t
ctypedef np.float_t DTYPE_float_t

#cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

#cdef multinomial(ndarray[double, ndim=1] p, unsigned int N):
#    """
#    from CythonGSL.
#    """
#    cdef:
#       size_t K = p.shape[0]
#       ndarray[uint32_t, ndim=1] n = np.empty_like(p, dtype='uint32')
#    # void gsl_ran_multinomial (const gsl_rng * r, size_t K, unsigned int N, const double p[], unsigned int n[])
#    gsl_ran_multinomial(r, K, N, <double*> p.data, <unsigned int *> n.data)
#    return n


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef DTYPE_float_t my_logsumexp(np.ndarray[DTYPE_float_t, ndim=1] log_vector,
                                int vector_len):
    """
    Log sum exp.

    Parameters:
    -----------

    log_vector : array of floats corresponding to log values.
    vector_len : int, length of vector.

    Returns:
    --------

    Result of log(sum(exp(log_vector)))
    """
    cdef DTYPE_float_t curr_exp_value = 0.0
    cdef DTYPE_float_t sum_of_exps = 0.0
    cdef DTYPE_float_t log_sum_of_exps = 0.0
    cdef int curr_elt = 0
    for curr_elt in xrange(vector_len):
        curr_exp_value = exp(log_vector[curr_elt])
        sum_of_exps += curr_exp_value
    # Now take log of the sum of exp values
    log_sum_of_exps = log(sum_of_exps)
    return log_sum_of_exps


def dirichlet_lnpdf(np.ndarray[double, ndim=1] alpha,
                    np.ndarray[double, ndim=1] vector):
    """
    Wrapper for dirichlet log pdf scoring function.
    """
    cdef int D = vector.size
    return dirichlet_log_pdf_raw(D,
                                 &alpha[0], alpha.strides[0],
                                 &vector[0], vector.strides[0])


@cython.infer_types(True)
cdef double dirichlet_log_pdf_raw(
    int D,
    double* alpha, int alpha_stride,
    double* vector, int vector_stride,
    ):
    """Compute the log of the Dirichlet PDF evaluated at one vector."""

    cdef void* alpha_p = alpha
    cdef void* vector_p = vector

    # first term
    term_a = 0.0

    for d in xrange(D):
        term_a += (<double*>(alpha_p + alpha_stride * d))[0]

    term_a = libc.math.lgamma(term_a)

    # second term
    term_b = 0.0

    for d in xrange(D):
        term_b += libc.math.lgamma((<double*>(alpha_p + alpha_stride * d))[0])

    # third term
    cdef double alpha_d
    cdef double vector_d

    term_c = 0.0

    for d in xrange(D):
        alpha_d = (<double*>(alpha_p + alpha_stride * d))[0]
        vector_d = (<double*>(vector_p + vector_stride * d))[0]

        term_c += (alpha_d - 1.0) * libc.math.log(vector_d)

    # ...
    return term_a - term_b + term_c


def my_cumsum(np.ndarray[double, ndim=1] input_array):
    """
    Return cumulative sum of array.
    """
    # Cumulative sum at every position
    cdef np.ndarray[double, ndim=1] cumsum_array = np.empty_like(input_array)
    cdef int curr_elt = 0
    cdef int num_elts = input_array.shape[0]
    # Current cumulative sum: starts at first element
    cdef double curr_cumsum = 0.0
    for curr_elt in xrange(num_elts):
        cumsum_array[curr_elt] = (input_array[curr_elt] + curr_cumsum)
        curr_cumsum = cumsum_array[curr_elt]
    return cumsum_array
        

def sample_from_multinomial(np.ndarray[double, ndim=1] probs,
                            int N):
    """
    Sample one element from multinomial probabilities vector.

    Assumes that the probabilities sum to 1.

    Parameters:
    -----------

    probs : array, vector of probabilities
    N : int, number of samples to draw
    """
    cdef int num_elts = probs.shape[0]
    # The samples: indices into probs
    cdef np.ndarray[double, ndim=1] samples = np.empty(N)
    # Current random samples
    cdef int random_sample = 0
    # Counters over number of samples and number of
    # elements in probability vector
    cdef int curr_sample = 0
    cdef int curr_elt = 0
    cdef double rand_val# = rand() / MY_MAX_INT
    # Get cumulative sum of probability vector
    cdef np.ndarray[double, ndim=1] cumsum = my_cumsum(probs)
    for curr_sample in xrange(N):
        # Draw random number
        rand_val = (rand() % MY_MAX_INT) / MY_MAX_INT
        for curr_elt in xrange(num_elts):
            # If the current cumulative sum is greater than the
            # random number, assign it the index
            if cumsum[curr_elt] >= rand_val:
                random_sample = curr_elt
                break
        samples[curr_sample] = random_sample
    return samples
        
    


# cdef sample_multinomial(ndarray[double, ndim=1] p, unsigned int N):
#     """
#     Sample from multinomial probabilities vector.  Return
#     position into array.
    
#     Parameters:
#     -----------
#     p : array, probabilities (must sum to 1)
#     N : int, number of elements to draw from multinomial
#     """
#     # Use CythonGSL implementation
#     return gsl_multinomial(p, N)


###
### Variants of sample_reassignments
###
def sample_reassignments(np.ndarray[DTYPE_t, ndim=2] reads,
                         np.ndarray[DTYPE_float_t, ndim=1] psi_vector,
                         np.ndarray[DTYPE_t, ndim=1] iso_lens,
                         np.ndarray[DTYPE_t, ndim=1] scaled_lens,
                         np.ndarray[DTYPE_t, ndim=1] num_parts_per_isoform,
                         int num_reads,
                         int read_len,
                         int overhang_len):
    """
    Sample a reassignments of reads to isoforms.
    Note that this does not depend on the read's current assignment since
    we're already considering the possibility of 'reassigning' the read to
    its current assignment in the probability calculations.
    """
    cdef DTYPE_t num_isoforms = psi_vector.shape[0]
    # Probabilities of reassigning current read to each of the isoforms
    cdef np.ndarray[DTYPE_float_t, ndim=1] reassignment_probs = \
        np.empty(num_isoforms)
    cdef np.ndarray[double, ndim=1] isoform_nums = \
        np.empty(num_isoforms)
    # New assignment of reads to isoforms
    cdef np.ndarray[int, ndim=1] new_assignments = np.empty(num_reads)
    cdef double read_probs = 0.0
    cdef DTYPE_float_t assignment_probs = 0.0
    cdef DTYPE_t curr_read_assignment = 0
    
    # For each read, compute the probability of reassigning it to
    # all of the isoforms and sample a new reassignment
    for curr_read in xrange(num_reads):
        # Compute isoform probabilities
        for curr_isoform in xrange(num_isoforms):
            # Log probabilities of these reads and
            # this assignment of reads to isoforms
            # Score assignment to current read
            isoform_nums[0] = <DTYPE_t>curr_isoform
            read_probs = log_score_reads(reads,
                                         isoform_nums,
                                         num_parts_per_isoform,
                                         iso_lens,
                                         read_len,
                                         overhang_len,
                                         num_reads)
            reassignment_probs[curr_isoform] = read_probs
            #asssignment_probs = log_score_assignment()
            #reassignment_probs[curr_isoform] = \
            #    read_probs + assignment_probs
        # Normalize probabilities
        reassignment_probs = my_logsumexp(reassignment_probs, num_isoforms)
        # Sample new assignment for read
        curr_read_assignment = 0#sample_multinomial(reassignment_probs, 1)
        new_assignments[curr_read] = curr_read_assignment
    return new_assignments



###
### Variants of log_score_assignments
###
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def log_score_assignments(np.ndarray[DTYPE_t, ndim=1] isoform_nums,
                          np.ndarray[DTYPE_float_t, ndim=1] psi_vector,
                          np.ndarray[long, ndim=1] scaled_lens,
                          DTYPE_t num_reads):
    """
    Score an assignment of a set of reads given psi
    and a gene (i.e. a set of isoforms).
    """
    cdef:
       np.ndarray[double, ndim=1] psi_frag
       np.ndarray[double, ndim=2] psi_frags
    psi_frag = np.log(psi_vector) + np.log(scaled_lens)
    psi_frag = psi_frag - scipy.misc.logsumexp(psi_frag)
    psi_frags = np.tile(psi_frag, [num_reads, 1])
    return psi_frags[np.arange(num_reads), isoform_nums]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def loop_log_score_assignments(np.ndarray[DTYPE_t, ndim=1] isoform_nums,
                               np.ndarray[DTYPE_float_t, ndim=1] log_psi_frag_vector,
                               int num_reads):
    """
    Score an assignment of a set of reads given psi
    and a gene (i.e. a set of isoforms).
    """
    cdef np.ndarray[DTYPE_float_t, ndim=1] log_scores = np.empty(num_reads)
    cdef DTYPE_float_t curr_log_psi_frag = 0.0
    cdef int curr_read = 0
    cdef int curr_iso_num = 0
    for curr_read in xrange(num_reads):
        curr_iso_num = isoform_nums[curr_read]
        curr_log_psi_frag = log_psi_frag_vector[curr_iso_num]
        log_scores[curr_read] = curr_log_psi_frag 
    return log_scores


###
### Variants of log_score_reads
###
@cython.boundscheck(False)
def log_score_reads(np.ndarray[DTYPE_t, ndim=2] reads,
                    np.ndarray[DTYPE_t, ndim=1] isoform_nums,
                    np.ndarray[DTYPE_t, ndim=1] num_parts_per_isoform,
                    np.ndarray[DTYPE_t, ndim=1] iso_lens,
                    DTYPE_t read_len,
                    DTYPE_t overhang_len,
                    DTYPE_t num_reads):
    """
    Score a set of reads given their isoform assignments.
    """
    cdef:
       np.ndarray[double, ndim=1] log_prob_reads
       np.ndarray[long, ndim=1] overhang_excluded
       np.ndarray[long, ndim=1] zero_prob_indx
       np.ndarray[long, ndim=1] num_reads_possible
       double log_one = log(1)
    # The probability of a read being assigned to an isoform that
    # could not have generated it (i.e. where the read is not a
    # substring of the isoform) is zero.  Check for consistency
    overhang_excluded = \
        2*(overhang_len - 1)*(num_parts_per_isoform[isoform_nums] - 1)
    # The number of reads possible is the number of ways a 36 nt long read can be
    # aligned onto the length of the current isoforms, minus the number of positions
    # that are ruled out due to overhang constraints.
    num_reads_possible = \
        (iso_lens[isoform_nums] - read_len + 1) - overhang_excluded
    log_prob_reads = log_one - np.log(num_reads_possible)
    zero_prob_indx = np.nonzero(reads[np.arange(num_reads), isoform_nums] == 0)[0]
    # Assign probability 0 to reads inconsistent with assignment
    log_prob_reads[zero_prob_indx] = -1# * np.inf
    return log_prob_reads


@cython.boundscheck(False)
def multiply_log_score_reads(np.ndarray[DTYPE_t, ndim=2] reads,
                             np.ndarray[DTYPE_t, ndim=1] isoform_nums,
                             np.ndarray[DTYPE_t, ndim=1] num_parts_per_isoform,
                             np.ndarray[DTYPE_t, ndim=1] iso_lens,
                             int read_len,
                             int overhang_len,
                             int num_reads):
    """
    Score a set of reads given their isoform assignments.
    """
    cdef:
       np.ndarray[double, ndim=1] log_prob_reads
       np.ndarray[long, ndim=1] overhang_excluded
       np.ndarray[long, ndim=1] num_reads_possible
    # The probability of a read being assigned to an isoform that
    # could not have generated it (i.e. where the read is not a
    # substring of the isoform) is zero.  Check for consistency
    overhang_excluded = \
        2*(overhang_len - 1)*(num_parts_per_isoform[isoform_nums] - 1)
    # The number of reads possible is the number of ways a 36 nt long read can be
    # aligned onto the length of the current isoforms, minus the number of positions
    # that are ruled out due to overhang constraints.
    num_reads_possible = \
        (iso_lens[isoform_nums] - read_len + 1) - overhang_excluded
    log_prob_reads = np.log(1) - np.log(num_reads_possible)
    # Assign probability 0 to reads inconsistent with assignment
    log_prob_reads[np.nonzero(reads[np.arange(num_reads), isoform_nums] == 0)[0]] = \
        -1 * np.inf
    return log_prob_reads



def outer_log_score_reads(int num_calls,
                          np.ndarray[DTYPE_t, ndim=2] reads,
                          np.ndarray[DTYPE_t, ndim=1] isoform_nums,
                          np.ndarray[DTYPE_t, ndim=1] num_parts_per_isoform,
                          np.ndarray[DTYPE_t, ndim=1] iso_lens,
                          int read_len,
                          int overhang_len,
                          int num_reads):
    for n in range(num_calls):
        inner_log_score_reads(reads,
                              isoform_nums,
                              num_parts_per_isoform,
                              iso_lens,
                              read_len,
                              overhang_len,
                              num_reads)
        

cdef inner_log_score_reads(np.ndarray[DTYPE_t, ndim=2] reads,
                           np.ndarray[DTYPE_t, ndim=1] isoform_nums,
                           np.ndarray[DTYPE_t, ndim=1] num_parts_per_isoform,
                           np.ndarray[DTYPE_t, ndim=1] iso_lens,
                           int read_len,
                           int overhang_len,
                           int num_reads):
    """
    Score a set of reads given their isoform assignments.
    """
    cdef:
       np.ndarray[double, ndim=1] log_prob_reads
       np.ndarray[long, ndim=1] overhang_excluded
       np.ndarray[long, ndim=1] zero_prob_indx
       np.ndarray[long, ndim=1] num_reads_possible
    # The probability of a read being assigned to an isoform that
    # could not have generated it (i.e. where the read is not a
    # substring of the isoform) is zero.  Check for consistency
    overhang_excluded = \
        2*(overhang_len - 1)*(num_parts_per_isoform[isoform_nums] - 1)
    # The number of reads possible is the number of ways a 36 nt long read can be
    # aligned onto the length of the current isoforms, minus the number of positions
    # that are ruled out due to overhang constraints.
    num_reads_possible = \
        (iso_lens[isoform_nums] - read_len + 1) - overhang_excluded
    log_prob_reads = np.log(1) - np.log(num_reads_possible)
    zero_prob_indx = np.nonzero(reads[np.arange(num_reads), isoform_nums] == 0)[0]
    # Assign probability 0 to reads inconsistent with assignment
    log_prob_reads[zero_prob_indx] = -1 * np.inf
    return log_prob_reads


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def loop_log_score_reads(np.ndarray[DTYPE_t, ndim=2] reads,
                         np.ndarray[DTYPE_t, ndim=1] isoform_nums,
                         np.ndarray[DTYPE_t, ndim=1] num_parts_per_isoform,
                         np.ndarray[DTYPE_t, ndim=1] iso_lens,
                         int num_reads,
                         int read_len,
                         int overhang_len):
    cdef np.ndarray[double, ndim=1] log_prob_reads = np.empty(num_reads)
    # Read counter
    cdef int curr_read = 0
    # Isoform counter
    cdef int curr_iso_num = 0
    # Current isoform's length
    cdef int curr_iso_len = 0
    # Number of reads possible in current isoform
    cdef int num_reads_possible = 0
    # Number of overhang excluded positions
    cdef int num_overhang_excluded = 0
    # Constant used in probability calculation
    cdef double log_one_val = log(1)
    for curr_read in xrange(num_reads):
        # For each isoform assignment, score its probability
        # Get the current isoform's number (0,...,K-1 for K isoforms)
        curr_iso_num = isoform_nums[curr_read]
        # Get the isoform's length
        curr_iso_len = iso_lens[curr_iso_num]
        # Compute overhang excluded for this isoform
        num_overhang_excluded = \
            2*(overhang_len - 1) * (num_parts_per_isoform[curr_iso_num] - 1)
        num_reads_possible = \
            (curr_iso_len - read_len + 1) - num_overhang_excluded
        log_prob_reads[curr_read] = log_one_val - log(num_reads_possible)
    return log_prob_reads


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def precomputed_loop_log_score_reads(np.ndarray[DTYPE_t, ndim=2] reads,
                                     np.ndarray[DTYPE_t, ndim=1] isoform_nums,
                                     np.ndarray[DTYPE_t, ndim=1] num_parts_per_isoform,
                                     np.ndarray[DTYPE_t, ndim=1] iso_lens,
                                     int num_reads,
                                     np.ndarray[double, ndim=1] log_num_reads_possible_per_iso):
    cdef np.ndarray[double, ndim=1] log_prob_reads = np.empty(num_reads)
    # Read counter
    cdef int curr_read = 0
    # Isoform counter
    cdef int curr_iso_num = 0
    # Current isoform's length
    cdef int curr_iso_len = 0
    # Constant used in probability calculation
    cdef double log_one_val = log(1)
    cdef double log_zero_prob_val = -2000
    
    for curr_read in xrange(num_reads):
        # For each isoform assignment, score its probability
        # Get the current isoform's number (0,...,K-1 for K isoforms)
        curr_iso_num = isoform_nums[curr_read]
        # Get the isoform's length
        curr_iso_len = iso_lens[curr_iso_num]
        if reads[curr_read, curr_iso_num] == 0:
            # Read consistent with isoform
            log_prob_reads[curr_read] = log_zero_prob_val
        else:
            log_prob_reads[curr_read] = \
                log_one_val - log_num_reads_possible_per_iso[curr_iso_num]
    return log_prob_reads


