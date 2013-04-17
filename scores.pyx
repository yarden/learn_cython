##
## MISO scoring functions in Cython for MCMC sampler
##
import scipy
import scipy.misc
import numpy as np
cimport numpy as np

np.import_array()

cimport cython

from libc.math cimport log

#DTYPE = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_t
ctypedef np.float_t DTYPE_float_t


###
### Variants of sample_reassignments
###
# def sample_reassignments(reads, psi_vector, gene):
#     """
#     Sample a reassignments of reads to isoforms.
#     Note that this does not dependent on the read's current assignment since
#     we're already considering the possibility of 'reassigning' the read to
#     its current assignment in the probability calculations.
#     """
#     reassignment_probs = []
#     all_assignments = transpose(tile(arange(self.num_isoforms, dtype=int32),
#                                      [self.num_reads, 1]))
#     for assignment in all_assignments:
#         # Single-end
#         # Score reads given their assignment
#         read_probs = log_score_reads(reads, assignment, gene)
#         # Score the assignments of reads given Psi vector
#         assignment_probs = \
#             log_score_assignment(assignment, psi_vector)
#         reassignment_p = read_probs + assignment_probs
#         reassignment_probs.append(reassignment_p)
#     reassignment_probs = transpose(array(reassignment_probs))
#     m = transpose(vect_logsumexp(reassignment_probs, axis=1)[newaxis,:])
#     norm_reassignment_probs = exp(reassignment_probs - m)

#     rvsunif = random.rand(self.num_reads, 1)
#     yrvs = (rvsunif<cumsum(norm_reassignment_probs,axis=1)).argmax(1)[:,newaxis]
#     ### Note taking first element of transpose(yrvs)!  To avoid a list of assignments
#     return transpose(yrvs)[0]    


###
### Variants of log_score_assignments
###
@cython.boundscheck(False)
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
    print log_prob_reads
    return log_prob_reads


