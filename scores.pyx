##
## MISO scoring functions in Cython for MCMC sampler
##
import scipy
from scipy.misc import logsumexp
import numpy as np
cimport numpy as cnp

cnp.import_array()

cimport cython

#DTYPE = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef cnp.int_t DTYPE_t


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
def log_score_assignment(isoform_nums, psi_vector, scaled_lens, num_reads):
    """
    Score an assignment of a set of reads given psi
    and a gene (i.e. a set of isoforms).
    """
    psi_frag = log(psi_vector) + log(scaled_lens)
    psi_frag = psi_frag - logsumexp(psi_frag)
    psi_frags = tile(psi_frag, [num_reads, 1])
    # NEW VERSION: uses xrange
    return psi_frags[np.arange(num_reads), isoform_nums]


def log_score_assignment(isoform_nums, psi_vector, scaled_lens, num_reads):
    """
    Score an assignment of a set of reads given psi
    and a gene (i.e. a set of isoforms).
    """
    psi_frag = log(psi_vector) + log(scaled_lens)
    psi_frag = psi_frag - logsumexp(psi_frag)
    psi_frags = tile(psi_frag, [num_reads, 1])
    # NEW VERSION: uses xrange
    return psi_frags[np.arange(num_reads), isoform_nums]



###
### Variants of log_score_reads
###
def log_score_reads(cnp.ndarray[DTYPE_t, ndim=2] reads,
                    cnp.ndarray[DTYPE_t, ndim=1] isoform_nums,
                    cnp.ndarray[DTYPE_t, ndim=1] num_parts_per_isoform,
                    cnp.ndarray[DTYPE_t, ndim=1] iso_lens,
                    int read_len,
                    int overhang_len,
                    int num_reads):
    """
    Score a set of reads given their isoform assignments.
    """
    cdef:
       cnp.ndarray[double, ndim=1] log_prob_reads
       cnp.ndarray[long, ndim=1] overhang_excluded
       cnp.ndarray[long, ndim=1] zero_prob_indx
       cnp.ndarray[long, ndim=1] num_reads_possible
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


def multiply_log_score_reads(cnp.ndarray[DTYPE_t, ndim=2] reads,
                             cnp.ndarray[DTYPE_t, ndim=1] isoform_nums,
                             cnp.ndarray[DTYPE_t, ndim=1] num_parts_per_isoform,
                             cnp.ndarray[DTYPE_t, ndim=1] iso_lens,
                             int read_len,
                             int overhang_len,
                             int num_reads):
    """
    Score a set of reads given their isoform assignments.
    """
    cdef:
       cnp.ndarray[double, ndim=1] log_prob_reads
       cnp.ndarray[long, ndim=1] overhang_excluded
       cnp.ndarray[long, ndim=1] num_reads_possible
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
                          cnp.ndarray[DTYPE_t, ndim=2] reads,
                          cnp.ndarray[DTYPE_t, ndim=1] isoform_nums,
                          cnp.ndarray[DTYPE_t, ndim=1] num_parts_per_isoform,
                          cnp.ndarray[DTYPE_t, ndim=1] iso_lens,
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
        

cdef inner_log_score_reads(cnp.ndarray[DTYPE_t, ndim=2] reads,
                           cnp.ndarray[DTYPE_t, ndim=1] isoform_nums,
                           cnp.ndarray[DTYPE_t, ndim=1] num_parts_per_isoform,
                           cnp.ndarray[DTYPE_t, ndim=1] iso_lens,
                           int read_len,
                           int overhang_len,
                           int num_reads):
    """
    Score a set of reads given their isoform assignments.
    """
    cdef:
       cnp.ndarray[double, ndim=1] log_prob_reads
       cnp.ndarray[long, ndim=1] overhang_excluded
       cnp.ndarray[long, ndim=1] zero_prob_indx
       cnp.ndarray[long, ndim=1] num_reads_possible
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


def loop_log_score_reads(cnp.ndarray[DTYPE_t, ndim=2] reads,
                         cnp.ndarray[DTYPE_t, ndim=1] isoform_nums,
                         cnp.ndarray[DTYPE_t, ndim=1] num_parts_per_isoform,
                         cnp.ndarray[DTYPE_t, ndim=1] iso_lens,
                         int read_len,
                         int overhang_len,
                         int num_reads):
    cdef cnp.ndarray[double, ndim=1] log_prob_reads = np.zeros([num_reads])
    cdef int curr_read = 0
    cdef int overhang_excluded = 1
    for iso_num in isoform_nums:
        # Compute overhang excluded for this read
        overhang_excluded = \
            2*(overhang_len - 1) * (num_parts_per_isoform[iso_num] - 1)
        num_reads_possible = \
            (iso_lens[iso_num] - read_len + 1) - overhang_excluded
        # TODO:
        # check for impossible reads
        log_prob_reads[curr_read] = np.log(1) - np.log(num_reads_possible)
        curr_read += 1
    return log_prob_reads


