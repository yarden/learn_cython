import os
import sys
import time

import numpy as np

import scores

def profile_log_score_reads():
    t1 = time.time()
    num_inc = 3245
    num_exc = 22
    num_com = 39874
    reads = [[1,0]] * num_inc + \
            [[0,1]] * num_exc + \
            [[1,1]] * num_com
    reads = np.array(reads)
    isoform_nums = []
    read_len = 40
    overhang_len = 4
    num_parts_per_isoform = np.array([3, 2], dtype=int)
    iso_lens = np.array([1253, 1172], dtype=int)
    # Assignment of reads to isoforms: assign half of
    # the common reads to isoform 0, half to isoform 1
    isoform_nums = [0]*3245 + [1]*22 + [0]*19937 + [1]*19937
    isoform_nums = np.array(isoform_nums, dtype=int)
    num_reads = len(reads)
    num_calls = 1000
    print "Profiling log_score_reads for %d calls..." %(num_calls)
    t1 = time.time()
    for n in xrange(num_calls):
        scores.log_score_reads(reads,
                               isoform_nums,
                               num_parts_per_isoform,
                               iso_lens,
                               read_len,
                               overhang_len,
                               num_reads)
    t2 = time.time()
    print "log_score_reads took %.2f seconds per %d calls." %(t2 - t1,
                                                              num_calls)
    # --
    print "Profiling multiply_log_score_reads for %d calls..." %(num_calls)
    t1 = time.time()
    for n in xrange(num_calls):
        scores.multiply_log_score_reads(reads,
                                        isoform_nums,
                                        num_parts_per_isoform,
                                        iso_lens,
                                        read_len,
                                        overhang_len,
                                        num_reads)
    t2 = time.time()
    print "MULTIPLY log_score_reads took %.2f seconds per %d calls." %(t2 - t1,
                                                                       num_calls)
    
    print "Profiling OUTER version..."
    t1 = time.time()
    scores.outer_log_score_reads(num_calls,
                                 reads,
                                 isoform_nums,
                                 num_parts_per_isoform,
                                 iso_lens,
                                 read_len,
                                 overhang_len,
                                 num_reads)
    t2 = time.time()
    print "OUTER version took %.2f seconds" %(t2 - t1)
    #t1 = time.time()
    #for n in xrange(num_calls):
    #    scores.loop_log_score_reads(reads,
    #                                isoform_nums,
    #                                num_parts_per_isoform,
    #                                iso_lens,
    #                                read_len,
    #                                overhang_len,
    #                                num_reads)
    #t2 = time.time()
    #print "PYTHON log_score_reads took %.2f seconds per %d calls." %(t2 - t1,
    #                                                          num_calls)
    return
    
    print "Profiling pure PYTHON version..."
    t1 = time.time()
    for n in xrange(num_calls):
        py_log_score_reads(reads,
                           isoform_nums,
                           num_parts_per_isoform,
                           iso_lens,
                           read_len,
                           overhang_len,
                           num_reads)
    t2 = time.time()
    print "PYTHON log_score_reads took %.2f seconds per %d calls." %(t2 - t1,
                                                              num_calls)
    

def py_log_score_reads(reads,
                       isoform_nums,
                       num_parts_per_isoform,
                       iso_lens,
                       read_len,
                       overhang_len,
                       num_reads):
    """
    Score a set of reads given their isoform assignments.
    """
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
    


def main():
    profile_log_score_reads()

if __name__ == "__main__":
    main()
