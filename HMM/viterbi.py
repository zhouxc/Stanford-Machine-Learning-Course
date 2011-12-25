#! /usr/bin/python
# viterbi.py

# imports
from __future__ import division
from optparse import OptionParser
import sys

from util import *
from dataset import DataSet
from hmm import HMM

import sys

@print_timing
def run_viterbi(hmm, d, debug=False):
    """Run the viterbi algorithm for each test sequence in the given dataset"""
    total_error = 0
    total_n = 0
    if debug:
	print "\nRunning viterbi on each test sequence..."
    for i in range(len(d.test_output)):
        if debug:
	    print "Test sequence %d:" % i
	errors = 0
	most_likely = [d.states[j] for j in hmm.most_likely_states(d.test_output[i])]
	actual = [d.states[j] for j in d.test_state[i]]
	n = len(most_likely)
#        print "len(most_likely) = %d  len(actual) = %d" % (n, len(actual))
	for j in range(n):
	    if debug:
		print "%s     %s      %s" % (
		actual[j], most_likely[j], d.outputs[d.test_output[i][j]])
	    if actual[j] != most_likely[j]:
		errors += 1
	    if debug:
		print "errors: %d / %d = %.3f\n" % (errors, n, errors * 1.0 / n)
	total_error += errors
	total_n += n

    err =  total_error * 1.0 / total_n
    if debug:
	print "Total mistakes = %d / %d = %f" % (total_error, total_n, err)
    return err

def train_hmm_from_data(data_filename, debug=False):
    if debug:
	print "\n\nReading dataset %s ..." % data_filename
    data_filename = normalize_filename(data_filename)
    d = DataSet(data_filename)
    #if options.verbose:
    #	print d
    if debug:
	print "Building an HMM from the full training data..."
    hmm = HMM(d.states, d.outputs)
    hmm.learn_from_labeled_data(d.train_state, d.train_output)
    if debug:
	print "The model:"
	print hmm
    return (hmm, d)
	
def main(argv=None):
    if argv is None:
        argv = sys.argv

    usage = "usage: %prog [options] file.data  (pass -h for more info)"
    parser = OptionParser(usage)
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="Print extra debugging info")

    (options, args) = parser.parse_args(argv[1:])
    if len(args) != 1:
        parser.error("Must pass in a datafile")
        
    hmm, d = train_hmm_from_data(args[0], options.verbose)
    err_full = run_viterbi(hmm, d , True)

    return 0

if __name__ == "__main__":
    sys.exit(main())
