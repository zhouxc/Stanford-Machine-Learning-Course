#! /usr/bin/python
# driver.py

# imports
from __future__ import division
from optparse import OptionParser
import sys
import os

from util import *
from dataset import DataSet
from hmm import *

import sys


def split_into_categories(d):
    """given a dataset d, return a dict mapping categories 
    to arrays of observation sequences.  Only splits the training data"""
    a = {}
    for seqnum in range(len(d.train_output)):
        seq = d.train_output[seqnum]
        category = d.states[d.train_state[seqnum][0]]
        if category in a:
            a[category].append(seq)
        else:
            a[category] = [seq]

    return a


def train_N_state_hmms_from_data(filename, num_states, debug=False):
    """ reads all the data, then split it up into each category, and then
    builds a separate hmm for each category in data """
    dataset = DataSet(filename)
    category_seqs = split_into_categories(dataset)
    
    # Build a hmm for each category in data
    hmms = {}
    for cat, seqs in category_seqs.items():
        if debug:
            print "\n\nLearning %s-state HMM for category %s" % (num_states, cat)
        
        model = HMM(range(num_states), dataset.outputs)
        model.learn_from_observations(seqs, debug)
        hmms[cat] = model
        if debug:
            print "The learned model for %s:" % cat
            print model
    return (hmms, dataset)


 

@print_timing
def compute_classification_performance(hmms, dataset, debug=False):
    if debug:
        print "Classifying test sequences"
    total = 0
    errors = 0
    for seqnum in range(len(dataset.test_output)):
        total += 1
        seq = dataset.test_output[seqnum]
        actual_category = dataset.states[dataset.test_state[seqnum][0]]
        log_probs = [(cat, hmms[cat].log_prob_of_sequence(seq))
                     for cat in hmms.keys()]
        # Want biggest first...
        log_probs.sort(lambda a,b: cmp(b[1], a[1]))
        if debug:
            ll_str = " ".join(["%s=%.4f" % (c, v) for c,v in log_probs])
            #print "Actual: %s; [%s]" % (actual_category, ll_str)

        # Sorted, so the first one is the one we predicted.
        best_cat = log_probs[0][0]
        if actual_category != best_cat:
            errors += 1
    fraction_incorrect = errors * 1.0 / total
    #if debug:
    print "Classification mistakes: %d / %d = %.3f" % (errors, total, fraction_incorrect)
    return fraction_incorrect
    

def main(argv=None):
    if argv is None:
        argv = sys.argv

    usage = "usage: %prog [options] N datafile (pass -h for more info)"
    parser = OptionParser(usage)
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="Print extra debugging info")

    (options, args) = parser.parse_args(argv[1:])
    if len(args) != 2:
        print "ERROR: Missing arguments"
        parser.print_usage()
        sys.exit(1)
        
    num_states = int(args[0])
    filename = args[1]
    filename = normalize_filename(filename)

    # Read all the data, then split it up into each category
    # Build models from the category data files
    hmms, dataset = train_N_state_hmms_from_data(filename, num_states, options.verbose)
    
    # See how well we do in classifying test sequences
    fraction_incorrect = compute_classification_performance(hmms, dataset, options.verbose)   
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
