#!/usr/bin/env python

"""
test_hmm.py -- unit tests for hmms implemented in hmm.py
"""

from hmm import *
from viterbi import *
from dataset import DataSet
#from util import * 
import functools
import math
import unittest

class HMMsTest(unittest.TestCase):
    # test for learn_from_labeled_data()
    def test_simple_hmm_learning(self):
	state_seq = [[0,1,1,0,1,0,1,1], [0,0,1,0]]
	obs_seq =   [[0,0,1,1,0,0,0,1], [0,1,0,0]]
	hmm = HMM(range(2), range(2))
	hmm.learn_from_labeled_data(state_seq, obs_seq)
	print hmm
	eps = 0.00001
	self.assertTrue(max_delta(hmm.initial, [0.750000,0.250000]) < eps)
	self.assertTrue(max_delta(hmm.transition, 
				 [[0.285714, 0.714286],
	                          [0.571429, 0.428571]]) < eps)
	self.assertTrue(max_delta(hmm.observation,
	                          [[0.625000, 0.375000],
	                           [0.625000, 0.375000]]) < eps)
	
	
def simple_weather_model():
    hmm = HMM(['s1','s2'], ['R','NR'])
    init = [0.7, 0.3]
    trans = [[0.8,0.2],
             [0.1,0.9]]
    observ = [[0.75,0.25], 
              [0.4,0.6]]
    hmm.set_hidden_model(init, trans, observ)
    return hmm

class ViterbiTest(unittest.TestCase):
    def toy_model(self):
	hmm = HMM(['s1','s2'], ['R','NR'])
	init = [0.5, 0.5]
	trans = [[0.2,0.8],
		 [0.8,0.2]]
	observ = [[0.8,0.2], 
		  [0.2,0.8]]
	hmm.set_hidden_model(init, trans, observ)
	return hmm

		
    def test_viterbi_simple_sequence(self):
	hmm = simple_weather_model()
	print "*******************************************"
	print hmm
	print "******************************************"
	seq = [1, 1, 0]  # NR, NR, R
	hidden_seq = hmm.most_likely_states(seq)
	print "most likely states for [NR, NR, R] = %s" % hidden_seq
	self.assertEqual(hidden_seq, [1,1,1])

    def test_viterbi_long_sequence(self):
	hmm = self.toy_model()
	N = 10
	seq = [1,0,1,0,1,0,1,1,0] * 400
	hidden_seq = hmm.most_likely_states(seq, False)
	# Check if we got right answer from the version with logs.
	self.assertEqual(hidden_seq[2000:2010], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1])


class RobotTest(unittest.TestCase):
    def test_small_robot_dataset(self):
	data_filename = "robot_small.data"
	data_filename = normalize_filename(data_filename)
	hmm, d = train_hmm_from_data(data_filename)
	err_full = run_viterbi(hmm, d, True)
	self.assertAlmostEqual(err_full, 2.0/9)
	
class BaumWelchTest(unittest.TestCase):
    def setUp(self):
	# Initialize things to specific values, for testing
	N = 3 # num hidden states 
	# Normalized below
	transition = array([[1.0,1.0,1.0], 
	                    [1.0,1.0,1.0], 
	                    [1.0,1.0,1.0]])
	observation  = array([[1.0,1.0], [3.0,1.0], [1.0,3.0]])
	initial    = ones([N])
    
	# Normalize
	initial    = initial/sum(initial)
	for i in range(N): 
	    transition[i,:] = transition[i,:]/sum(transition[i,:])
	    observation[i,:]  = observation[i,:]/sum(observation[i,:])
	self.model = (initial, transition, observation)
	self.seq = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
	
        
    def test_bw_beta_all_equal(self):
	# check that all betas are the same
	# get_beta() function already implemented in hmm.py as part of the support code
	
	beta = get_beta(self.seq, self.model)
	print "beta: "
	format_array_print (beta)
	num_rows = shape(beta)[0]
	num_cols = shape(beta)[1]
	for r in range(num_rows):
	    for c in range(num_cols):
		self.assertAlmostEqual(beta[r,c], 1.0/3)
    
    def test_bw_gamma_first_col_equal(self):
	# check that the first column of gamma are the same
	# get_gamma(), get_beta(), get_alpha() functions already implemented in hmm.py as part of the support code
	alpha, logp = get_alpha(self.seq, self.model)
	beta = get_beta(self.seq, self.model)
	gamma = get_gamma(alpha, beta)
	print "gamma: "
	format_array_print (gamma)
	gamma_first_value = gamma[0,0]
	num_rows = shape(gamma)[0]
	for r in range(1, num_rows):
	    self.assertAlmostEqual(gamma_first_value, gamma[r, 0])
	
	# Run EM on this one sequence, with the initial model above.
    
	#model = (initial, transition, observation)
	#baumwelch(seq, 3, 2, 1, True, model)
   
class BaumWelchWeatherTest(unittest.TestCase):
    def setUp(self):
	weather_hmm = simple_weather_model()
	self.seqs = [[0, 0], [1, 1, 0]]
	self.init_model = weather_hmm.get_model()
	import hmm
	# turn off smoothing just for unit test, to match numbers from lecture notes
	hmm.PRODUCTION = False  
	
	
    def test_bw_simple_weather_model(self):
	# example from lecture notes 15, p14, just runs one iteration here
	model = baumwelch(self.seqs, 2, 2, 1, True, self.init_model) # just one EM iteration
	(transition, observation, initial) = model
	
	eps = 0.0001
	self.assertTrue(max_delta(initial, [ 0.646592, 0.353408  ]) < eps)

	self.assertTrue(max_delta(transition, [[ 0.841285, 0.158715 ], 
	                                        [ 0.127844, 0.872156 ]]) < eps)
	self.assertTrue(max_delta(observation, [[ 0.731416, 0.268584 ], 
	                                         [ 0.426629, 0.573371 ]]) < eps)
    
   

	
		
		
if __name__ == '__main__':
    unittest.main()
    
