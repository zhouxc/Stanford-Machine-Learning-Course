#!/usr/bin/env python 

from util import * 
from numpy import *
from math import log
import copy
import sys


# If PRODUCTION is false, don't do smoothing 

PRODUCTION = True

# Pretty printing for 1D/2D numpy arrays
MAX_PRINTING_SIZE = 30

def format_array(arr):
    s = shape(arr)
    if s[0] > MAX_PRINTING_SIZE or (len(s) == 2 and s[1] > MAX_PRINTING_SIZE):
        return "[  too many values (%s)   ]" % s

    if len(s) == 1:
        return  "[  " + (
            " ".join(["%.6f" % float(arr[i]) for i in range(s[0])])) + "  ]"
    else:
        lines = []
        for i in range(s[0]):
            lines.append("[  " + "  ".join(["%.6f" % float(arr[i,j]) for j in range(s[1])]) + "  ]")
        return "\n".join(lines)



def format_array_print(arr):
    print format_array(arr)

def init_random_model(N, max_obs, seed=None):
    if seed==None:
        random.seed()
    else:
        random.seed(seed)

    # Initialize things to random values
    tran_model = random.random([N,N])
    obs_model  = random.random([N,max_obs])    
    initial    = random.random([N])

    initial    = ones([N])

    # Normalize
    initial    = initial/sum(initial)
    for i in range(N): 
        tran_model[i,:] = tran_model[i,:]/sum(tran_model[i,:])
        obs_model[i,:]  = obs_model[i,:]/sum(obs_model[i,:])
    
    return (initial, tran_model, obs_model)



def string_of_model(model, label):
    (initial, tran_model, obs_model) = model
    return """
Model: %s 
initial: 
%s

transition: 
%s

observation: 
%s
""" % (label, 
       format_array(initial),
       format_array(tran_model),
       format_array(obs_model))

    
def check_model(model):
    """Check that things add to one as they should"""
    (initial, tran_model, obs_model) = model
    for state in range(len(initial)):
        assert((abs(sum(tran_model[state,:]) - 1)) <= 0.01)
        assert((abs(sum(obs_model[state,:]) - 1)) <= 0.01)
        assert((abs(sum(initial) - 1)) <= 0.01)


def print_model(model, label):
    check_model(model)
    print string_of_model(model, label)    

def max_delta(model, new_model):
    """Return the largest difference between any two corresponding 
    values in the models"""
    return max( [(abs(model[i] - new_model[i])).max() for i in range(len(model))] )


def get_alpha(obs, model):
	""" Returns the array of alphas and the log likelyhood of the sequence.
	
	Note: doing normalization as described in Ghahramani '01--just normalizing
	both alpha and beta to sum to 1 at each time step."""
	
	(initial, tran_model, obs_model) = model
	N = shape(tran_model)[0]
	n = len(obs)
	loglikelyhood = 0
	
	alpha = zeros((n,N))
	alpha[0,:] = initial * obs_model[:,obs[0]]
	normalization = sum(alpha[0,:])
	alpha[0,:] /= normalization
	loglikelyhood += log(normalization)
	
	for t in range(1,n):
		for j in range(N):
			s = sum(tran_model[:,j]*alpha[t-1,:])
			alpha[t,j] = s * obs_model[j,obs[t]]
		normalization = sum(alpha[t,:])
		loglikelyhood += log(normalization)
		alpha[t,:] /= normalization
		
	return alpha, loglikelyhood


def get_beta(obs,model):
	""" Note: doing normalization as described in Ghahramani '01--just normalizing
	both alpha and beta to sum to 1 at each time step."""
	
	(initial, tran_model, obs_model) = model
	N = shape(tran_model)[0]
	n = len(obs)
	# beta[time,state]
	beta = zeros((n,N))
	beta[n-1,:] = ones(N) / N
	for t in range(n-2,-1,-1):
		for i in range(N):
			beta[t,i] = sum(tran_model[i,:]*obs_model[:,obs[t+1]]*beta[t+1,:])
		normalization = sum(beta[t,:])
		beta[t,:] /= normalization
	return beta


def get_gamma(alpha, beta):
	(n,N) = shape(alpha)
	gamma = zeros((n,N))
	for t in range(n):
		normalization = sum(alpha[t,:]*beta[t,:])
		gamma[t,:] = alpha[t,:] * beta[t,:] / normalization
	return gamma


def get_xi(obs,alpha, beta, model):
	(initial, tran_model, obs_model) = model
	N = shape(tran_model)[0]
	n = len(obs)
	xi = zeros((n, N, N))
	for t in range(n-1):
		s = 0
		for i in range(N):
			for j in range(N):
				xi[t,i,j] = alpha[t,i] * tran_model[i,j] * obs_model[j,obs[t+1]] * beta[t+1,j]
				s += xi[t,i,j]
		xi[t,:,:] = xi[t,:,:] / s
	return xi


def compute_expectation_step(obs, N, N_ho, N_h1h2, N_h1, N_h, model, debug=False):
	""" E-step, update the sufficient statistics given the current model,
	and return the loglikelihood of the dataset under the current model
	
	obs: the observation sequences in the training data
	
	N: number of hidden states
	
	the sufficient statistics, refer to lecture 15 notes, p13
	all are stored in numpy arrays
	N_ho: expected number of times in the training data that
		an observation is the output in hidden state.
		It is a numpy array with the number of rows
		equal to the number of hidden states (N)
		and the number of cols equal to the number of observations (M)
	N_h1h2: expected number of times a transition from one hidden state to another
	N_h1: expected number of times in each initial state
	N_h: expected of times in each state at all (used for obs model)
	
	model: the current hmm model of initial, transition and observation probs
	debug: for printing out model parameters or not, set to True by -v option in command line
	
	Return dataset_logliklihood
	note get_alpha() returns the likelihood of an observation seq and
	note that functions for getting beta, xi and gamma values are also implemented for you"""
	
	datasetLoglikelihood = 0.0
	for obser in obs:
		(alpha , loglikelihood) = get_alpha(obser , model)
		beta = get_beta(obser , model)
		gamma = get_gamma(alpha , beta)
		xi = get_xi(obser , alpha , beta , model)
		
		datasetLoglikelihood += loglikelihood

		V = shape(N_ho)[1]
		n = len(obser)
		
		for i in range(N):
			for j in range(V):
				N_ho[i , j] += sum([gamma[t , i] for t in range(n) if obser[t] == j])
	
		for i in range(N):
			for j in range(N):
				N_h1h2[i , j] += sum(xi[:, i , j])
	
		N_h1 += gamma[0 , :]
		N_h += sum(gamma , 0)

	return datasetLoglikelihood


def compute_maximization_step(N, M, N_ho, N_h1h2, N_h1, N_h, model, debug=False):
	"""M-step, update the hmm model by using the incoming sufficient statistics, and return an updated model
	model = (initial, tran_model, obs_model)
	
	N: number of hidden states
	M: number of possible observations
	
	the sufficient statistics, refer to lecture 15 notes, p13,
	all are stored in numpy arrays
	N_ho: expected number of times in the training data that
		an observation is the output in hidden state.
		It is a numpy array with the number of rows
		equal to the number of hidden states (N)
		and the number of cols equal to the number of observations (M)
	N_h1h2: expected number of times a transition from one hidden state to another
	N_h1: expected number of times in each initial state 
	N_h: expected of times in each state at all (used for obs model)
	
	model: the current hmm model of initial, transition and observation probs
	debug: for printing out model parameters or not, set to True by -v option in command line
	
	Return model, an updated hmm model of initial, transition and observation probs
	"""
	
	(initial , tran_model , obs_model) = model

	initial = N_h1 / sum(N_h1)
	for i in range(N):
		tran_model[i , :] = N_h1h2[i , :] / sum(N_h1h2[i , :])
	
	for i in range(N):
		obs_model[i , :] = N_ho[i , :] / sum(N_ho[i , :])

	return (initial , tran_model , obs_model)



# Note: This implementation is as presented in the Rabiner '89 HMM tutorial.
# Variable definitions
# obs    = list of numpy arrays representing multiple observation sequences
# K = the number of observation sequences
# N = num hidden states 
# M = number of possible observations (assuming 0-indexed)
# num_iters = maximum number of iterations allowed (if set to 0 then no limit)
# For each observation sequence:
#   n = number of observations in the sequence.  (indexed 0..n-1)
def baumwelch(obs,N,M, num_iters=0, debug=True,init_model=None, flag=False):    
    K = len(obs)

    if debug:
        print "K=%d N=%d  M=%d" % (K, N, M)

    smoothing = PRODUCTION
    if debug:
        print "smoothing", PRODUCTION

    if init_model == None:
        if debug:
            seed = 42
        else:
            # Just making things deterministic for now.
            # Change to "seed = None" if you want to experiment with
            # random restart, for example.
            seed = 42   
        model = init_random_model(N,M, seed)
    else:
        model = init_model

    if debug:
        print_model(model, "Initial model")
    
    # Loop variables
    iters = 1
    # Keep track of the likelihood of the observation sequences
    loglikelihoods = []    
    while True:
        if debug:
            print "\n\n======= Starting iteration %d ========" % iters
        # Pull out latest parameters
        #(initial, tran_model, obs_model) = model

        if smoothing:
            # Using prior that we've been in every state once, and seen
            # uniform everything.
            N_ho = ones((N,M)) / M
            N_h1h2 = ones((N,N)) / N
            # Number of times in each initial state (for init model)
            N_h1 = ones(N) / N
        
            # Number of times in each state at all (for obs model)
            N_h = ones(N)
        else:
            N_ho = zeros((N,M))
            N_h1h2 = zeros((N,N))
            # Number of times in each initial state (for init model)
            N_h1 = zeros(N)
        
            # Number of times in each state at all (for obs model)
            N_h = zeros(N)


        old_model = copy.deepcopy(model)
        
        #### Expectation step ####
        #N_ho, N_h1h2, N_h1, N_h are numpy arrays and are passed by reference, updated through "side-effects"
        dataset_loglikelihood = compute_expectation_step(obs, N, N_ho, N_h1h2, N_h1, N_h, model, debug)
        loglikelihoods.append(dataset_loglikelihood)

        ### Maximization step ###
        model = compute_maximization_step(N, M, N_ho, N_h1h2, N_h1, N_h, model, debug)
        

        # Termination
        if debug:
            print_model(model, "After %d iterations" % iters)
        delta = max_delta(model, old_model)
        if debug:
            print "Iters = %d, delta = %f, Log prob of sequences: %f" % (
            iters, delta, loglikelihoods[-1])
        sys.stdout.flush()

        iters += 1

        improvement = 1
        if len(loglikelihoods) > 1:
            cur = loglikelihoods[-1]
            prev = loglikelihoods[-2]
            
            improvement = (cur-prev) / abs(prev)

        # Two ways to stop: 
        # (1) the probs stop changing
        epsilon = 0.001
        if delta < epsilon:
            if debug:
                print "Converged to within %f!\n\n" % epsilon
            break
        
        # (2) the improvement in log likelyhood is too small to bother
        smaller = 0.0002
        if improvement < smaller:
            if debug:
                print "Converged. Log likelyhood improvement was less that %f.\n\n" % smaller
            break
        
        if num_iters:
            if iters-1 == num_iters:
                if debug:
                    print "Maximum number of iterations (%d iterations) reached.\n\n" % (iters-1)
                break

    (initial, tran_model, obs_model) = model
    if not flag:
        return tran_model, obs_model, initial
    else:
        return tran_model, obs_model, initial, loglikelihoods






class HMM:
    """ HMM Class that defines the parameters for HMM """
    def __init__(self, states, outputs):
        """If the hmm is going to be trained from data with labeled states,
        states should be a list of the state names.  If the HMM is
        going to trained using EM, states can just be range(num_states)."""
        self.states = states
        self.outputs = outputs
        n_s = len(states)
        n_o = len(outputs)
        self.num_states = n_s
        self.num_outputs = n_o
        self.initial = zeros(n_s)
        self.transition = zeros([n_s,n_s])
        self.observation = zeros([n_s, n_o])

    def set_hidden_model(self, init, trans, observ):
        """ Debugging function: set the model parameters explicitly """
        self.num_states = len(init)
        self.num_outputs = len(observ[0])
        self.initial = array(init)
        self.transition = array(trans)
        self.observation = array(observ)
        self.compute_logs()
        
    def get_model(self):
        return (self.initial, self.transition, self.observation)

    def compute_logs(self):
        """Compute and store the logs of the model"""
        f = lambda xs: map(log, xs)
        self.log_initial = f(self.initial)
        self.log_transition = map(f, self.transition)
        self.log_observation = map(f, self.observation)
        

    def __repr__(self):
        return """states = %s
observations = %s
%s
""" % (" ".join(array_to_string(self.states)), 
       " ".join(array_to_string(self.outputs)), 
       string_of_model((self.initial, self.transition, self.observation), ""))

     
    # declare the @ decorator just before the function, invokes print_timing()
    @print_timing
    def learn_from_labeled_data(self, state_seqs, obs_seqs):
		"""
		Learn the parameters given state and observations sequences. 
		Tje ordering of states in states[i][j] must correspond with observations[i][j].
		Uses Laplacian smoothing to avoid zero probabilities.
		"""

		# Fill this in...
#		self.initial = normalize(...)
#		self.transition = ...
#		self.observation = ...
#		self.compute_logs()
		
		prefix = zeros(self.num_states)
		for state in state_seqs:
			self.initial[state[0]] += 1
			for i in range(len(state) - 1):
				self.transition[state[i]][state[i+1]] += 1
				prefix[state[i]] += 1
			#prefix[state[-1]] += 1

		for i in range(self.num_states):
			self.initial[i] = (self.initial[i] + 1.0) / (len(state_seqs) + self.num_states)
			for j in range(self.num_states):
				self.transition[i][j] = (self.transition[i][j] + 1.0) / (prefix[i] + self.num_states)

		prefix = zeros(self.num_states)
		for i in range(len(state_seqs)):
			for j in range(len(state_seqs[i])) :
				self.observation[state_seqs[i][j]][obs_seqs[i][j]] += 1
				prefix[state_seqs[i][j]] += 1

		for i in range(self.num_states):
			for j in range(self.num_outputs):
				self.observation[i][j] = (self.observation[i][j] + 1.0) / (prefix[i] + self.num_outputs)

		self.compute_logs()
                     
    # declare the @ decorator just before the function, invokes print_timing()
    @print_timing
    def learn_from_observations(self, instances, debug=False, flag=False):
        """
        Learn hmm parameters based on the specified instances.
        This would find the maximum likelyhood transition model,
        observation model, and initial probabilities.
        """
        #def baumwelch(obs,N,M, num_iters=0, debug=False,init_model=None, flag=False):   
        loglikelihoods = None
        if not flag:
            (self.transition, 
             self.observation,
             self.initial) = baumwelch(instances,
                                       len(self.states), 
                                       len(self.outputs), 
                                       0,
                                       debug)
        else:
            (self.transition, 
             self.observation,
             self.initial,
             loglikelihoods) = baumwelch(instances,
                                       len(self.states), 
                                       len(self.outputs), 
                                       0,
                                       debug, None, flag)
            
        
        self.compute_logs()

        if flag:
            return loglikelihoods    

    # Return the log probability that this hmm assigns to a particular output
    # sequence
    def log_prob_of_sequence(self, sequence):
        model = (self.initial, self.transition, self.observation) 
        alpha, loglikelyhood = get_alpha(sequence, model)

        return loglikelyhood

    def most_likely_states(self, sequence, debug=False):
        """Return the most like sequence of states given an output sequence.
        Uses Viterbi algorithm to compute this.
        """
        # Code modified from wikipedia
        # Change this to use logs
       
        cnt = 0
        states = range(0, self.num_states)
        T = {}
        for state in states:
            ##          V.path   V. prob.
            output = sequence[0]
            p = self.log_initial[state] + self.log_observation[state][output]
            T[state] = ([state], p)
        for output in sequence[1:]:
            cnt += 1
            if debug:
                if cnt % 500 == 0:
                    print "processing sequence element %d" % cnt
                    sys.stdout.flush()
            U = {}
            for next_state in states:
                argmax = None
                valmax = None
                for source_state in states:
                    (v_path, v_prob) = T[source_state]
                    p = (self.log_transition[source_state][next_state] +
                         self.log_observation[next_state][output])
                    v_prob += p

                    if valmax is None or v_prob > valmax:
                        argmax = v_path
                        valmax = v_prob
                # Using a nested (reversed) list for performance
                # reasons: the wikipedia code does a list copy, which
                # causes problems with long lists.  The reverse is
                # needed to make the flatten easy.  (This is
                # essentially using a lisp-like Cons cell representation)
                argmax = [next_state, argmax]
                U[next_state] = (argmax, valmax)
            T = U
        ## apply sum/max to the final states:
        argmax = None
        valmax = None
        for state in states:
            (v_path, v_prob) = T[state]
#            print "%s  %s" % T[state]
            if valmax is None or v_prob > valmax:
                argmax = v_path
                valmax = v_prob

        # Kept the list as in reverse order, and nested to make things fast.
        ans = custom_flatten(argmax)
        ans.reverse()
        return ans
   
    def gen_random_sequence(self, n):
        """
        Use the underlying model to generate a sequence of 
        n (state, observation) pairs
        """
        # pick a starting point
        state = random_from_dist(self.initial);
        obs = random_from_dist(self.observation[state])
        seq = [(state,obs)]
        for i in range(1,n):
            state = random_from_dist(self.transition[state])
            obs = random_from_dist(self.observation[state])
            seq.append( (state, obs) )
        return seq

    
def get_wikipedia_model():
    # From the rainy/sunny example on wikipedia (viterbi page)
    hmm = HMM(['Rainy','Sunny'], ['walk','shop','clean'])
    init = [0.6, 0.4]
    trans = [[0.7,0.3], [0.4,0.6]]
    observ = [[0.1,0.4,0.5], [0.6,0.3,0.1]]
    hmm.set_hidden_model(init, trans, observ)
    return hmm

def get_toy_model():
    hmm = HMM(['h1','h2'], ['A','B'])
    init = [0.6, 0.4]
    trans = [[0.7,0.3], [0.4,0.6]]
    observ = [[0.1,0.9], [0.9,0.1]]
    hmm.set_hidden_model(init, trans, observ)
    return hmm
    

def test():
	hmm = get_wikipedia_model()
	print "HMM is:"
	print hmm
	
	seq = [0,1,2]
	logp = hmm.log_prob_of_sequence(seq)
	p = exp(logp)
	print "prob ([walk, shop, clean]): logp= %f  p= %f" % (logp, p)
	print "most likely states (walk, shop, clean) = %s" % hmm.most_likely_states(seq)

if __name__ == "__main__":
    test()
