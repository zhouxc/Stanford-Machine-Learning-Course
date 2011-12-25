#!/usr/bin/env python


def list_index(xs):
    """
    Return a mapping from each element of xs to its index
    """
    m = {}
    for (i, x) in enumerate(xs):
        m[x] = i
    return m

class DataSet:
    """    
    This class provides the following fields:

    d.states      an array containing the names of all of the states
    d.outputs     an array containing the names of all of the outputs

    The set of all training state sequences.  train_state[i] is
    an array representing the i-th sequence of states encountered
    during training (corresponding to train_output[i]).
    train_state[][]

    The set of all training output sequences.  train_output[i] is
    an array representing the i-th sequence of outputs encountered
    during training (corresponding to train_state[i]).
    train_output[][]

    The set of all testing state sequences.  test_state[i] is
    an array representing the i-th sequence of states encountered
    during testing (corresponding to test_output[i]).
    test_state[][]

    The set of all testing output sequences.  test_output[i] is
    an array representing the i-th sequence of outputs encountered
    during testing (corresponding to test_state[i]).
    int test_output[][]

    The constructor reads in data from filename and sets
    all these fields.  See assignment instructions for
    information on the required format of this file.
     /"""


    def __init__(self, filename, debug=False):
        self.debug = debug

        file = open(filename,"r")

        states = set([])
        outputs = set([])
        
        # A sequence is a list of (state, output) tuples
        sequences = []
        seq = []
        switched = False

	for line in file.readlines():
            line = line.strip()
            if len(line) == 0:
                continue

	    if line ==  "." or line == "..":
                # end of sequence
                sequences.append(seq)
                seq = []
                if line == "..":
                    if switched:
                        raise Exception("File must have exactly one '..' line")
                    # Switch to test sequences
                    switched = True
                    train_sequences = sequences
                    sequences = []

            else:
                words = line.split();
                
                state = words[0]
                # Keep track of all the states/outputs
                states.add(state)

                for output in words[1:]:
                    outputs.add(output)
                    seq.append( (state, output) )

        # By the time we get here, better have seen the train/test
        # divider
        if not switched:
            raise Exception("File must have exactly one '..' line")

        # Don't forget to add the last sequence!
        if len(seq) > 0:
            sequences.append(seq)
                    
        # Ok, the sequences we have now are the test ones
        test_sequences = sequences

        # Now that we have all the states and outputs, create a numbering
        self.states = list(states)
        self.states.sort()
        self.outputs = list(outputs)
        self.outputs.sort()
        state_map = list_index(self.states)
        output_map = list_index(self.outputs)

        self.train_state = map((lambda seq: map(lambda p: state_map[p[0]], seq)),
                               train_sequences)
        self.train_output = map((lambda seq: map (lambda p: output_map[p[1]], seq)), 
                               train_sequences)

        self.test_state = map((lambda seq: map (lambda p: state_map[p[0]], seq)), 
                               test_sequences)
        self.test_output = map((lambda seq: map (lambda p: output_map[p[1]], seq)), 
                               test_sequences)

        if self.debug:
            print self

    def __repr__(self):
        return """
States:
%s

Outputs:
%s

Training states:
%s

"Training output:"
%s

"Testing states:"
%s

"Testing output:"
%s

""" % (self.states,
       self.outputs,
       self.train_state,
       self.train_output,
       self.test_state,
       self.test_output)
    

if __name__ == "__main__":
    from sys import argv
    if len(argv) > 1:
        d = DataSet(argv[1], True)
        
