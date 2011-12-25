#!/usr/bin/env python

"""
task_hmm.py -- Visualizations for hmms.
"""

from os import path
import random

from tfutils import tftask

from hmm import *
from viterbi import *
from classify import *
from dataset import DataSet

MAX_NUM_HIDDEN_STATES = 8

class Robot(tftask.ChartTask):
    def get_name(self):
        return "Robot experiments -- what path did the robot take to see these sequences of colors?"
    
    def get_priority(self):
        return 1
    def get_description(self):
        return ("Train an HMM for each robot condition, without and with momentum.")
    
    def task(self):
	data_filename = "robot_no_momentum.data"
        hmm, d = train_hmm_from_data(data_filename)
        err_full = run_viterbi(hmm, d)
	
        data_filename_m = "robot_with_momentum.data"
        hmm_m, d_m = train_hmm_from_data(data_filename_m)
        err_full_m = run_viterbi(hmm_m, d_m)
	
	listNames = ["Without momentum", "With momentum"]
	listData = [1-err_full, 1-err_full_m]
	chart = {"chart": {"defaultSeriesType": "column"},
                 "xAxis": {"categories": listNames},
                 "yAxis": {"title": {"text": "Fraction Correct"}},
                 "title": {"text": "HMM performance on infering robot location."}, 
                 "series": [ {"name": "Test set performance", 
	                      "data": listData} ] }
        return chart
    


class WeatherStates_boston_la(tftask.ChartTask):
    def get_name(self):
        return "1st Weather experiments -- which city has this weather, boston or LA?"
    
    def get_priority(self):
        return 2
    
    def get_description(self):
        return ("Train HMMs with different number of hidden states, and see how well they can distinguish between the weather of different cities.")
    
    def task(self):
	num_states = range(1, MAX_NUM_HIDDEN_STATES)
	filename = "weather_bos_la.data"
		
	dataset_performance = []
	for N in num_states:
	    hmms, dataset = train_N_state_hmms_from_data(filename, N)
	    fraction_incorrect = compute_classification_performance(hmms, dataset)
	    dataset_performance.append(1-fraction_incorrect)
	
	chart = {"chart": {"defaultSeriesType": "line"},
                 "xAxis": {"title": {"text": "number of hidden states"}, 
	                    "categories": num_states},
                 "yAxis": {"title": {"text": "Fraction Correct"}, 
	                   "min":0.0, "max":1.0},
                 "title": {"text": "HMM performance on classifying weather sequences by city"}, 
                 "series": [{"name": "Boston_LA",
	                     "data": dataset_performance}]}
        return chart      
    
        
#listNames = ["Boston_LA", "Boston_Seattle", "Boston_Phoenix_Seattle_LA"]
class WeatherStates_boston_seattle(tftask.ChartTask):
    def get_name(self):
        return "2nd Weather experiments -- which city has this weather, boston or seattle?"
    
    def get_priority(self):
        return 3
    
    def get_description(self):
        return ("Train HMMs with different number of hidden states, and see how well they can distinguish between the weather of different cities.")
    
    def task(self):
	num_states = range(1, MAX_NUM_HIDDEN_STATES)
	filename = "weather_bos_sea.data"
		
	dataset_performance = []
	for N in num_states:
	    hmms, dataset = train_N_state_hmms_from_data(filename, N)
	    fraction_incorrect = compute_classification_performance(hmms, dataset)
	    dataset_performance.append(1-fraction_incorrect)
	
	chart = {"chart": {"defaultSeriesType": "line"},
                 "xAxis": {"title": {"text": "number of hidden states"}, 
	                    "categories": num_states},
                 "yAxis": {"title": {"text": "Fraction Correct"}, 
	                   "min":0.0, "max":1.0},
                 "title": {"text": "HMM performance on classifying weather sequences by city"}, 
                 "series": [{"name": "Boston_Seattle",
	                     "data": dataset_performance}]}
        return chart      
        
    
    
class WeatherStates_all(tftask.ChartTask):
    def get_name(self):
        return "3rd Weather experiments -- which city has this weather, boston, seattle, LA, phoenix?"
    
    def get_priority(self):
        return 4
    
    def get_description(self):
        return ("Train HMMs with different number of hidden states, and see how well they can distinguish between the weather of different cities.")
    
    def task(self):
	num_states = range(1, MAX_NUM_HIDDEN_STATES)
	filename = "weather_all.data"
		
	dataset_performance = []
	for N in num_states:
	    hmms, dataset = train_N_state_hmms_from_data(filename, N)
	    fraction_incorrect = compute_classification_performance(hmms, dataset)
	    dataset_performance.append(1-fraction_incorrect)
	
	chart = {"chart": {"defaultSeriesType": "line"},
                 "xAxis": {"title": {"text": "number of hidden states"}, 
	                    "categories": num_states},
                 "yAxis": {"title": {"text": "Fraction Correct"}, 
	                   "min":0.0, "max":1.0},
                 "title": {"text": "HMM performance on classifying weather sequences by city"}, 
                 "series": [{"name": "Boston_Phoenix_Seattle_LA",
	                     "data": dataset_performance}]}
        return chart      
    
    
 
   
class BostonLikelihood(tftask.ChartTask):
    def get_name(self):
        return "4th Weather experiments -- how is the weather here in boston?"
    
    def get_priority(self):
        return 5
    
    def get_description(self):
        return ("Train HMMs with different number of hidden states to see how many hidden states does boston need to model its weather.")
    
    def task(self):
	num_states = range(1, MAX_NUM_HIDDEN_STATES)
	
	filename = "weather_bos_la.data"
	dataset = DataSet(filename)
	category_seqs = split_into_categories(dataset)
	boston_seqs = category_seqs["boston"]
	
	likelihoods = []
	for N in num_states:
	    model = HMM(range(N), dataset.outputs)
	    ll = model.learn_from_observations(boston_seqs, False, True)
	    likelihoods.append(ll[-1])
	
	chart = {"chart": {"defaultSeriesType": "line"},
                 "xAxis": {"title": {"text": "number of hidden states"}, 
	                   "categories": num_states},
                 "yAxis": {"title": {"text": "Fraction Correct"}},
                 "title": {"text": "log likelihood of HMMs modeling boston weather"}, 
                 "series": [{"name": "boston training data",
	                     "data": likelihoods}]}
		
        return chart     

	

def main(argv):
    return tftask.main()

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
   
    
