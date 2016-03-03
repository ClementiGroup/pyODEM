import numpy as np

from pyfexd.observables.hist_analysis_pkg import HistogramO
import pyfexd.basic_functions as bf



class ExperimentalObservables(object):
    def __init__(self):
        self.observables = []
        self.q_functions = []
        self.num_q_functions = 0
        
    def add_histogram(self, exp_file, compute=False, nbins=None, histrange=None, spacing=None, edges=None, errortype="gaussian"):
        if compute == False:
		    observable = HistogramO(nbins, histrange, spacing, edges)
		    self.observables.append(observable)
        
        else:
        	pass #implement a method for analyzing a data file based on the parameters for the histogram given here.
        
        exp_data = np.loadtxt(exp_file)
        
        for i in range(np.shape(exp_data)[0]):
	        self.num_q_functions += 1
	        mean = exp_data[i,0]
	        std = exp_data[i,1]
	        if std == 0:
	            std = 1.0 #std of zero could lead to divide by zero exception. Set to 1 if it's zero (due to no sampling) 
	        self.q_functions.append(bf.statistical.wrapped_gaussian(mean, std))
    
    def compute_observations(self, data, weights=None):
        if weights == None:
            weights = np.ones(np.shape(data)[0])
        
        all_obs = np.array([])
        all_std = np.array([])
        for observable in self.observables:
            obs, std = observable.compute_observed(data, weights)
            all_obs = np.append(all_obs, obs)
            all_std = np.append(all_std, std)
        
        return all_obs, all_std
    
    def get_q_function(self):
        def q_simple(observations):
            if not np.shape(observations)[0] == self.num_q_functions:
                #check to see if number of observations match
                #NOTE: Depending on how values are arranged and passed in this wrapper  function, we can add q_functions later in a script and cause this to fail when trying ot use the same "old" function.
                raise IOError("Number of observations not equal to number of q_functions. This is a problem")
            q_value = 1.0
            for i in range(self.num_q_functions):
                q_value *= self.q_functions[i](observations[i])
            
            return q_value
        
        return q_simple
