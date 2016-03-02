""" Observables class object along with objects holding information about each possible observable
The information contained in each observable varies greatly from what experiment you are using

"""
import numpy as np

import pyfexd.observables.hist_analysis_pkg.HistogramO
import pyfexd.basic_functions as bf


class Observable(object):
    def __init__(self):
        pass
        
    def compute_observation(self):
        """ return the observation and observation_std"""
        return 0, 1
    
class ExperimentalObservables(object):
    def __init__():
        self.observables = []
        self.q_functions = []
        self.num_q_functions = 0
        
    def add_histogram(exp_file, nbins=10, range=(0,10), spacing=None, edges=None, weights=None, errortype="gaussian"):
        observable = HistogramO(nbins, range, spacing, edges, weights)
        self.observables.append(observable)
        for i in range(np.shape(exp_file)[0]):
            self.num_q_functions += 1
            self.q_functions.append(bf.statistical.wrapped_gaussian(exp_file[i,0], exp_file[i,1]))
        
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
                raise IOError("Number of obserations not equal to number of q_functions. This is a problem")
            q_value = 1.0
            for i in range(self.num_q_functions):
                q_value *= self.q_functions[i](observations[i])
            
            return q_value
        
        return q_simple
