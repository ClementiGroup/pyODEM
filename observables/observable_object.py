""" Observables class object along with objects holding information about each possible observable
The information contained in each observable varies greatly from what experiment you are using

"""
import numpy as np

import max_likelihood.observables.hist_analysis_pkg.HistogramO

class Observable(object):
    def __init__(self):
        pass
        
    def compute_observation(self):
        """ return the observation and observation_std"""
        return 0, 1
    
class Obs(object):
    def __init__():
        self.observables = []
    
    def add_histogram(type, nbins=10, range=(0,10), spacing=None, edges=None, weights=None):
        observable = HistogramO(nbins, range, spacing, edges, weights)
        self.observables.append(observable)
        
    def compute_observations(self, data, weights):
        if weights == None:
            weights = np.ones(np.shape(data)[0])
        
        all_obs = np.array([])
        all_std = np.array([])
        for observable in self.observables:
            obs, std = observable.compute_observed(data, weights)
            all_obs = np.append(all_obs, obs)
            all_std = np.append(all_std, std)
        
        return all_obs, all_std
        