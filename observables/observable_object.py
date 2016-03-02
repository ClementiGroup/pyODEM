""" Observables class object along with objects holding information about each possible observable
The information contained in each observable varies greatly from what experiment you are using

"""
import numpy as np


class Observable(object):
    def __init__(self):
        pass
        
    def compute_observation(self):
        """ return the observation and observation_std"""
        return 0, 1
    

