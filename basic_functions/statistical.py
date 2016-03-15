""" Package of basic functions for statistics
All functions should be normalized to take values between 0 and 1
All functions are wrapped for ease of use

TO DO LIST:
1. re-write wrappers to use decorators in python

NOTE: WE SHOULD DISCUSS
Should we normalize the functions so that the integral is 1?
This would matter for gaussians... as a very sharp gaussian should matter a lot more... 
In effect, normalizing by the integral guarantees that the parameter is given its due weight if it's very specific
But then it's not forced to be between 0 and 1... it's forced between 0*a1*a2*a3.... which is not unitless and hard to interpret
"""

import numpy as np

def gaussian(r,r0,width):
    V = np.exp(-((r - r0)**2)/(2.*(width**2)))
    return V
    
def derivative_gaussian(r, r0, width):
    dV = gaussian(r, r0, width) * (-(r - r0) / width**2)
    return dV
    
def wrapped_gaussian(r0, width):
    def new_gaussian(r):
        return gaussian(r,r0,width)
    return new_gaussian

def wrapped_derivative_gaussian(r0, width):
    def new_derivative_gaussian(r):
        return derivative_gaussian(r, r0, width)
    return new_derivative_gaussian
