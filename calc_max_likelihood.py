""" Solves using the max_likelihood_method"""

import numpy as np

import scipy.optimize as optimize

def solve_simplex(func, x0):
    optimal = optimize.minimize(func, x0, method="Nelder-Mead")
    print optimal.message
    
    if not optimal.success == True:
        raise IOError("Minimization failed to find a local minima using the simplex method")
    
    return optimal.x
    

def estimate_new_epsilons(data, data_sets, observables, model):
    #first calculate average value of all observables and associated functions 
    
    expectation_observables = []
    epsilons_functions = []
    for i in data_sets:
        use_data = data[i]
        epsilon_function = model.get_potentials_epsilon(use_data)
        observed = observables.compute_observations(use_data)
    
    #then wrap up a funciton that takes only epsilons, and outputs a value for Q
    
    #Then run the solver
    
    #then return a new set of epsilons... consider adding a method to the model object to automatically udpate epsilons
