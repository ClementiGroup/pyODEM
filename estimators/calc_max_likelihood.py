""" Optimizes model parameters using a maximum likelihood approach

This module has the functions for optimizing the a quality funciton Q,
referred to as Qfunc or Q throughout. Q has many components, but mainly
you seek to optimize some parameters epsilons, hereby referred to as
epsilons, x0 or x. Epsilons are model dependent. 

"""
import random
import time
import numpy as np
import scipy.optimize as optimize

from estimators_class import EstimatorsObject
from optimizers import function_dictionary 
        
def max_likelihood_estimate(data, data_sets, observables, model, obs_data=None, solver="simplex", logq=False, derivative=None, x0=None, kwargs={}, stationary_distributions=None, K_shift=0, K_shift_step=100, Max_Count=10):
    """ Optimizes model's paramters using a max likelihood method
    
    Args:
        data (array): Contains all the data for a system, First index 
            should be for the frame number, remaining indices are up to 
            the user to handle using the ExperimentalObservables object.
        data_sets (list): List of arrays, where each array contains the 
            indices in the data for each equilibrium state. 
        observables (ExperimentalObservables): An object that is used
            for computing Q function value as well as the observed
            quantities. See: pyfexd/observables/exp_observables for
            full description.
        model (ModelLoader): Object that is used to load data and
            compute potential energies for the data set. See: 
            pyfexd/model_loaders/super_model for full description.
        obs_data (list): Use if data set for computing observables is 
            different from data for computing the energy. List  contains 
            arrays where each array-entry corresponds to the observable 
            in the ExperimentalObservables object. Arrays are specified 
            with first index corresponding to the frame and second index 
            to the data. Default: Use the array specified in data for 
            all observables. 
        solver (str): Optimization procedures. Defaults to Simplex. 
            Available methods include: simplex, anneal, cg, custom.
        logq (bool): Use the logarithmic Q functions. Default: False.
        derivative (bool): True if Q function returns a derivative. 
            False if it does not. Default is None, automatically 
            selected based upon the requested solver.
        x0 (array): Specify starting epsilons for optimization methods. 
            Defaults to current epsilons from the model.
        kwargs (dictionary): Key word arguments passed to the solver.
            
    Returns:
        eo (EstimatorsObject): Object that contains the data used for 
            the computation and the results.
            
    """
    
    eo = EstimatorsObject(data, data_sets, observables, model, obs_data=obs_data, stationary_distributions=stationary_distributions, K_shift=K_shift, K_shift_step=K_shift_step, Max_Count=Max_Count)

    if derivative is None:
        if solver in ["cg", "newton", "bfgs", "one"]:
            derivative = True
        else:
            derivative = False
    Qfunction_epsilon = eo.get_function(derivative, logq)
    
    if x0 is None:
        current_epsilons = eo.current_epsilons
    else:
        current_epsilons = x0
    
    print "Starting Optimization"
    t1 = time.time()
    #Then run the solver
    
    ##add keyword args thatn need to be passed
    #kwargs["logq"] = logq
    
    if isinstance(solver, str):
        if solver not in function_dictionary:
            raise IOError("Invalid Solver. Please specify a valid solver")
        func_solver = function_dictionary[solver]
    else:
        func_solver = solver #assume a valid method was passed to it
    
    new_epsilons = func_solver(Qfunction_epsilon, current_epsilons, **kwargs)
    
    t2 = time.time()
    total_time = (t2-t1) / 60.0
    print "Optimization Complete: %f minutes" % total_time
    
    #then return a new set of epsilons inside the EstimatorsObject
    eo.save_solutions(new_epsilons)
    return eo
    
    
    
    
    
    
    
    
