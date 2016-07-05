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
        
def max_likelihood_estimate(data, data_sets, observables, model, obs_data=None, solver="simplex", logq=False, derivative=None, x0=None, kwargs={}, stationary_distributions=None, model_state=None):
    """ Optimizes model's paramters using a max likelihood method
    
    Args:
        See pyfexd.estimators.estimators_class.EstimatorsObject for:
            data (array), data_sets (list), 
            observables (ExperimentalObservables), model (ModelLoader), 
            obs_data(list) and stationary_distributions (list)
             
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
    
    eo = EstimatorsObject(data, data_sets, observables, model, obs_data=obs_data, stationary_distributions=stationary_distributions, model_state=model_state)

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
    
    try:
        new_epsilons = func_solver(Qfunction_epsilon, current_epsilons, **kwargs)
        
        t2 = time.time()
        total_time = (t2-t1) / 60.0
        print "Optimization Complete: %f minutes" % total_time
        
        #then return a new set of epsilons inside the EstimatorsObject
        eo.save_solutions(new_epsilons)
        return eo
    except:
        t2 = time.time()
        total_time = (t2-t1) / 60.0
        print "Optimization Failed: %f minutes" % total_time
        return eo
    
    
    
    
    
    
    
    
