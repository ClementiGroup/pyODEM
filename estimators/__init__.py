""" Methods for solving new model parameters

This module should not contain any functions that a user would need 
to customize for their own system. Instead, those customizations can be 
handled using the pyfex/model_loaders modules and the pyfexd/observables 
modules.

Instead, this module would hold all the various estimators the user can
use to estimate a new set of model parameters. It should be formatted
so only the user-level methods are imported for their convenience.

Methods:
    max_likelihood_estimate: uses the calc_max_likelihood.py module 
    bayesian_estimate: In development

Example:
    solution = estimators.max_likelihood_estimate(data, data_sets, 
                                                    observables, model)
    solution.new_epsilons
        attribute containing optimal epsilons found  

    
"""
from calc_max_likelihood import max_likelihood_estimate 


