""" Methods for solving new model parameters

This module would hold all the various estimators the user can use to estimate a
new set of model parameters. It should be formatted so only the user-level
methods are imported for their convenience.

Methods:
    max_likelihood_estimate(): uses the calc_max_likelihood.py module.
    kfold_crossvalidation_max_likelihood(): Perform an optimization after using
        checking the hyper parameters with a k-fold cross validation.
    bayesian_estimate: In development

Example:
    solution = estimators.max_likelihood_estimate(data, data_sets,
                                                    observables, model)
    solution.new_epsilons
        attribute containing optimal epsilons found


"""
from calc_max_likelihood import max_likelihood_estimate_serial
from calc_max_likelihood import max_likelihood_estimate
from calc_max_likelihood import kfold_crossvalidation_max_likelihood
from estimators_class import EstimatorsObject
