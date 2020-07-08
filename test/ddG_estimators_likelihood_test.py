"""
Tests for the likelihood function calculations
"""

import pytest
import pyODEM
import numpy as np
likelihood = pyODEM.ddG_estimators.likelihood_functions.Likelihood

class test_values_for_likelihood():
    """Class that generates simple test
    values and mimic observable class
    behavior
    """
    def __init__(self):
        self.type='test'
        return

    def compute_observation(self,epsilons):
        values = np.array([1.0, 2.0, 3.0])
        epsilons = np.array(epsilons)
        ones =  np.ones(epsilons.shape[0])
        derivatives = np.array([ones*0.5,ones*1.0,ones*1.5])
        return values, derivatives


class TestLikelihood(object):
    """ Test different likelihood functions
    """
    def test_ln_gauss(self):
        """
        Tests calculation of ln_gauss function
        """
        Q = likelihood()
        test_object = test_values_for_likelihood()
        experimental_values = [0.9,2.1,3.2]
        errors = [0.5,0.4,0.3]
        reference_q_value = 0.27347222222222267
        reference_q_derivative = np.array([-3.75833333, -3.75833333])
        Q.add_observable(test_object,experimental_values,errors,scale=1.0)
        q_value, q_derivative = Q.compute_ln_gauss(np.array([1.0,1.0]))
        assert np.isclose(q_value,reference_q_value)
        assert np.all(np.isclose(q_derivative,reference_q_derivative))
        return
