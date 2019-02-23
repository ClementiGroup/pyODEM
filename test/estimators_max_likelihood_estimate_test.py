""" Test the Protein class that loads using model_builder"""
import pytest
import pyODEM
import os
import numpy as np
from model_loaders_LangevinCustom_test import make_lmodel_objects
from observables_Qfactor_test import get_observables_histogram

@pytest.fixture
def get_formatted_data():

    data1 = {"index":0, "data":np.array([0.5, 0.5])}
    data2 = {"index":1, "data":np.array([0.5, 0.5])}

    return [data1, data2]

class TestMaxLikelihood(object):
    def test_find_maxlikelihood(self, make_lmodel_objects, get_observables_histogram, get_formatted_data):
        """ Check a basic optimization"""

        lmodel = make_lmodel_objects
        obs = get_observables_histogram
        formatted_data = get_formatted_data


        for thing in formatted_data:
            obs_result, obs_std = obs.compute_observations([thing["data"]])
            thing["obs_result"] = obs_result

        eo = pyODEM.estimators.max_likelihood_estimate(formatted_data, obs, lmodel, stationary_distributions=np.array([0.5,0.5]), logq=True)

        assert eo.new_epsilons[0] == 0.5
