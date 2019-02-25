""" Test the Protein class that loads using model_builder"""
import pytest
import pyODEM
import os
import numpy as np
from model_loaders_LangevinCustom_test import make_lmodel_objects
from observables_Qfactor_test import get_observables_histogram

@pytest.fixture
def get_formatted_data():
    """ Basic input data for the lmodel object

    Four points are observed, divided into two states centered at 0.5 and 2.5.
    """

    data1 = {"index":0, "data":np.array([0.5, 0.5])}
    data2 = {"index":1, "data":np.array([2.5, 2.5])}

    return [data1, data2]

@pytest.fixture
def get_perfect_data():
    """ This data set should result in no changes to the epsilons """

    data1 = {"index":0, "data":np.array([0.5, 0.5])}
    data2 = {"index":1, "data":np.array([1.5, 1.5, 1.5, 1.5])}
    data3 = {"index":2, "data":np.array([2.5, 2.5, 2.5])}
    data4 = {"index":3, "data":np.array([3.5])}

    return [data1, data2, data3, data4]

class TestMaxLikelihood(object):
    def test_find_maxlikelihood_nochange(self, make_lmodel_objects, get_observables_histogram, get_perfect_data):
        """ Check an optimization of a simple data-set

        See fixture for more details. The goal of this test is to confirm the
        epsilon values do not change at all when given the exact observable.

        Args:
        -----
        make_lmodel_object (fixture): Sets up the
            pyODEM.model_loaders.LangevinCustom object.
        get_observables_histogram (fixture): Sets up the
            pyODEM.observables.ExperimentalObservables object.
        get_perfect_data (fixture): Sets up the input data for the
            lmodel object data.

        """

        lmodel = make_lmodel_objects
        obs = get_observables_histogram
        formatted_data = get_perfect_data

        for thing in formatted_data:
            obs_result, obs_std = obs.compute_observations([thing["data"]])
            thing["obs_result"] = obs_result

        eo = pyODEM.estimators.max_likelihood_estimate(formatted_data, obs, lmodel, logq=True, kwargs={"gtol":0.001})

        assert eo.new_epsilons[0] == 1
        assert eo.new_epsilons[1] == 1

        eo = pyODEM.estimators.max_likelihood_estimate(formatted_data, obs, lmodel, logq=False, kwargs={"gtol":0.001})

        assert eo.new_epsilons[0] == 1
        assert eo.new_epsilons[1] == 1


    def test_find_maxlikelihood(self, make_lmodel_objects, get_observables_histogram, get_formatted_data):
        """ Check an optimization of a simple data-set

        The goal of this test is to confirm that the method finds the correct
        maximized q-factor by adjusting the epsilons to the right ratio.

        make_lmodel_object (fixture): Sets up the
            pyODEM.model_loaders.LangevinCustom object.
        get_observables_histogram (fixture): Sets up the
            pyODEM.observables.ExperimentalObservables object.
        get_formatted_data (fixture): Sets up the input data for the
            lmodel object data.
        """

        lmodel = make_lmodel_objects
        obs = get_observables_histogram
        formatted_data = get_formatted_data

        gtol = 0.001

        for thing in formatted_data:
            obs_result, obs_std = obs.compute_observations([thing["data"]])
            thing["obs_result"] = obs_result

        eo = pyODEM.estimators.max_likelihood_estimate(formatted_data, obs, lmodel, stationary_distributions=np.array([0.5,0.5]), logq=True, kwargs={"gtol":gtol})

        assert np.abs((1-eo.new_epsilons[0]) - (1-eo.new_epsilons[1]) - 0.6021754023542187) <= gtol # accurate to the termination criterion

        eo = pyODEM.estimators.max_likelihood_estimate(formatted_data, obs, lmodel, stationary_distributions=np.array([0.5,0.5]), logq=False, kwargs={"gtol":gtol})

        assert np.abs((1-eo.new_epsilons[0]) - (1-eo.new_epsilons[1]) - 0.6021754023542187) <= gtol # accurate to the termination criterion
