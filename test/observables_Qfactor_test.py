""" Test the Quality factor Computation """
import pytest
import pyODEM
import os
import numpy as np

@pytest.fixture
def get_observables_histogram():
    """ Make the ExperimentalObservables object with a histogram observable

    The histogram values and standard deviations are:
    value | standard deviation
    --------------------------
    0.2   | 0.1
    0.4   | 0.2
    0.3   | 0.15
    0.1   | 0.05

    for edges = [0, 1, 2, 3, 4, 5]
    """
    
    cwd = os.getcwd()
    obs = pyODEM.observables.ExperimentalObservables()
    obs.add_histogram("%s/test_data/qfactor_load/exp_data.dat" % cwd, edges=np.loadtxt("%s/test_data/qfactor_load/edges.dat" % cwd))
    obs.prep()

    return obs

class TestQfactor(object):

    def test_histogramo(self, get_observables_histogram):
        """ Make sure it loaded """
        obs = get_observables_histogram
        assert True

    def test_q_factor(self, get_observables_histogram):
        """ Q-factor is accurate to within 7 orders of magnitude """
        obs = get_observables_histogram

        # put all values in 0
        data = [np.array([0, 0, 0, 0, 0, 0, 0, 0])]

        observations, obs_std = obs.compute_observations(data)
        qfunc, derivq = obs.get_q_functions()

        # confirm Q-factor counts only the first bin
        diff = np.abs(np.exp(-((1-0.2)**2)/(2 * 0.01)) - qfunc(observations))
        diff /= qfunc(observations)
        assert diff < 0.0000001 # ~ single point error

        obs.prep()
        data = [np.array([0.5, 2.5])]

        observations, obs_std = obs.compute_observations(data)
        qfunc, derivq = obs.get_q_functions()

        manual_calculation = np.exp(-((0.5-0.2)**2)/(2 * 0.01))
        manual_calculation *= np.exp(-((0.5-0.3)**2)/(2 * (0.15**2)))
        #confirm q-factor can skip two bins
        diff = np.abs(manual_calculation - qfunc(observations))
        diff /= qfunc(observations)
        assert diff < 0.0000001 # ~ single point error

    def test_q_factor_logarithm(self, get_observables_histogram):
        """ Q-factor is accurate to within 7 orders of magnitude """
        obs = get_observables_histogram

        # put all values in 0
        data = [np.array([0, 0, 0, 0, 0, 0, 0, 0])]

        observations, obs_std = obs.compute_observations(data)
        qfunc, derivq = obs.get_log_q_functions()

        # confirm Q-factor counts only the first bin
        diff = np.abs((((1-0.2)**2)/(2 * 0.01)) - qfunc(observations))
        diff /= qfunc(observations)
        assert diff < 0.0000001 # ~ single point error

        obs.prep()
        data = [np.array([0.5, 2.5])]

        observations, obs_std = obs.compute_observations(data)
        qfunc, derivq = obs.get_log_q_functions()

        manual_calculation = (((0.5-0.2)**2)/(2 * 0.01))
        manual_calculation +=(((0.5-0.3)**2)/(2 * (0.15**2)))
        #confirm q-factor can skip two bins
        diff = np.abs(manual_calculation - qfunc(observations))
        diff /= qfunc(observations)
        assert diff < 0.0000001 # ~ single point error
