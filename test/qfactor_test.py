""" Test the Quality factor Computation """
import pytest
import pyODEM
import os
import numpy as np

@pytest.fixture
def get_observables_histogram():
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

    def test_obs_seen(self, get_observables_histogram):
        """ Q-factor is accurate to within 7 orders of magnitude """
        obs = get_observables_histogram

        # put all values in 0
        data = [np.array([0, 0, 0, 0, 0, 0, 0, 0])]

        observations, obs_std = obs.compute_observations(data)
        qfunc, derivq = obs.get_q_functions()

        # confirm Q-factor counts only the first bin
        diff = np.abs(np.exp(-((1-0.2)**2)/(2 * 0.01)) == qfunc(observations))
        diff /= qfunc(observations)
        assert diff < 0.0000001 # ~ single point error


        data = [np.array([0.5, 2.5])]

        observations, obs_std = obs.compute_observations(data)
        qfunc, derivq = obs.get_q_functions()

        manual_calculation = np.exp(-((0.5-0.2)**2)/(2 * 0.01))
        manual_calculation *= np.exp(-((0.5-0.3)**2)/(2 * (0.15**2)))
        #confirm q-factor can skip two bins
        diff = np.abs(manual_calculation - qfunc(observations))
        diff /= qfunc(observations)
        assert diff < 0.0000001 # ~ single point error

    def test_obs_seen()
