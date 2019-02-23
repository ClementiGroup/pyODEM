""" Test the Protein class that loads using model_builder"""
import pytest
import pyODEM
import os
import numpy as np

@pytest.fixture
def make_lmodel_objects():
    return _make_lmodel_objects()

def _make_lmodel_objects():
    """ Make Langevin 1-D object

    Two wells with sigma 0.1 are placed at r0=0.5 and r0=2.5
    """
    cwd = os.getcwd()

    lmodel = pyODEM.model_loaders.LangevinCustom()

    lmodel.set_beta(1.0)

    parameters = {"epsilons":1.0, "r0": 0.5, "sigma":0.1}
    lmodel.add_gaussian(parameters)
    parameters = {"epsilons":1.0, "r0": 2.5, "sigma":0.1}
    lmodel.add_gaussian(parameters)

    return lmodel

class TestLangevin(object):
    def test_import_langevin(self, make_lmodel_objects):
        """ Check Protein class loads values correctly """
        # test that the various values are correctly loaded
        lmodel = make_lmodel_objects
        assert lmodel.parameters[0][0] == 0.5
        assert lmodel.parameters[1][0] == 2.5
        assert lmodel.parameters[0][1] == 0.1
        assert lmodel.parameters[1][1] == 0.1
        assert lmodel.beta == 1.0


    def test_langevin_energies(self, make_lmodel_objects):
        """ Check potential energy values and compare with expected values

        There are two gaussian wells placed

        """

        lmodel = make_lmodel_objects

        data = np.array([0.5, 2.5])

        heps, deps = lmodel.get_potentials_epsilon(data)

        # note, values are not truly 1 or 0, 1 + 1.3*10**-87 == 1 due to discrete arithmetic
        assert heps(lmodel.epsilons)[0] == 1
        assert heps(lmodel.epsilons)[1] == 1
        assert heps(lmodel.epsilons - 0.5)[0] == 0.5
        assert heps(lmodel.epsilons - 0.5)[1] == 0.5
        assert heps(np.array([0,1]))[0] < 10**-7
        assert heps(np.array([0,1]))[1] == 1

    def test_lmodel_derivatives(self, make_lmodel_objects):
        """ Test the derivatives calculations, deps.

        This test will confirm if deps correctly determines the derivative by
        comparing with the numeric derivative computed from heps

        """

        lmodel = make_lmodel_objects
        data = np.array([0.5, 2.5])
        heps, deps = lmodel.get_potentials_epsilon(data)
        derivatives = np.array(deps(lmodel.epsilons))

        for frame in range(np.shape(derivatives)[1]):
            deriv = derivatives[:, frame]
            magnitude = np.sqrt(np.sum(deriv ** 2))
            direction = deriv / magnitude
            step_size = 0.01
            diff = heps(lmodel.epsilons + (direction*step_size*0.5)) - heps(lmodel.epsilons - (direction*step_size*0.5))
            numeric_magnitude = diff[frame] / step_size

            assert np.abs(magnitude - numeric_magnitude) < 0.000001
