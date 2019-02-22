""" Test the Protein class that loads using model_builder"""
import pytest
import pyODEM
import os
import numpy as np

@pytest.fixture
def make_objects():
    cwd = os.getcwd()
    obs = pyODEM.observables.ExperimentalObservables()

    os.chdir("test_data/protein_load")
    pmodel = pyODEM.model_loaders.Protein("ww_domain.ini")
    os.chdir(cwd)


    obs.prep()
    pmodel.set_temperature(120.)

    return obs, pmodel

@pytest.fixture
def make_pmodel_energies():
    cwd = os.getcwd()

    os.chdir("test_data/protein_load")
    pmodel = pyODEM.model_loaders.Protein("ww_domain.ini")
    os.chdir(cwd)

    data = pmodel.load_data("test_data/protein_load/traj/traj_test.xtc")
    heps, dheps = pmodel.get_potentials_epsilon(data)

    true_energies = np.loadtxt("test_data/protein_load/traj/energy_gaussian_test.dat")

    return pmodel, data, heps, dheps, true_energies

class TestProtein(object):
    def test_dumb(self):
        assert 5 == 5 # duhh

    def test_import_pmodel(self, make_objects):
        """ Check Protein class loads values correctly """
        # test that the various values are correctly loaded
        obs, pmodel = make_objects
        assert pmodel.epsilons[0] == 0.0
        assert pmodel.epsilons[496] == 1.0

        assert pmodel.use_pairs[10][0] == 0
        assert pmodel.use_pairs[10][1] == 26
        assert pmodel.use_pairs[844][0] == 42
        assert pmodel.use_pairs[844][1] == 50

    def test_pmodel_energies(self, make_pmodel_energies):
        """ Check potential energy values and compare with values from GROMACS

        Perform two checks:
        1. Confirm energies from model_builder agree with energies from GROMACS
        2. Confirm potential energy varies linearly with respect to the epsilons

        Note: Python computes internally using float double, while GROMACS is
        typically compiled to use float single. This results in a floating-point
        error that carries, meaning the values won't agree exactly.
        """
        pmodel, data, heps, dheps, true_energies = make_pmodel_energies

        # first compute the total Gaussian energies between GROMACS and pmodel
        total_energy = np.zeros(np.shape(true_energies))
        for i in pmodel.use_params:
            total_energy += pmodel.model.Hamiltonian._pairs[i].V(data[:,i])
        assert np.max(total_energy - true_energies) < 0.2

        # now confirm that hepsilon calculates the correct difference in energies
        diff = heps(pmodel.epsilons + 0.1) - heps(pmodel.epsilons - 0.1)

        total_diff = np.zeros(np.shape(true_energies))
        for i in pmodel.use_params:
            pmodel.model.Hamiltonian._pairs[i].set_epsilon(pmodel.epsilons[i] + 0.1)
            total_diff += pmodel.model.Hamiltonian._pairs[i].V(data[:,i])

        for i in pmodel.use_params:
            pmodel.model.Hamiltonian._pairs[i].set_epsilon(pmodel.epsilons[i] - 0.1)
            total_diff += pmodel.model.Hamiltonian._pairs[i].V(data[:,i])

        # confirms potential energies are linear in epsilons
        assert np.max(total_diff - diff) < 0.001

    def test_pmodel_derivatives(self, make_pmodel_energies):
        """ Test the derivatives calculations, deps.

        This test will confirm if deps correctly determines the derivative by
        comparing with the numeric derivative computed from heps

        """

        pmodel, data, heps, dheps, true_energies = make_pmodel_energies

        derivatives = np.array(dheps(pmodel.epsilons))

        for frame in range(np.shape(derivatives)[1]):
            deriv = derivatives[:, frame]
            magnitude = np.sqrt(np.sum(deriv ** 2))
            direction = deriv / magnitude
            step_size = 0.01
            diff = heps(pmodel.epsilons + (direction*step_size*0.5)) - heps(pmodel.epsilons - (direction*step_size*0.5))
            numeric_magnitude = diff[frame] / step_size

            assert np.abs(magnitude - numeric_magnitude) < 0.000001
