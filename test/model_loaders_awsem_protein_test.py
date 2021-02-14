import pytest
import pyODEM
import numpy as np
import mdtraj as md
import os
import sys

ml = pyODEM.model_loaders

OPENAWSEM_LOCATION = os.environ["OPENAWSEM_LOCATION"]
sys.path.append(OPENAWSEM_LOCATION)

from openmmawsem  import *
from helperFunctions.myFunctions import *


DATA_PATH = 'test_data/awsem_sample_data'
sequence = 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'

class TestAWSEMProtein():
    """
    Test model loader, that loads information for AWSEM contact
    parameters  and computes corresponding contribution to the Hamiltonian
    and derivatives
    """

    def test_direct_contacts(self):
        traj = md.load(f'{DATA_PATH}/movie.pdb')
        topology = traj.top
        openawsem_protein = ml.OpenAWSEMProtein()
        openawsem_protein.prepare_system(
                           f'{DATA_PATH}/1pgb_openmmawsem.pdb',
                           os.path.abspath(f'{DATA_PATH}/params_direct_only/.'),
                           [contact_term],
                           sequence,
                           chains='A')
        H_ref = openawsem_protein.calculate_H_for_trajectory(traj)
        my_protein = ml.AWSEMProtein(topology)
        my_protein.load_data(traj)
        #my_protein._compute_pairwise_distances(traj)
        direct_interaction = ml.DirectInteraction(len(sequence))
        direct_interaction.load_paramters(f'{DATA_PATH}/params_direct_only/gamma.dat')
        derivs = direct_interaction.calculate_derivatives(sequence,
                                                         input=my_protein.distances)
        params = direct_interaction.get_parameters()
        H_computed = np.sum(np.multiply(derivs, params), axis = 1)
        assert np.isclose(H_ref, H_computed,  atol=1e-05).all()


    def test_mediated_contacts(self):
        traj = md.load(f'{DATA_PATH}/movie.pdb')
        topology = traj.top
        openawsem_protein = ml.OpenAWSEMProtein()
        openawsem_protein.prepare_system(
                           f'{DATA_PATH}/1pgb_openmmawsem.pdb',
                           os.path.abspath(f'{DATA_PATH}/params_mediated_only/.'),
                           [contact_term],
                           sequence,
                           chains='A')
        #openawsem_protein.oa.
        H_ref = openawsem_protein.calculate_H_for_trajectory(traj)
        my_protein = ml.AWSEMProtein(topology)
        my_protein.load_data(traj)
        mediated_interaction = ml.MediatedInteraction(len(sequence))
        mediated_interaction.load_paramters(f'{DATA_PATH}/params_mediated_only/gamma.dat')
        derivs = mediated_interaction.calculate_derivatives(sequence, my_protein.distances, my_protein.local_density)
        params = mediated_interaction.get_parameters()
        H_computed = np.sum(np.multiply(derivs, params), axis = 1)
        print(H_ref)
        print(H_computed)
        assert np.isclose(H_ref, H_computed,  atol=1e-05).all()
        return


    def test_burial_contacts(self):
        traj = md.load(f'{DATA_PATH}/movie.pdb')
        topology = traj.top
        openawsem_protein = ml.OpenAWSEMProtein()
        openawsem_protein.prepare_system(
                           f'{DATA_PATH}/1pgb_openmmawsem.pdb',
                           os.path.abspath(f'{DATA_PATH}/params_burial_only/.'),
                           [contact_term],
                           sequence,
                           chains='A')
        H_ref = openawsem_protein.calculate_H_for_trajectory(traj)
        my_protein = ml.AWSEMProtein(topology)
        my_protein.load_data(traj)
        mediated_interaction = ml.BurialInteraction(len(sequence))
        mediated_interaction.load_paramters(f'{DATA_PATH}/params_burial_only/burial_gamma.dat')
        derivs = mediated_interaction.calculate_derivatives(sequence, my_protein.local_density)
        params = mediated_interaction.get_parameters()
        H_computed = np.sum(np.multiply(derivs, params), axis = 1)
        print(H_ref)
        print(H_computed)
        assert np.isclose(H_ref, H_computed,  atol=1e-05).all()
        return



    def test_get_potentials_epsilon(self):
        traj = md.load(f'{DATA_PATH}/movie.pdb')
        topology = traj.top
        openawsem_protein = ml.OpenAWSEMProtein()
        openawsem_protein.prepare_system(
                           f'{DATA_PATH}/1pgb_openmmawsem.pdb',
                           os.path.abspath(f'{DATA_PATH}/params_all/.'),
                           [contact_term],
                           sequence,
                           chains='A')
        H_ref = openawsem_protein.calculate_H_for_trajectory(traj)
        my_protein = ml.AWSEMProtein(topology, parameter_location=f'{DATA_PATH}/params_all/.')
        my_protein.load_data(traj)
        my_protein.setup_Hamiltonian(terms=['direct','mediated','burial'])
        H_func, _ = my_protein.get_potentials_epsilon(sequence)
        params = my_protein.get_epsilons()
        H_calculated = -1.0*H_func(params) # Returns negative value
        assert np.isclose(H_ref, H_calculated, atol=1e-5).all()
        return

test = TestAWSEMProtein()
#test.test_direct_contacts()
test.test_burial_contacts()
#test.test_get_potentials_epsilon()
