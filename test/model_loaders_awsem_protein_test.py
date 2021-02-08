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

    def __init__(self):
        pass

    def test_direct_contacts(self):
        traj = md.load(f'{DATA_PATH}/movie.pdb')
        topology = traj.top
        openawsem_protein = ml.OpenAWSEMProtein()
        openawsem_protein.prepare_system(
                           f'{DATA_PATH}/1pgb_openmmawsem.pdb',
                           os.path.abspath(f'{DATA_PATH}/.'),
                           [contact_term],
                           sequence,
                           chains='A')
        H_ref = openawsem_protein.calculate_H_for_trajectory(traj)
        my_protein = ml.AWSEMProtein(topology)
        my_protein.load_data(traj)
        my_protein._compute_pairwise_distances(traj)
        direct_interaction = ml.DirectInteraction(sequence)
        direct_interaction.load_paramters(f'{DATA_PATH}/gamma.dat')
        params, types = direct_interaction.get_all_parameters()
        derivs, unique_params = direct_interaction.calculate_derivatives(my_protein.distances)
        print(unique_params)
        H_computed = np.sum(np.multiply(derivs,unique_params), axis = 1)
        assert (numpy.isclose(H_ref, H_computed, rtol=1e-4))
# test = TestAWSEMProtein()
# H  = test.test_direct_contacts()
# print(H)
