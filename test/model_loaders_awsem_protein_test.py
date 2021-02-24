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


def prepare_energy_function(pmodel, data, fraction=None):
     """
     Here, prepare an energy function in a format used by
     ddG_generic class. Need to unwrap it back to 2d array
     """
     Q = []
     # Need to use a random set of parameters. For safety, make all parameters
     # equal to 1
     epsilon = pmodel.get_epsilons()
     epsilon = np.ones(len(epsilon))
     for dictionary in data:
         state = dictionary['index']
         values = np.array(dictionary['data'])
         _, depsilons = pmodel.get_potentials_epsilon(values)
         q_microstate = np.array(depsilons(epsilon)).T
         Q.append(q_microstate)
     derivative_array = np.concatenate(Q, axis=0)
     return(derivative_array)

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
        direct_interaction = ml.AWSEMDirectInteraction(len(sequence))
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
        mediated_interaction = ml.AWSEMMediatedInteraction(len(sequence))
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
        mediated_interaction = ml.AWSEMBurialInteraction(len(sequence))
        mediated_interaction.load_paramters(f'{DATA_PATH}/params_burial_only/burial_gamma.dat')
        derivs = mediated_interaction.calculate_derivatives(sequence, my_protein.local_density)
        params = mediated_interaction.get_parameters()
        H_computed = np.sum(np.multiply(derivs, params), axis = 1)
        print(H_ref)
        print(H_computed)
        assert np.isclose(H_ref, H_computed,  atol=1e-05).all()
        return


    def test_sbm_nonbonded_contacts(self):
        sbm_data_path = 'test_data/CA_sample_data'
        traj_file = f'{sbm_data_path}/CA_sample_traj.xtc'
        top_file = f'{sbm_data_path}/ref_CA.pdb'
        param_description_file = f'{sbm_data_path}/pairwise_params'
        param_value_file =  f'{sbm_data_path}/model_params'
        model_name = f'{sbm_data_path}/ubq.ini'
        dtrajs = np.loadtxt(f'{sbm_data_path}/dtrajs.txt', dtype=int)
        traj = md.load(traj_file, top=top_file)
        topology = traj.top
        # Put standart model loader here
        pmodel, data_formatted = pyODEM.model_loaders.load_protein(dtrajs,
                                                                    traj_file,
                                                                    model_name,
                                                                    observable_object=None,
                                                                    obs_data=None)
        reference_derivatives = -1.0*prepare_energy_function(pmodel, data_formatted)
        my_protein = ml.HybridProtein(topology=topology, traj_type='sbm_ca')
        my_protein.load_data(traj)
        sbm_nonbonded_interaction = ml.SBMNonbondedInteraction(topology.n_residues, param_description_file)
        derivatives = sbm_nonbonded_interaction.calculate_derivatives(my_protein.distances)
        assert np.isclose(reference_derivatives, derivatives).all()
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


    def test_get_potentials_epsilon_hybrid(self):
        """
        The test just check if everything is called without errors, and
        shapes of the outputs is as expected.
        For now, does not check if actual calculations are correct.

        """
        # Loading required data
        sbm_data_path = 'test_data/CA_sample_data'
        traj_file = f'{sbm_data_path}/CA_sample_traj.xtc'
        top_file = f'{sbm_data_path}/ref_CA.pdb'
        traj = md.load(traj_file, top=top_file)
        param_description_file = f'{sbm_data_path}/pairwise_params'
        param_value_file =  f'{sbm_data_path}/model_params'
        model_name = f'{sbm_data_path}/ubq.ini'
        my_protein = ml.HybridProtein(topology=traj.topology, traj_type='sbm_ca', parameter_location=f'{sbm_data_path}/.')
        my_protein.load_data(traj)
        my_protein.setup_Hamiltonian(terms=['sbm_nonbonded', 'burial'])
        ubq_seq = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'
        H_func = my_protein.get_H_func(sequence=ubq_seq, fraction=None)
        params = my_protein.get_epsilons()
        assert len(params) == 65 # Correct number of parameters, 5 sbm + 60 from burial terms
        H, dH = H_func(params, return_derivatives=True)
        assert len(H) == traj.n_frames
        assert dH.shape == (traj.n_frames, 65)
        return

test = TestAWSEMProtein()
test.test_get_potentials_epsilon_hybrid()