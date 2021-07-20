"""
The script contains 
"""
import pytest
import numpy as np
import mdtraj as md
import pyODEM
ddG_est_new = pyODEM.ddG_estimators.ddG_estimate
ddG_est_linear = pyODEM.ddG_estimators.ddG_estimate_linear
mutational_model = pyODEM.ddG_estimators.Mutational_model_linear
ml = pyODEM.model_loaders



def _reference_ddG_implementation(trajfile, topfile, stationary_distribution):
    """
    Reference manual implementation of ddG for mutant M1A on test data.
    Mutation is represented as change of all the parameters for all the pairs,
    were one of the residues is M1
    """
    trajectory = md.load(trajfile, top=topfile)
    
    pair_1_14_distances = md.compute_distances(trajectory, [[0, 13]])[:, 0]
    pair_1_16_distances = md.compute_distances(trajectory, [[0, 15]])[:, 0]
    pair_1_19_distances = md.compute_distances(trajectory, [[0, 18]])[:, 0]
    # Create a custom parameter dictionary:

    # Just example set of parameters, represents nothing
    param_dict = {'MT' : 1.1e0,
                  'ME' :  0.3e0,
                  'MP' : 0.2e0,
                  'AT' : 0.5, 
                  'AE' : 0.12e0,
                  'AP' : 0.53e0 
                  }
    

    # Here, only works for attractive gaussian interactions
    def compute_delta_H(r0, param_1, param_2, distances, sigma=0.05):
        energies = []
        for distance in distances:
            energy = (param_1-param_2)*np.exp(-1*(distance-r0)**2/(2*(sigma**2)))
            energies.append(energy)
        return np.array(energies)

    # Temperature of the test trajectory is hard-coded here
    beta = 1000/(8.31446261815324*130)

    
    def compute_weight(energies):
        res = (np.exp(-1*beta*energies))
        return np.array(res)

    pair_1_14_energy = compute_delta_H(r0=0.820535, param_1=param_dict['MT'], param_2=param_dict['AT'],   distances=pair_1_14_distances)
    pair_1_16_energy = compute_delta_H(
        r0=0.550463, param_1=param_dict['ME'], param_2=param_dict['AE'], distances=pair_1_16_distances)
    pair_1_19_energy = compute_delta_H(
        r0=0.536207, param_1=param_dict['MP'], param_2=param_dict['AP'], distances=pair_1_19_distances)

    Hamiltonian = np.add(pair_1_14_energy, pair_1_16_energy)
    Hamiltonian = np.add(Hamiltonian, pair_1_19_energy)
    weighted_Hamiltonian = compute_weight(Hamiltonian)
    state0_average = np.mean(weighted_Hamiltonian[0:4])
    state1_average = np.mean(weighted_Hamiltonian[4:8])
    state2_average = np.mean(weighted_Hamiltonian[8:12])
    state3_average = np.mean(weighted_Hamiltonian[12:16])
    state4_average = np.mean(weighted_Hamiltonian[16:20])

    folded_state_average = (state0_average*stationary_distribution[0] + state1_average*stationary_distribution[1])/(
        stationary_distribution[0]+stationary_distribution[1])
    unfolded_state_average = (state3_average*stationary_distribution[3] + state4_average*stationary_distribution[4])/(
        stationary_distribution[3]+stationary_distribution[4])
    beta_ddG = np.log(1.0*unfolded_state_average/folded_state_average)
    return(beta_ddG)



class TestDDGEstimator(object):

    
    def test_compute_delta_delta_G(self):
        """
        Compute delta_delta_G and corresponding derivatives
        """
        

        test_files_folder = 'test_data/CA_sample_data'
        trajfile = '{}/CA_sample_traj.xtc'.format(test_files_folder)
        topfile = '{}/ref_CA.pdb'.format(test_files_folder)
        fasta_file = '{}/sbm_res_specific/ref.fasta'.format(test_files_folder)
        stationary_distribution = np.loadtxt('{}/stationary_distribution.txt'.format(test_files_folder))
        macrostates = np.loadtxt('{}/macrostates.txt'.format(test_files_folder))
        dtrajs = np.loadtxt('{}/dtrajs.txt'.format(test_files_folder), dtype=int)
        reference = _reference_ddG_implementation(trajfile, topfile, stationary_distribution)
        print(reference)

        temperature = 130


        ddG = []

        def read_fasta(fasta_file):
            with open(fasta_file, "r") as f:
                sequence_data =  f.readlines()[1]
            sequence = sequence_data.rstrip("\n")
            return sequence


        def mutate_sequence(wt_sequence, mutation):
            res_wt = mutation[0]
            res_mutant = mutation[-1]
            res_id_0_based =  int(mutation[1:-1]) -1
            assert wt_sequence[res_id_0_based] == res_wt, "Impossible mutant for a current sequence"
            return  f'{wt_sequence[:res_id_0_based]}{res_mutant}{wt_sequence[res_id_0_based+1:]}'

        topology = md.load(topfile).topology
        wt_sequence = read_fasta(fasta_file)
        my_protein = ml.HybridProtein(topology, traj_type='sbm_ca', parameter_location=f'{test_files_folder}/sbm_res_specific/.')
        my_protein.load_data(md.load(trajfile, top=topfile))
        my_protein.setup_Hamiltonian(terms=['sbm_nonbonded_residue_specific'])
        my_protein.set_temperature(temperature)
        H_func_wt  = my_protein.get_H_func(sequence=wt_sequence, fraction=None)
        params_initial = my_protein.get_epsilons()
        for mutation in ['M1A']:
            mutated_sequence = mutate_sequence(wt_sequence, mutation)
            H_func_mutant = my_protein.get_H_func(sequence=mutated_sequence, fraction=None)
            obs  = ddG_est_linear.ddG_generic(H_func_wt, 
                                              H_func_mutant,
                                              stationary_distribution,
                                              rescale_temperature=False,
                                              dtrajs=dtrajs,
                                              partition=macrostates)
            obs.prepare_observables(optimize=True,epsilon=params_initial)
            ddG.append(obs.compute_delta_delta_G(params_initial, compute_derivative=False))
        ddG = np.array(ddG)
        assert np.isclose(ddG, reference)
        print(ddG)

test = TestDDGEstimator()
test.test_compute_delta_delta_G()
