"""
The script contains a set of tests for ddG_estimator module.
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
    Reference manual implementation of ddG on test data.
    """
    trajectory = md.load(trajfile, top=topfile)
    # The first pair (1,28) should not contribute to delta delta G, as it does not changes upon mutation
    # So it was excluded from distances calculations
    pair_1_30_distances = md.compute_distances(trajectory, [[0, 29]])[:, 0]
    pair_1_32_distances = md.compute_distances(trajectory, [[0, 31]])[:, 0]
    pair_2_33_distances = md.compute_distances(trajectory, [[1, 32]])[:, 0]

    def compute_delta_H(r0, factor, epsilon, distances, sigma=0.05):
        energies = []
        for distance in distances:
            energy = epsilon*(1-factor)*np.exp(-1*(distance-r0)**2/(2*(sigma**2)))
            energies.append(energy)
        return energies

    # Temperature of the test trajectory is hard-coded here
    beta = 1000/(8.31446261815324*130)

    def compute_weight(energies):
        res = (np.exp(-1*beta*energies))
        return np.array(res)

    pair_1_30_energy = compute_delta_H(
        r0=0.550463, factor=0, epsilon=1, distances=pair_1_30_distances)
    pair_1_32_energy = compute_delta_H(
        r0=0.536207, factor=0.6666, epsilon=1, distances=pair_1_32_distances)
    pair_2_33_energy = compute_delta_H(
        r0=0.578442, factor=0.1, epsilon=1, distances=pair_2_33_distances)

    Hamiltonian = np.add(pair_1_30_energy, pair_1_32_energy)
    Hamiltonian = np.add(Hamiltonian, pair_2_33_energy)
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

    # Calculate derivative

    def compute_dH(r0, distances, sigma=0.05):
        """
        Function computes derivative of initial Hamiltonian with respect to initial set of parameters
        Units for all the distances are nm
        """
        dH_array = []
        for distance in distances:
            dH = -1*np.exp(-1*(distance-r0)**2/(2*(sigma**2)))
            dH_array.append(dH)
        return np.array(dH_array)

    def compute_dH_mutated(fraction, r0, distances, sigma=0.05):
        """
        Function computes derivative of initial Hamiltonian with respect to initial set of parameters
        Units for all the distances are nm
        """
        dH_mutated_array = compute_dH(r0, distances, sigma)
        dH_mutated_array = np.multiply(fraction, np.array(dH_mutated_array))
        return dH_mutated_array

    pair_1_28_distances = md.compute_distances(trajectory, [[0, 27]])[:, 0]
    pair_1_30_distances = md.compute_distances(trajectory, [[0, 29]])[:, 0]
    pair_1_32_distances = md.compute_distances(trajectory, [[0, 31]])[:, 0]
    pair_2_33_distances = md.compute_distances(trajectory, [[1, 32]])[:, 0]

    pair_1_28_delta_energy = compute_delta_H(
        r0=0.820535, factor=1, epsilon=1, distances=pair_1_28_distances)
    pair_1_30_delta_energy = compute_delta_H(
        r0=0.550463, factor=0, epsilon=1, distances=pair_1_30_distances)
    pair_1_32_delta_energy = compute_delta_H(
        r0=0.536207, factor=0.6666, epsilon=1, distances=pair_1_32_distances)
    pair_2_33_delta_energy = compute_delta_H(
        r0=0.578442, factor=0.1, epsilon=1, distances=pair_2_33_distances)

    delta_Hamiltonian = np.add(pair_1_30_energy, pair_1_32_energy)
    delta_Hamiltonian = np.add(delta_Hamiltonian, pair_2_33_energy)
    weighted_delta_Hamiltonian = compute_weight(delta_Hamiltonian)

    dH0_d_pair_1_28 = compute_dH(r0=0.820535, distances=pair_1_28_distances)
    dH0_d_pair_1_30 = compute_dH(r0=0.550463, distances=pair_1_30_distances)
    dH0_d_pair_1_32 = compute_dH(r0=0.536207, distances=pair_1_32_distances)
    dH0_d_pair_2_33 = compute_dH(r0=0.578442, distances=pair_2_33_distances)

    dH_mutated_d_pair_1_28 = compute_dH_mutated(
        r0=0.820535, fraction=1, distances=pair_1_28_distances)
    dH_mutated_d_pair_1_30 = compute_dH_mutated(
        r0=0.550463, fraction=0, distances=pair_1_30_distances)
    dH_mutated_d_pair_1_32 = compute_dH_mutated(
        r0=0.536207, fraction=0.6666, distances=pair_1_32_distances)
    dH_mutated_d_pair_2_33 = compute_dH_mutated(
        r0=0.578442, fraction=0.1, distances=pair_2_33_distances)

    product_exp_dH_mutated_pair_1_28 = np.multiply(
        weighted_delta_Hamiltonian, dH_mutated_d_pair_1_28)
    product_exp_dH_mutated_pair_1_30 = np.multiply(
        weighted_delta_Hamiltonian, dH_mutated_d_pair_1_30)
    product_exp_dH_mutated_pair_1_32 = np.multiply(
        weighted_delta_Hamiltonian, dH_mutated_d_pair_1_32)
    product_exp_dH_mutated_pair_2_33 = np.multiply(
        weighted_delta_Hamiltonian, dH_mutated_d_pair_2_33)

    # compute derivative with respect to epsilon 1_30

    # derivative of initial Hamiltonian average

    state0_dH0_average = np.mean(dH0_d_pair_1_30[0:4])
    state1_dH0_average = np.mean(dH0_d_pair_1_30[4:8])
    state2_dH0_average = np.mean(dH0_d_pair_1_30[8:12])
    state3_dH0_average = np.mean(dH0_d_pair_1_30[12:16])
    state4_dH0_average = np.mean(dH0_d_pair_1_30[16:20])

    folded_state_dH0_average = (state0_dH0_average*stationary_distribution[0] + state1_dH0_average*stationary_distribution[1])/(
        stationary_distribution[0]+stationary_distribution[1])
    unfolded_state_dH0_average = (state3_dH0_average*stationary_distribution[3] + state4_dH0_average*stationary_distribution[4])/(
        stationary_distribution[3]+stationary_distribution[4])

    # derivative of mutated hamiltonian times exponent average

    state0_product_average = np.mean(product_exp_dH_mutated_pair_1_30[0:4])
    state1_product_average = np.mean(product_exp_dH_mutated_pair_1_30[4:8])
    state2_product_average = np.mean(product_exp_dH_mutated_pair_1_30[8:12])
    state3_product_average = np.mean(product_exp_dH_mutated_pair_1_30[12:16])
    state4_product_average = np.mean(product_exp_dH_mutated_pair_1_30[16:20])

    folded_state_product_average = (state0_product_average*stationary_distribution[0] + state1_product_average*stationary_distribution[1])/(
        stationary_distribution[0]+stationary_distribution[1])
    unfolded_state_product_average = (state3_product_average*stationary_distribution[3] + state4_product_average*stationary_distribution[4])/(
        stationary_distribution[3]+stationary_distribution[4])

    d_folded_d_pair_1_30 = folded_state_product_average/folded_state_average - folded_state_dH0_average
    d_unfolded_d_pair_1_30 = unfolded_state_product_average / \
        unfolded_state_average - unfolded_state_dH0_average
    delta_delta_G_derivative_pair_1_30 = d_folded_d_pair_1_30 - d_unfolded_d_pair_1_30
    beta_delta_delta_G_derivative_pair_1_30 = delta_delta_G_derivative_pair_1_30*beta

    # compute derivative with respect to epsilon 1_32

    # exponent average
    state0_average = np.mean(weighted_delta_Hamiltonian[0:4])
    state1_average = np.mean(weighted_delta_Hamiltonian[4:8])
    state2_average = np.mean(weighted_delta_Hamiltonian[8:12])
    state3_average = np.mean(weighted_delta_Hamiltonian[12:16])
    state4_average = np.mean(weighted_delta_Hamiltonian[16:20])
    folded_state_average = (state0_average*stationary_distribution[0] + state1_average*stationary_distribution[1])/(
        stationary_distribution[0]+stationary_distribution[1])
    unfolded_state_average = (state3_average*stationary_distribution[3] + state4_average*stationary_distribution[4])/(
        stationary_distribution[3]+stationary_distribution[4])

    state0_dH0_average = np.mean(dH0_d_pair_1_32[0:4])
    state1_dH0_average = np.mean(dH0_d_pair_1_32[4:8])
    state2_dH0_average = np.mean(dH0_d_pair_1_32[8:12])
    state3_dH0_average = np.mean(dH0_d_pair_1_32[12:16])
    state4_dH0_average = np.mean(dH0_d_pair_1_32[16:20])

    folded_state_dH0_average = (state0_dH0_average*stationary_distribution[0] + state1_dH0_average*stationary_distribution[1])/(
        stationary_distribution[0]+stationary_distribution[1])
    unfolded_state_dH0_average = (state3_dH0_average*stationary_distribution[3] + state4_dH0_average*stationary_distribution[4])/(
        stationary_distribution[3]+stationary_distribution[4])

    # derivative of mutated hamiltonian times exponent average

    state0_product_average = np.mean(product_exp_dH_mutated_pair_1_32[0:4])
    state1_product_average = np.mean(product_exp_dH_mutated_pair_1_32[4:8])
    state2_product_average = np.mean(product_exp_dH_mutated_pair_1_32[8:12])
    state3_product_average = np.mean(product_exp_dH_mutated_pair_1_32[12:16])
    state4_product_average = np.mean(product_exp_dH_mutated_pair_1_32[16:20])

    folded_state_product_average = (state0_product_average*stationary_distribution[0] + state1_product_average*stationary_distribution[1])/(
        stationary_distribution[0]+stationary_distribution[1])
    unfolded_state_product_average = (state3_product_average*stationary_distribution[3] + state4_product_average*stationary_distribution[4])/(
        stationary_distribution[3]+stationary_distribution[4])

    d_folded_d_pair_1_32 = folded_state_product_average/folded_state_average - folded_state_dH0_average
    d_unfolded_d_pair_1_32 = unfolded_state_product_average / \
        unfolded_state_average - unfolded_state_dH0_average
    delta_delta_G_derivative_pair_1_32 = d_folded_d_pair_1_32 - d_unfolded_d_pair_1_32
    beta_delta_delta_G_derivative_pair_1_32 = beta*delta_delta_G_derivative_pair_1_32

    # compute derivative with respect to epsilon 1_28

    # derivative of initial Hamiltonian average

    state0_dH0_average = np.mean(dH0_d_pair_1_28[0:4])
    state1_dH0_average = np.mean(dH0_d_pair_1_28[4:8])
    state2_dH0_average = np.mean(dH0_d_pair_1_28[8:12])
    state3_dH0_average = np.mean(dH0_d_pair_1_28[12:16])
    state4_dH0_average = np.mean(dH0_d_pair_1_28[16:20])

    folded_state_dH0_average = (state0_dH0_average*stationary_distribution[0] + state1_dH0_average*stationary_distribution[1])/(
        stationary_distribution[0]+stationary_distribution[1])
    unfolded_state_dH0_average = (state3_dH0_average*stationary_distribution[3] + state4_dH0_average*stationary_distribution[4])/(
        stationary_distribution[3]+stationary_distribution[4])

    # derivative of mutated hamiltonian times exponent average

    state0_product_average = np.mean(product_exp_dH_mutated_pair_1_28[0:4])
    state1_product_average = np.mean(product_exp_dH_mutated_pair_1_28[4:8])
    state2_product_average = np.mean(product_exp_dH_mutated_pair_1_28[8:12])
    state3_product_average = np.mean(product_exp_dH_mutated_pair_1_28[12:16])
    state4_product_average = np.mean(product_exp_dH_mutated_pair_1_28[16:20])

    folded_state_product_average = (state0_product_average*stationary_distribution[0] + state1_product_average*stationary_distribution[1])/(
        stationary_distribution[0]+stationary_distribution[1])
    unfolded_state_product_average = (state3_product_average*stationary_distribution[3] + state4_product_average*stationary_distribution[4])/(
        stationary_distribution[3]+stationary_distribution[4])

    d_folded_d_pair_1_28 = folded_state_product_average/folded_state_average - folded_state_dH0_average
    d_unfolded_d_pair_1_28 = unfolded_state_product_average / \
        unfolded_state_average - unfolded_state_dH0_average
    delta_delta_G_derivative_pair_1_28 = d_folded_d_pair_1_28 - d_unfolded_d_pair_1_28
    beta_delta_delta_G_derivative_pair_1_28 = beta*delta_delta_G_derivative_pair_1_28

    # compute derivative with respect to epsilon 2_33

    # derivative of initial Hamiltonian average

    state0_dH0_average = np.mean(dH0_d_pair_2_33[0:4])
    state1_dH0_average = np.mean(dH0_d_pair_2_33[4:8])
    state2_dH0_average = np.mean(dH0_d_pair_2_33[8:12])
    state3_dH0_average = np.mean(dH0_d_pair_2_33[12:16])
    state4_dH0_average = np.mean(dH0_d_pair_2_33[16:20])

    folded_state_dH0_average = (state0_dH0_average*stationary_distribution[0] + state1_dH0_average*stationary_distribution[1])/(
        stationary_distribution[0]+stationary_distribution[1])
    unfolded_state_dH0_average = (state3_dH0_average*stationary_distribution[3] + state4_dH0_average*stationary_distribution[4])/(
        stationary_distribution[3]+stationary_distribution[4])

    # derivative of mutated hamiltonian times exponent average

    state0_product_average = np.mean(product_exp_dH_mutated_pair_2_33[0:4])
    state1_product_average = np.mean(product_exp_dH_mutated_pair_2_33[4:8])
    state2_product_average = np.mean(product_exp_dH_mutated_pair_2_33[8:12])
    state3_product_average = np.mean(product_exp_dH_mutated_pair_2_33[12:16])
    state4_product_average = np.mean(product_exp_dH_mutated_pair_2_33[16:20])

    folded_state_product_average = (state0_product_average*stationary_distribution[0] + state1_product_average*stationary_distribution[1])/(
        stationary_distribution[0]+stationary_distribution[1])
    unfolded_state_product_average = (state3_product_average*stationary_distribution[3] + state4_product_average*stationary_distribution[4])/(
        stationary_distribution[3]+stationary_distribution[4])

    d_folded_d_pair_2_33 = folded_state_product_average/folded_state_average - folded_state_dH0_average
    d_unfolded_d_pair_2_33 = unfolded_state_product_average / \
        unfolded_state_average - unfolded_state_dH0_average
    delta_delta_G_derivative_pair_2_33 = d_folded_d_pair_2_33 - d_unfolded_d_pair_2_33
    beta_delta_delta_G_derivative_pair_2_33 = beta*delta_delta_G_derivative_pair_2_33

    return (beta_ddG, [beta_delta_delta_G_derivative_pair_1_28,
                       beta_delta_delta_G_derivative_pair_1_30,
                       beta_delta_delta_G_derivative_pair_1_32,
                       beta_delta_delta_G_derivative_pair_2_33])

class TestDDGEstimator(object):
    def test_compute_delta_delta_G(self):
        """
        Compute delta_delta_G and corresponding derivatives
        """

        test_files_folder = 'test_data/1UBQ_sample_data'
        trajfile = '{}/sample_traj.xtc'.format(test_files_folder)
        topfile = '{}/ref.pdb'.format(test_files_folder)
        fraction = np.loadtxt('{}/fraction.txt'.format(test_files_folder))[:, 2]
        stationary_distribution = np.loadtxt('{}/stationary_distribution.txt'.format(test_files_folder))
        macrostates = np.loadtxt('{}/macrostates.txt'.format(test_files_folder))
        dtrajs = np.loadtxt('{}/dtrajs.txt'.format(test_files_folder), dtype=int)
        model_name = '{}/ubq.ini'.format(test_files_folder)
        reference = _reference_ddG_implementation(trajfile, topfile, stationary_distribution)
        print(reference)

        pmodel, data_formatted = pyODEM.model_loaders.load_protein(dtrajs,
                                                                   trajfile,
                                                                   model_name,
                                                                   observable_object=None,
                                                                   obs_data=None)
        pmodel.set_temperature(130)  # Hardcoded temperature of the test set
        obs = ddG_est_new.ddG(pmodel, data_formatted, macrostates, fraction,
                              dtrajs, stationary_distribution, debug=False)
        obs.prepare_observables(optimize=True, epsilon=pmodel.get_epsilons())
        results = obs.compute_delta_delta_G(pmodel.get_epsilons(), compute_derivative=True)
        print(results)

        assert np.abs(reference[0]-results[0]) < 1.0e-7, "beta_DDG deviates from the reference!"
        for ndx in range(len(reference[1])):
            assert np.abs(reference[1][ndx]-results[1][ndx]
                          ) < 1.0e-7, "Derivative number {} deviates from the reference!".format(ndx)


    def test_compute_delta_delta_G_linear(self):
        """
        Compute delta_delta_G and corresponding derivatives
        """

        test_files_folder = 'test_data/1UBQ_sample_data'
        trajfile = '{}/sample_traj.xtc'.format(test_files_folder)
        topfile = '{}/ref.pdb'.format(test_files_folder)
        fraction = np.loadtxt('{}/fraction.txt'.format(test_files_folder))[:, 2]
        stationary_distribution = np.loadtxt('{}/stationary_distribution.txt'.format(test_files_folder))
        macrostates = np.loadtxt('{}/macrostates.txt'.format(test_files_folder))
        dtrajs = np.loadtxt('{}/dtrajs.txt'.format(test_files_folder), dtype=int)
        model_name = '{}/ubq.ini'.format(test_files_folder)
        reference = _reference_ddG_implementation(trajfile, topfile, stationary_distribution)
        print(reference)

        pmodel, data_formatted = pyODEM.model_loaders.load_protein(dtrajs,
                                                                   trajfile,
                                                                   model_name,
                                                                   observable_object=None,
                                                                   obs_data=None)
        pmodel.set_temperature(130)  # Hardcoded temperature of the test set
        Q_energy_terms = ddG_est_linear.compute_Q(pmodel, data_formatted)
        obs = ddG_est_linear.ddG_linear(pmodel, Q_energy_terms, macrostates,
                                        fraction, dtrajs, stationary_distribution, debug=False)
        obs.prepare_observables(optimize=True, epsilon=pmodel.get_epsilons())
        results = obs.compute_delta_delta_G(pmodel.get_epsilons(), compute_derivative=True)
        print(results)

        assert np.abs(reference[0]-results[0]) < 1.0e-7, "beta_DDG deviates from the reference!"
        for ndx in range(len(reference[1])):
            assert np.abs(reference[1][ndx]-results[1][ndx]
                          ) < 1.0e-7, "Derivative number {} deviates from the reference!".format(ndx)

    def test_Mutational_model_linear(self):
        """
        Compute delta_delta_G and corresponding derivatives
        """

        test_files_folder = 'test_data/1UBQ_sample_data'
        trajfile = '{}/sample_traj.xtc'.format(test_files_folder)
        topfile = '{}/ref.pdb'.format(test_files_folder)
        fraction = np.loadtxt('{}/fraction.txt'.format(test_files_folder))[:, 2]
        stationary_distribution = np.loadtxt('{}/stationary_distribution.txt'.format(test_files_folder))
        macrostates = np.loadtxt('{}/macrostates.txt'.format(test_files_folder))
        dtrajs = np.loadtxt('{}/dtrajs.txt'.format(test_files_folder), dtype=int)
        model_name = '{}/ubq.ini'.format(test_files_folder)
        reference = _reference_ddG_implementation(trajfile, topfile, stationary_distribution)
        print(reference)

        pmodel, data_formatted = pyODEM.model_loaders.load_protein(dtrajs,
                                                                   trajfile,
                                                                   model_name,
                                                                   observable_object=None,
                                                                   obs_data=None)
        pmodel.set_temperature(130)  # Hardcoded temperature of the test set
        obs = mutational_model.Mutational_Model_Linear(protein_model=pmodel,
                                           partition=macrostates,
                                           dtrajs=dtrajs,
                                           distribution=stationary_distribution,
                                           data=data_formatted)
        obs.prepare_observables(optimize=True, epsilon=pmodel.get_epsilons())
        obs.add_mutant("Test",
           fraction,
           compute_ddG_U2F=True,
           compute_ddG_U2T=True)
        results = obs.compute_delta_delta_G(epsilons=pmodel.get_epsilons(),
                               compute_derivative=True)
        print(results)

        assert np.abs(reference[0]-results[0]) < 1.0e-7, "beta_DDG deviates from the reference!"
        assert len(reference[1]) == results[1][0].shape[0], "Reference and resulting derivative have different number of values"
        for ndx in range(len(reference[1])):
            assert np.abs(reference[1][ndx]-results[1][0][ndx]
                          ) < 1.0e-7, "Derivative number {} deviates from the reference!".format(ndx)



    def test_ddG_generic(self):
        test_files_folder = 'test_data/1UBQ_sample_data'
        trajfile = '{}/sample_traj.xtc'.format(test_files_folder)
        topfile = '{}/ref.pdb'.format(test_files_folder)
        fraction = np.loadtxt('{}/fraction.txt'.format(test_files_folder))[:, 2]
        stationary_distribution = np.loadtxt('{}/stationary_distribution.txt'.format(test_files_folder))
        macrostates = np.loadtxt('{}/macrostates.txt'.format(test_files_folder))
        dtrajs = np.loadtxt('{}/dtrajs.txt'.format(test_files_folder), dtype=int)
        model_name = '{}/ubq.ini'.format(test_files_folder)
        reference = _reference_ddG_implementation(trajfile, topfile, stationary_distribution)
        pmodel, data_formatted = pyODEM.model_loaders.load_protein(dtrajs,
                                                                   trajfile,
                                                                   model_name,
                                                                   observable_object=None,
                                                                   obs_data=None)

        data = data_formatted
        pmodel.set_temperature(130)  # Hardcoded temperature of the test set

        def prepare_energy_function(pmodel, data, fraction):
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

            def H_wt(epsilon, return_derivatives=False):
                energy = np.sum(derivative_array, axis=1)
                if return_derivatives:
                    return energy, derivative_array
                else:
                    return energy

            def H_mutated(epsilon, return_derivatives=False):
                corrected_epsilon = np.multiply(fraction, epsilon)
                mutant_derivative = np.multiply(derivative_array, fraction)
                energy =  np.sum(np.multiply(derivative_array,corrected_epsilon), axis=1)
                if return_derivatives:
                    return energy, mutant_derivative
                else:
                    return energy
            return H_wt, H_mutated

        H_wt, H_mutated = prepare_energy_function(pmodel, data, fraction)
        epsilons = np.ones((4))
        ddG = ddG_est_linear.ddG_generic(H_wt,
                                         H_mutated,
                                         stationary_distribution,
                                         rescale_temperature=False,
                                         dtrajs=dtrajs,
                                         partition=macrostates)
        ddG.prepare_observables(optimize=True, epsilon=epsilons)
        results = ddG.compute_observation(epsilons)
        print(results)
        print(reference)
        assert np.abs(reference[0]-results[0]) < 1.0e-7, "beta_DDG deviates from the reference!"
        assert len(reference[1]) == results[1].shape[0], "Reference and resulting derivative have different number of values"
        for ndx in range(len(reference[1])):
            assert np.abs(reference[1][ndx]-results[1][ndx]
                         ) < 1.0e-7, "Derivative number {} deviates from the reference!".format(ndx)


    def  test_split_data_by_microstates(self):
        dtrajs = np.array([1, 3, 3, 2, 2, 0, 2, 0], dtype=int)
        data_1d = np.array([1, 3, 3, 2, 2, 0, 2, 0], dtype=int)
        data_2d = np.array([ [1, 1, 1],
                            [3, 3, 3],
                            [3, 3, 3],
                            [2, 2, 2],
                            [2, 2, 2],
                            [0, 0, 0],
                            [2, 2, 2],
                            [0, 0, 0]
                              ], dtype=int)
        print(data_2d.shape)

        reference_1d = {0: np.array([0, 0]),
                        1: np.array([1]),
                        2: np.array([2,2,2]),
                        3: np.array([3,3])}
        reference_2d = {0: np.array([[0,0,0],[0,0,0]]),
                        1: np.array([[1,1,1]]),
                        2: np.array([[2,2,2],[2,2,2],[2,2,2]]),
                        3: np.array([[3,3,3],[3,3,3]]),
                        }
        split_1d = ddG_est_linear.ddG_generic._split_data_by_microstates(dtrajs, data_1d)
        split_2d = ddG_est_linear.ddG_generic._split_data_by_microstates(dtrajs, data_2d)
        # Check that number of elements in both dictionaries is correct.
        assert len(reference_1d) == len(split_1d)
        assert len(reference_2d) == len(split_2d)
        for key in split_1d:
            assert np.equal(split_1d[key], reference_1d[key]).all
        for key in split_2d:
            assert np.equal(split_2d[key], reference_2d[key]).all

test = TestDDGEstimator()

test.test_ddG_generic()
test. test_split_data_by_microstates()
