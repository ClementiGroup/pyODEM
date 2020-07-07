"""
The package includes classes and functions, needed to use ddG inside
the pyODEM framework.
Initial plan is to create a separate estimator for ddG. It seems to be an
easier approach to development. However, in this case ddG cannot be used in
conjunction with other observables.
Eventually,  different estimators should be combined together.

Goal of the package:
--------------------
The goal of the pachage is to provide access to functions, that can compute
Q, logQ, dQ, dlogQ values based on epsilon values.

Data that are or may be available to package
the experimental ddG values,
 equilibrium
distribution for microstate,
equilibrium distribution for macrostates,
trajectories in macrostate and microstate,
 hamiltonian functions and
derivatives of  hamiltonian functions with respect to epsilons

Functions that should be included in the package:
-------------------------------------------------

 compute_H:
 --------------------
 input: trajectory, model, model_parameters and pairwise_parameters.
 Is probably available in estimators  module.

 compute_mutated_H:
 ----------------------------
 input: trajectory, model, model_parameters, pairwise_parameters,
 The function computes mutated hamiltonian values for each frame of the supplied
 trajectory, heavily used compute_hamiltonian function. Is an auxilary function.
 May be removed in future versions of pachage

  compute_delta_H:
  ----------------
  function that comutes changes in hamiltonian upon mutation

  comute_dH:
  ---------------
  Function that computes derivatives of unmutated hamiltonian.
  Returns a numpy array of length n, where n - number of parameters.
  Each entry represents a partial derivative with respect to corresponding
  parameter.

  compute_mutated_dH
  ------------------
  Basicly the same as comute_dH, but for mutated humiltonian

  comute_d_delta_H
  ----------------
  Function that computes derivative in delta_H. Behaves similarly to
  comute_dH and compute_mutated_dH

  compute_delta_G
  ---------------
  Function that computes delta_G for a given macrostate. Will take all the
  data formated for a particular macrostate. This function does not perform
  any data formating or extraction. That should be done in separate helper
  function.

  compute_d_delta_G
  ------------------
  Function computes derivative of the delta_G of mutation for a specific
  microstate  with respect to model  parameters

  compute_delta_delta_G
  ----------------------
  Function computes delta_delta_G upon mutation. can be done easily based on
  compute_delta_G as a difference of functions for two different ensembles.

  compute_d_delta_delta_G
  ------------------------

  Function computes derivative of delta_delta_G with respect to

  Q_function
  -----------

  Function computes Q_function based on observables. Will probably use
  observables object from pyODEM.observables.
"""

import numpy as np
from pyODEM.observables import Observable
from mpi4py import MPI

class Mutational_Model_Linear():
    """
    Class holds all the information needed to model protein mutational experiments

    """
    def __init__(self, protein_model,partition, dtrajs, distribution,data):
        """
        Initializer. Holds everything that is required for calculations
        """
        self.mutant_list = []
        self.distribution = distribution
        self.rescale_temperature = False
        self.folded_states = self._get_microstates(0, dtrajs, partition)
        self.unfolded_states = self._get_microstates(1, dtrajs, partition)
        self.transition_states = self._get_microstates(2,dtrajs,partition)
        self.Q = self.compute_Q(protein_model, data)
        return

    def add_mutant(self,
                   name,
                   fraction,
                   compute_ddG_U2F=True,
                   experiment_ddG_U2F=None,
                   experiment_error_ddG_U2F=None,
                   compute_ddG_U2T=False,
                   experiment_ddG_U2T=None,
                   experiment_error_ddG_U2T=None):
        """
        Add another mutant to experiment model.
        See description of the parameters in docs for Mutant class
        """
        new_mutant = Mutant(name,
                            fraction,
                            compute_ddG_U2F=compute_ddG_U2F,
                            experiment_ddG_U2F=experiment_ddG_U2F,
                            experiment_error_ddG_U2F=experiment_error_ddG_U2F,
                            compute_ddG_U2T=compute_ddG_U2T,
                            experiment_ddG_U2T=experiment_ddG_U2T,
                            experiment_error_ddG_U2T=experiment_error_ddG_U2T
                            )
        self.mutant_list.append(new_mutant)
        return

    def _get_microstates(self, index, dtrajs, partition):
        """
        The method finds indexes of all the microstates, that correspond
        to particular macrostate.

        Parameters
        ----------
        index : int
                index of macrostate

        Return: set of int
                Set of indexes of microstates

        """
        microstates = []
        for microstate, macrostate in zip(dtrajs, partition):
            if macrostate == index:
                microstates.append(microstate)
        microstates = np.array(list(set(microstates)), dtype='int')
        return microstates


    def _compute_H(self, epsilons):
        """
        The method computes Hamiltonian for all the frames.
        Returns list of 1 dimensional numpy arrays. Element with index X in the
        returned list corresponds to the microstate X. Elemens of 1D array correspond
        to energy of Hamiltonian, one value for each frame, that corresponds to that
        microstate
        """
        H = []
        for microstate_array in self.Q:
            microstate_H = np.dot(epsilons, microstate_array)
            H.append(microstate_H)
        return(H)

    def _compute_reweight_prefactor(self, epsilon_old):
        """
        The method computes prefactor, that does not change during optimization.
        it is exp(-beta*H(epsilon_old))
        """
        H = self._compute_H(epsilon_old)
        prefactor = []
        for microstate in H:
            prefactor.append(np.exp(microstate))
        self.prefactor = prefactor
        return

    def prepare_observables(self, optimize=True, epsilon=None):
        """
        Method prepares observable for optimization by
        calculating prefactor and counting number of microstates for a
        """
        if optimize:
            self._compute_reweight_prefactor(epsilon)
        return

    def _reweight_microstates(self, epsilons_new):
        """
        The function produces reweighted values for microstates based on change in epsilon
        upon optimization
        """

        distribution_reweighted = np.zeros(np.shape(self.distribution)[0])
        new_H = self._compute_H(epsilons_new)
        for microstate_ndx, microstate_array in enumerate(new_H):
            new_exponent = np.exp(microstate_array)
            change = np.divide(new_exponent, self.prefactor[microstate_ndx])
            distribution_reweighted[microstate_ndx] = self.distribution[microstate_ndx] * \
                np.mean(change)
        distribution_reweighted = np.divide(
            distribution_reweighted, np.sum(distribution_reweighted))
        self.distribution_reweighted = np.array(distribution_reweighted)
        return

    def set_temperature_rescaling(self, experiment_temperature, folding_temperature):
        """
        The method creates rescaling factor atribute, i.e ratio of folding temperature
        and experimental temperature. It alows to normalize results simulated at folding
        temperature to the temperature yielding the same stablity as measured in experiment

        experiment_temperature: float
                                 Temperature(K), at which mutational experiment was counducted
        folding_temperature: float
                                  Tagret temperature(K), usually experimental folding
                                  temperature of protein.
        """

        self.rescale_temperature = True
        self.scaling_facror = float(folding_temperature)/float(experiment_temperature)
        return

    def compute_delta_delta_G(self,
                              epsilons,
                              compute_derivative=False,
                              reweighted=True,
                              grad_parameters=None):
        """
        The function computes a delta_delta_G for all the mutations in the model
        -----------

        compute_derivative: bool
                             If true, the function returns also derivative of
                             deltaG
        epsilon: array of array-like object of float
                             Contains  parameters of the model

        reweighted:
        """
        #First, compute everything that is the same for all the mutants
        # Microstate reweighting
        if reweighted:
            self._reweight_microstates(epsilons)
            distribution = self.distribution_reweighted
        else:
            distribution = self.distribution

        # Create the references to  distribution for folded, unfolded and transition
        # states, as well as corresponding normalizations. Is reusable for different
        # mutants
        distribution_folded_slice = distribution[self.folded_states]
        distribution_folded_normalization = np.sum(distribution_folded_slice)
        distribution_unfolded_slice = distribution[self.unfolded_states]
        distribution_unfolded_normalization = np.sum(distribution_unfolded_slice)
        distribution_transition_slice = distribution[self.transition_states]
        distribution_transition_normalization = np.sum(distribution_transition_slice)

        # Calculate the last term in the equation for derivatives
        if compute_derivative:
            aver_Q_values_folded = self.compute_average_Q(self.folded_states,
                                                         distribution_folded_slice,
                                                         distribution_folded_normalization)
            aver_Q_values_unfolded = self.compute_average_Q(self.unfolded_states,
                                                           distribution_unfolded_slice,
                                                           distribution_unfolded_normalization)
            aver_Q_values_transition = self.compute_average_Q(self.transition_states,
                                                           distribution_transition_slice,
                                                           distribution_transition_normalization)
        ddG_U2F_list = []
        ddG_U2T_list = []
        ddG_U2F_derivative_list = []
        ddG_U2T_derivative_list = []

        for mutant in self.mutants:
            corrected_epsilons = np.multiply(epsilons[self.mask], mutant.deleted_negative)
            if mutant.compute_ddG_U2F:
                aver_exp_delta_H_folded, aver_product_folded = self.compute_mutant_dependent_terms(
                                                   self.folded_states,
                                                   distribution_folded_slice,
                                                   distribution_folded_normalization,
                                                   mutant.fraction,
                                                   mutant.mask,
                                                   compute_derivative=compute_derivative)
                aver_exp_delta_H_unfolded, aver_product_unfolded = self.compute_mutant_dependent_terms(
                                                  self.unfolded_states,
                                                  distribution_unfolded_slice,
                                                  distribution_unfolded_normalization,
                                                  mutant.fraction,
                                                  mutant.mask,
                                                  compute_derivative=compute_derivative)
                dG_folded = -np.log(aver_exp_delta_H_folded)
                dG_unfolded = -np.log(aver_exp_delta_H_unfolded)
                ddG_U2F = dG_folded - dG_unfolded
                if self.rescale_temperature:
                    ddG_U2F *= self.scaling_facror
                ddG_U2F_list.append(ddG_U2F)
                if compute_derivative:
                    derivative_folded = np.add(-1*np.divide(np.multiply(mutant.fraction,
                                                                 aver_product_folded),
                                                                 aver_exp_delta_H_folded),
                                                                 aver_Q_values_folded)
                    derivative_unfolded = np.add(-1*np.divide(np.multiply(mutant.fraction,
                                                                 aver_product_unfolded),
                                                                 aver_exp_delta_H_unfolded),
                                                                 aver_Q_values_unfolded)
                    ddG_U2F_derivative = np.subtract(derivative_folded,derivative_unfolded)
                    if self.rescale_temperature:
                        ddG_U2F_derivative *= self.scaling_factor
                    ddG_U2F_derivative_list.append(ddG_U2F_derivative)
            if mutant.compute_ddG_U2F and mutant.comute_ddG_U2T:
                # If both ddG nots and ddG dagger should be computed,
                # need to add data for transition state
                aver_exp_delta_H_transition, aver_product_transition = self.compute_mutant_dependent_terms(
                                                   self.folded_states,
                                                   distribution_transition_slice,
                                                   distribution_transition_normalization,
                                                   mutant.fraction,
                                                   mutant.mask,
                                                   compute_derivative=compute_derivative)
                dG_transition = -np.log(aver_exp_delta_H_transition)
                ddG_U2T = dG_transition - dG_unfolded
                if self.rescale_temperature:
                    ddG_U2T *= self.scaling_facror
                ddG_U2T_list.append(ddG_U2T)

                if compute_derivative:
                    derivative_transition = np.add(-1*np.divide(np.multiply(mutant.fraction,
                                                                 aver_product_transition),
                                                                 aver_exp_delta_H_transition),
                                                                 aver_Q_values_transition)
                    ddG_U2T_derivative = np.subtract(derivative_transition,derivative_unfolded)
                    if self.rescale_temperature:
                        ddG_U2T_derivative *= self.scaling_factor
                    ddG_U2F_derivative_list.append(ddG_U2F_derivative)
        return ddG_U2F_list, ddG_U2T_list, ddG_U2F_derivative_list, ddG_U2T_derivative_list


    def compute_average_Q(self,
                          macrostate,
                          distribution_slice,
                          distribution_normalization):
        """
        Compute <Qij> for macrostate `macrostate`.
        Should be calculated once for each call for ddG calculations for each macrostate.
        Is only calculated when derivative are of interest.
        """
        mean_Q_values = []
        # Calculating average value of Q for each microstate
        for microstate in macrostate:
            Q_microstate = self.Q[microstate]
            mean_Q = np.mean(Q_microstate, axis=1)
            mean_Q_values.append(mean_Q)
        #
        mean_Q_values = np.array(mean_Q_values)
        aver_Q_values = np.dot(distribution_slice, mean_Q_values)/distribution_normalization
        return aver_Q_values

    def compute_mutant_dependent_terms(self,
                                       macrostate,
                                       distribution_slice,
                                       distribution_normalization,
                                       fraction,
                                       mask,
                                       compute_derivative=False):
        """
        Function computes terms of dG and \partial dG \partial \epsilon:
        ensemble average for exp(-beta*delta H) and
        for exp(-beta*delta H)*Q
        """
        mean_exp_delta_H = []
        if compute_derivative:
            mean_products = []
        for microstate in macrostate:
            Q_microstate = self.Q[microstate]
            microstate_delta_H = np.dot(corrected_epsilons, Q_microstate[mask])
            microstate_exp_delta_H = np.exp(microstate_delta_H)
            mean_microstate_exp_delta_H = np.mean(microstate_exp_delta_H)
            mean_exp_delta_H.append(mean_microstate_exp_delta_H)
            if compute_derivative:
                product = np.multiply(Q_microstate, microstate_exp_delta_H)
                mean_product = np.mean(product, axis=1)
                mean_products.append(mean_product)
        mean_exp_delta_H = np.array(mean_exp_delta_H)
        aver_exp_delta_H = np.dot(distribution_slice, mean_exp_delta_H)/distribution_normalization
        if compute_derivative:
            mean_product = np.array(mean_product)
            aver_product = np.dot(distribution_slice, mean_products)/distribution_normalization
            return aver_exp_delta_H, None

        return aver_exp_delta_H, aver_product



    @staticmethod
    def compute_Q(model, data):
        """
        The function can be used only for cases when Hamiltonian can be represented as
        H = sum(epsilon_i*Q_i).
        The function computes Q values for each parameter of each frame.
        Needs to be computed only once per each optimization in case of
        linear hamiltonian. The result is presented in the same form as self.data
        Ideally,
        The q values computed are in kT units and include "-" sign.

        Parameters
        ----------

        model - protein model
        data  - formatted data, as generated by model_loader
        """
        Q = []
        # Need to use a random set of parameters. For safety, make all parameters
        # equal to 1
        epsilon = model.get_epsilons()
        epsilon = np.ones(len(epsilon))
        for dictionary in data:
            state = dictionary['index']
            values = np.array(dictionary['data'])
            _, depsilons = model.get_potentials_epsilon(values)
            Q.append(np.array(depsilons(epsilon)))
        return(Q)


class Mutant():
    """
    Class holds  information regarding each mutant and its
    representation in SBM model.

    Attributes
    -----------
    name : str
           destription of the mutant in format <1-letter aminoacid code for aminoacid
         to be replaced in the wild-type protein><number of altered residue (indexed
           from 1)><1-letter aminoacid code for corresponding aminoacid in mutant>
    fraction : 1d-numpy array, dtype=float
             Define fraction of contacts, that remain after mutation.
             Length should be equal to number of contacts in the model
    compute_ddG_U2F : bool
                    If true, ddG nots will be computed
    compute_ddG_U2T : bool
                    If true, ddG dagger will be computed
    experiment_ddG_U2F : float
                    Experimental value for ddG nots, units of RT
    experiment_ddG_T2F : float
                    Experimental value for ddG dagger, units of RT
    experiment_error_ddG_U2F : float
                    Experimental error for ddG nots, units of RT
    experiment_error_ddG_T2F : float
                    Experimental error for ddG dagger, units of RT

    """
    def __init__(self,
                 name,
                 fraction,
                 compute_ddG_U2F=False,
                 experiment_ddG_U2F=None,
                 experiment_error_ddG_U2F=None,
                 compute_ddG_U2T=False,
                 experiment_ddG_U2T=None,
                 experiment_error_ddG_U2T=None
                ):
        self.name=name
        self.fraction = fraction  # list, that defines fractions of contacts,
        # that remains after mutation
        # Negative fraction of delated contacts (Negative for calculation simplification)
        self.mask = self._get_mask()
        self.deleted_negative = fraction[self.mask] - 1
        self.compute_ddG_U2T =  compute_ddG_U2T
        self.compute_ddG_U2F = compute_ddG_U2F
        self.experiment_ddG_U2T = experiment_ddG_U2T
        self.experiment_ddG_U2F = experiment_ddG_U2F
        self.experiment_error_ddG_U2T = experiment_error_ddG_U2T
        self.experiment_error_ddG_U2T = experiment_error_ddG_U2F
        return

    def _get_mask(self):
        """
        The function returns indexes of the parameters, that are affected by the
        mutation. Only those parameters need to be included in the calculation of ddG
        """
        mask = np.transpose(np.argwhere(self.fraction < 0.99999999))[0]
        return mask

    def compute_observation(self, epsilons):
        """
        The function returns observed values of delta_delta_G and
         corresponding derivatives
        """

        return self._compute_delta_delta_G(epsilons, compute_derivative=True)
