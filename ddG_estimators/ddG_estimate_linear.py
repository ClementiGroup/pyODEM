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

# Set of helper functions will go here.


class ddG_linear(Observable):
    """
    The class holds all the methods, that are required to compute and manipulate
    mutational delta delta G value. It works when Hamiltonian linearly depends on
    parameters.
    The class inherits from pyODEM observable
    object.
    """

    def __init__(self, model, data, partition, fraction,  distribution, dtrajs, debug=False):
        """Initialize object.
        During initialization, data are converted into Q factor.
        Parameters
        -----------

        model  : Model object
                 object of a class that inherits from ModelLoader
                 see (model_loaders pachage). Should contain information
                 about model epsilons, and be able to compute variable part
                 of hamiltonian based on epsilons.

        partition :  list of int
                     list of length N, where N is number of frames
                     each entry corresponds to index of a macrostate for each
                     frame
        fraction :   list of floats
                     list of length m, where m - number of parameters. Each entry
                     is equal to a frection of


        """
        self.type = 'ddG'
        self.model = model
        self.fraction = fraction  # list, that defines fractions of contacts,
        # that remains after mutation
        # Negative fraction of delated contacts (Negative for calculation simplification)
        self.deleted_negative = fraction - 1
        self.distribution = distribution
        self.debug = debug  # debug flag
        self.rescale_temperature = False
        self.folded_states = self._get_microstates(0, dtrajs, partition)
        self.unfolded_states = self._get_microstates(1, dtrajs, partition)
        self._compute_Q(data)
        if self.debug:
            print("fractions for mutations")
            print(self.fraction)

    def _compute_Q(self, data):
        """
        The function computes Q values for each parameter of each frame.
        Needs to be computed only once per each optimization in case of
        linear hamiltonian. The result is presented in the same form as self.data
        Ideally, Q should olso be computed only once for each mutant.
        Probably, can create another class, that will inherit from this one.
        If clean=True, states that to not belong to either folded or unfolded state
        and have index -1 in macrostates
        The q values computed are in kT units and include "-" sign.
        """
        self.Q = []
        # Need to use a random set of parameters. For safety, make all parameters
        # equal to 1
        epsilon = self.model.get_epsilons()
        epsilon = np.ones(len(epsilon))
        for dictionary in data:
            state = dictionary['index']
            values = np.array(dictionary['data'])
            _, depsilons = self.model.get_potentials_epsilon(values)
            self.Q.append(depsilons(epsilon))
        if self.debug:
            print("Q-functions")
            print(self.Q)

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

    def compute_delta_G(self,
                        macrostate,
                        corrected_epsilons,
                        distribution,
                        compute_derivative=False):
        """
        The function computes delta_G and corresponding derivative for
         a particular ensemble.

         Parameters
         ----------

         macrostate : str {'folded' or 'unfolded'}
         A name of ensemble, for which calculations will be performed.

         corrected_epsilons : 1D numpy  array

         Model parameters, multiplied by NEGATIVE fraction of contacts,
          that were deleted upon mutation

          distribution :  1D numpy array
          equilibrium distribution to use for calculation. If reweighting is
          required, the distribution put here should already be reweighted.

          compute_derivative : bool
          If True, derivative is returend.

          Returns
          -------

         dG : float
         delta_G upon mutation for a particular ensemble

         derivative : 1D numpy arrray
         derivative of delta_G with respect to model parameters

        """

        microstate_sets = {'folded': self.folded_states, 'unfolded': self.unfolded_states}

        microstates = microstate_sets[macrostate]
        distribution_slice = distribution[microstates]
        mean_exp_delta_H = []
        if compute_derivative:
            mean_Q_values = []
            mean_products = []
        for microstate in microstates:
            Q_microstate = self.Q[microstate]
            microstate_delta_H = np.dot(corrected_epsilons, Q_microstate)
            microstate_exp_delta_H = np.exp(microstate_delta_H)
            mean_microstate_exp_delta_H = np.mean(microstate_exp_delta_H)
            mean_exp_delta_H.append(mean_microstate_exp_delta_H)
            if compute_derivative:
                mean_Q = np.mean(Q_microstate, axis=1)
                mean_Q_values.append(mean_Q)
                product = np.multiply(Q_microstate, microstate_exp_delta_H)
                mean_product = np.mean(product, axis=1)
                mean_products.append(mean_product)
        mean_exp_delta_H = np.array(mean_exp_delta_H)
        normalization = np.sum(distribution_slice)
        aver = np.dot(distribution_slice, mean_exp_delta_H)/normalization
        dG = -np.log(aver)
        if compute_derivative:
            mean_product = np.array(mean_product)
            mean_Q_values = np.array(mean_Q_values)
            aver_product = np.dot(distribution_slice, mean_products)/normalization
            aver_Q_values = np.dot(distribution_slice, mean_Q_values)/normalization
            derivative = np.add(-1*np.divide(np.multiply(self.fraction,
                                                         aver_product), aver), aver_Q_values)
            return dG, derivative

        return dG

    def compute_delta_delta_G(self, epsilons, compute_derivative=False, reweighted=True):
        """
        The function computes a delta_delta_G of mutation for a particular macrostate.
        Parameters
        -----------

        compute_derivative: bool
                             If true, the function returns also derivative of
                             deltaG
        epsilon: array of array-like object of float
                             Contains  parameters of the model

        reweighted:
        """
        # Find all the  microstates, that correspond to a particular microstate
        if reweighted:
            self._reweight_microstates(epsilons)
            distribution = self.distribution_reweighted
        else:
            distribution = self.distribution

        # exponent exp(-beta*delta_H) for folded state.
        corrected_epsilons = np.multiply(epsilons, self.deleted_negative)
        # All the analysis  required for folded microstates. Done in one
        # place to reduce unnecessary repeating loops.
        if compute_derivative:
            folded_DG, folded_derivative = self.compute_delta_G(macrostate='folded', corrected_epsilons=corrected_epsilons,
                                                                distribution=distribution, compute_derivative=compute_derivative)
            unfolded_DG, unfolded_derivative = self.compute_delta_G(macrostate='unfolded', corrected_epsilons=corrected_epsilons,
                                                                    distribution=distribution, compute_derivative=compute_derivative)
            delta_delta_G = folded_DG - unfolded_DG
            derivative = np.subtract(folded_derivative, unfolded_derivative)
            if self.rescale_temperature:
                delta_delta_G *= self.scaling_facror
                derivative *= self.scaling_facror
            return delta_delta_G, derivative
        else:
            folded_DG = self.compute_delta_G(macrostate='folded', corrected_epsilons=corrected_epsilons,
                                             distribution=distribution, compute_derivative=compute_derivative)
            unfolded_DG = self.compute_delta_G(macrostate='unfolded', corrected_epsilons=corrected_epsilons,
                                               distribution=distribution, compute_derivative=compute_derivative)
            delta_delta_G = folded_DG - unfolded_DG
            if self.rescale_temperature:
                delta_delta_G *= self.scaling_facror
            return delta_delta_G

    def compute_observation(self, epsilons):
        """
        The function returns observed values of delta_delta_G and
         corresponding derivatives
        """

        return self._compute_delta_delta_G(epsilons, compute_derivative=True)
