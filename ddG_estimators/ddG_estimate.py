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



# Set of helper functions will go here.
def compute_ensemble_average(data,dtraj,pi):
    """
    Ensemble average  for a given dataset and discrete trajectory.

    Parameters
    -----------

    data :  numpy array
            1D array or array-like object of length N, where N represents
            number of frames in ensemble. Each entry of array is a single value
            for one frame.
    dtraj : list of int
            list of integers with length N. Each entry corresponds to number
            of microstate. Microstates are indexed from 0.
    pi    : numpy array
            1D array or array-like object that contains equilibrium distribution.
            Length of the array is equal to number of different states in dtraj.
            i-th element of pi should represent probability of state i.

    Returns
    -------

    average : float
              Value of ensemble average
    """

    n_microstate = len(pi)

    state_average = np.zeros(n_microstate).astype('float')
    state_count = np.zeros(n_microstate).astype('int')

    for state, value in zip(dtraj,data):
        state_average[state] += value
        state_count[state] += 1

    average = 0
    for microstate in range(n_microstate):
        average += state_average[microstate]/state_count[microstate]*pi[microstate]
    return average


# Test. Should be moved to tests later.

def test_compute_ensemble_average():
    data = np.array([0.1,0.2,0.7,0.9,1.5,0.1,0.34,0.56,0.36,0.3])
    dtraj = [0,0,1,1,1,1,1,2,2,0]
    pi = np.array([0.25,0.4,0.35])
    expected_value = 0.4942
    computed_value = compute_ensemble_average(data,dtraj,pi)
    assert(computed_value == expected_value)


def unwrap_formatted_data(input_data,key):
    """
    The method converts formatted data of the following format:
    [
    {'index': <int>, 'key':[ <one list of values for each frame, that belongs to state with a particular index>]},
    ...
    ]

    to two numpy arrays.
    output_states : 1D  numpy array of ints
                    Length is equal to number of frames in trajectory.
    output_data   : 2D numpy array. Each row represents on frame, each column represents on parameter.

    Parameters
    -----------

    input_data : list of dictionaries
                 see description above

    key       : str
                Key of dictionary used for unwrapping
    """

    output_states = []
    output_data = []
    for microstates in input_data:
        for frames in microstates[key]:
            output_states.append(microstates['index'])
            output_data.append(frames)
    output_states = np.array(output_states,dtype=int)
    output_data = np.array(output_data)
    assert output_data.ndim == 2
    return output_states, output_data

class ddG(Observable):
    """
    The class holds all the methods, that are required to compute and manipulate
    mutational delta delta G value. The class inherits from pyODEM observable
    object.
    """



    def __init__(self,model,data,partition,fraction,dtrajs_old,distribution,debug=False):
        """Initialize object.
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
        self.partition = partition #hould be a descrete trajectory between macrostates.
        self.dtrajs, self.data = unwrap_formatted_data(data,'data')
        self.fraction  = fraction # list, that defines fractions of contacts,
        self.dtrajs_old = dtrajs_old
                                   # that remains after mutation
        self.distribution = distribution
        self.debug = debug # debug flag
        if self.debug:
            print("fractions for mutations")
            print(self.fraction)
            print("Discrete trajectory in microstate space")
            print(self.dtrajs)

    def _get_microstates(self,index):
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
        for microstate, macrostate in zip(self.dtrajs_old,self.partition):
            if macrostate == index:
                microstates.append(microstate)
        microstates = set(microstates)
        return microstates


    def _compute_H(self,epsilons,compute_derivative=False):
        """
        The method computes unmutated hamiltonian values  and
        corresponding derivatives for each frame of the trajectory
        each microstate.

        Parameters
        ----------

        epsilons : numpy.ndarray of floats
                   Array of model parameters. Order should be the same as
                   order of parameters in the model used to create object
        compute_deriative : bool
                            If true, both hamiltonian and derivative with respect
                            to epsilon are computed

        Return
        ------

        H : 1D numpy.ndarray
            1 float value per frame, representing energy of Hamiltonian
        dH : 2D numpy.ndarray
             Array of H derivatives with respect to epsilons
             shape (N,M), where N - index of parameter (epsilon),
             M - index of the frame. Order of frame is the seme as in self.data

        """
        hepsilons, depsilons  = self.model.get_potentials_epsilon(self.data)
        H = hepsilons(epsilons)
        dH = None
        if compute_derivative:
                dH = depsilons(epsilons)
        return np.array(H), np.array(dH)

    def _compute_reweight_prefactor(self,epsilon_old):
        """
        The method computes prefactor, that does not change during optimization.
        it is exp(-beta*H(epsilon_old))
        """
        H,_ = self._compute_H(epsilon_old)
        prefactor  = np.exp(H)
        self.prefactor = prefactor
        return

    def _count_frames_in_microstates(self):
        """
        The method counts number of frames in each microstate.
        Creates a new atribute count, wich is a 1d numpy array,
        element [i] of which holds number of frames, that correspond
        to  microstate i.
        """
        counts = np.zeros(np.shape(self.distribution)[0],dtype=int)
        for state in self.dtrajs:
            counts[state] += 1
        self.counts = counts
        return

    def prepare_observables(self,optimize=True, epsilon=None):
        """
        Method prepares observable for optimization by
        calculating prefactor and  counting number of microstates for a
        """
        self._count_frames_in_microstates()
        if optimize:
            self._compute_reweight_prefactor(epsilon)
        return

    def _reweight_microstates(self,epsilons_new):
        """
        The function produces reweighted values for microstates based on change in epsilon
        upon optimization
        """
        average_weight = np.zeros(np.shape(self.distribution)[0])
        new_H, _ = self._compute_H(epsilons_new)
        new_exponent = np.exp(new_H)
        change = np.divide(new_exponent,self.prefactor)
        for state, difference in zip (self.dtrajs, change):
            average_weight[state] += difference
        average_weight = np.divide(average_weight,self.counts)
        distribution_reweighted =  np.multiply(self.distribution,average_weight)
        norm = np.sum(distribution_reweighted)
        distribution_reweighted /= norm
        self.distribution_reweighted = distribution_reweighted
        return


    def _compute_mutated_H(self,epsilons,compute_derivative=False):
        """
        The function computes mutated Hamiltonian. Parameters and returns is the
        same as in _compute_H method
        """
        mutated_epsilons = np.multiply(epsilons,self.fraction)


        H, dH = self._compute_H(mutated_epsilons,compute_derivative=compute_derivative)
        if compute_derivative:
            for frame_ndx in range(0,dH.shape[1]):
                dH[:,frame_ndx] = np.multiply(dH[:,frame_ndx],self.fraction)
        return H, dH

    def _compute_delta_H(self,H0,mutated_H):
        """
        The function computes change in Hamiltonian upon mutation.
        Parameters
        ----------
        H0, mutated_H : 1D numpy.ndarray
                        Native and mutated Hamiltonian, correspondingly

        """
        assert H0.shape == mutated_H.shape
        return np.subtract(mutated_H,H0)

    def _compute_d_delta_H(d_H0,d_mutated_H):
        """
        The function computes derivative of change in Hamiltonian upon mutation
        with respect to model parameters.
        Parameters
        ----------
        d_H0, d_mutated_H : 2D numpy.ndarray
                        Derivatives of native and mutated Hamiltoian, correspondingly
        """
        assert d_H0.shape == d_mutated_H.shape
        return np.subtract(d_mutated_H,d_H0)


    def _get_microstate_averages(self,data,non_frame_axis=None):
        """
        The function computes microstate averages for an array.
        data : 1D or 2D numpy array.
        """
        num_of_microstates = len(set(self.dtrajs))
        if data.ndim == 1:
            microstate_averages = np.zeros((num_of_microstates))
            counts = np.zeros(num_of_microstates,dtype=int)
            for microstate, value in zip(self.dtrajs,data):
                counts[microstate] += 1
                microstate_averages[microstate] += value
            microstate_averages = np.divide(microstate_averages,counts)
            return np.array(microstate_averages)
        elif data.ndim == 2:
            assert non_frame_axis is not None, "Specify an additional axis"
            number_of_values = data.shape[non_frame_axis]
            microstate_averages = np.zeros((num_of_microstates,number_of_values))
            counts = np.zeros(num_of_microstates,dtype=int)
            if non_frame_axis == 0:
                for frame,microstate in enumerate(self.dtrajs):
                    microstate_averages[microstate,:] += data[:,frame]
                    counts[microstate] += 1
            if non_frame_axis == 1:
                for frame,microstate in enumerate(self.dtrajs):
                    microstate_averages[microstate,:] += data[frame,:]
                    counts[microstate] += 1
            for parameter in range(number_of_values):
                microstate_averages[:,parameter] /= counts
            return microstate_averages
        else:
            raise ValueError("Array for averaging should have no more than 1 dimensions")

    def _get_ensemble_averages(self,macrostate,microstate_averages,reweighted=False,epsilons=None):
        """
        The function computes ensemble average for a particular macrostate
        data : 1D or 2D numpy array.
        """
        microstates = self._get_microstates(macrostate)
        if reweighted:
            distribution = self.distribution_reweighted
        else:
            distribution = self.distribution

        if microstate_averages.ndim == 1:
            average = 0.0
            normalization = 0.0
            for microstate  in microstates:
                average += microstate_averages[microstate]*distribution[microstate]
                normalization += distribution[microstate]

            average /= normalization
            return average
        else:
            raise ValueError("Array for averaging should have no more than 1 dimensions")



    def compute_delta_delta_G(self,epsilons,compute_derivative=False,reweighted=True):
        """
        The function computes a delta_delta_G of mutation for a particular macrostate.
        Parameters
        -----------

        compute_derivative : bool
                             If true, the function returns also derivative of
                             deltaG
        epsilon            : array of array-like object of float
                             Contains  parameters of the model

        reweighted         :
        """
        # Find all the  microstates, that correspond to a particular microstate
        if reweighted:
             self._reweight_microstates(epsilons)
        H0,d_H0 = self._compute_H(epsilons,compute_derivative=compute_derivative)
        H_mutated, d_H_mutated = self._compute_mutated_H(epsilons,compute_derivative=compute_derivative)
        exp_delta_H = np.exp(self._compute_delta_H(H0,H_mutated))
        exp_delta_H_micro_aver = self._get_microstate_averages(exp_delta_H)
        aver_folded = self._get_ensemble_averages(0,exp_delta_H_micro_aver,reweighted=reweighted,epsilons=epsilons)
        aver_unfolded = self._get_ensemble_averages(1,exp_delta_H_micro_aver,reweighted=reweighted,epsilons=epsilons)
        delta_delta_G = -1*np.log(aver_folded) + np.log(aver_unfolded)

        if compute_derivative:
            derivatives = []
            exp_product_dHm = np.copy(d_H_mutated) #Compute product
            for parameters in range(exp_product_dHm.shape[0]):
                exp_product_dHm[parameters,:] = np.multiply(exp_product_dHm[parameters,:],exp_delta_H)
            # Compute microstate averages for
            aver_exp_product_dHm = self._get_microstate_averages(exp_product_dHm, non_frame_axis=0)
            aver_d_H0 = self._get_microstate_averages(d_H0, non_frame_axis=0)
            derivatives = []
            for parameters in range(aver_d_H0.shape[1]):
                product_folded = self._get_ensemble_averages(0,aver_exp_product_dHm[:,parameters],reweighted=True,epsilons=epsilons)
                product_unfolded = self._get_ensemble_averages(1,aver_exp_product_dHm[:,parameters],reweighted=True,epsilons=epsilons)
                dH_0_folded = self._get_ensemble_averages(0,aver_d_H0[:,parameters],reweighted=reweighted,epsilons=epsilons)
                dH_0_unfolded = self._get_ensemble_averages(1,aver_d_H0[:,parameters],reweighted=reweighted,epsilons=epsilons)
                d_delta_G_folded = product_folded/aver_folded - dH_0_folded
                d_delta_G_unfolded = product_unfolded/aver_unfolded - dH_0_unfolded
                result = -1*(d_delta_G_folded - d_delta_G_unfolded) #Need to multipy by -1, because all the hamiltonians return -beta*H
                derivatives.append(result)
            return delta_delta_G, derivatives


        return delta_delta_G



    def compute_observation(self,epsilons):
        """
        The function returns observed values of delta_delta_G and
         corresponding derivatives
        """

        return self._compute_delta_delta_G(epsilons,compute_derivative=True)
