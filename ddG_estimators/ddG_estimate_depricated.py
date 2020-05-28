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

test_compute_ensemble_average()



class DiscreteTrajectory():
    """
    The class holds methods to work with descrete trajectories. The descrete trajectory is
    build around a list of dictionaries. Each dictionary in this list represents one
    descrete microstate.  At this point, the list of dictionaries is passed during initiali
    zation. Later, an alternative constractor may be created
    """

    def __init__(self,data):
        """
        Initialization method. Directly passed pre-formated dictionary
        """
        self.data = data

    def check_dimensions(self):
        """
        Method checks, that all the data fields have two or less dimentions
        """
        for microstate in data:
            for key in microstate:
                if isinstance(microstate[key],list):
                    assert np.array(microstate[key]).ndim < 3, "Dimension of %s in state %d is > 2.  " %(key,microstate['index'])


    def add_new_function(self,function,key_initial,key_final):
        """
        The method creates a new dictionary, computed based on
        values in existant keys.
        Parameters
        ----------
        function :  function object
                    A function, that computes new values based on existing parameters.

        key_initial : str
                      Name of the key, that is used as parameters for function
        key_final :  str
                     Name of key for new value in dictionary
        """

        for microstate in self.data:
            self.data[key_final] = function([self.data[key_initial]])



class ddG(Observable):
    """
    The class holds all the methods, that are required to compute and manipulate
    mutational delta delta G value. The class inherits from pyODEM observable
    object.
    """

    def __init__(self,model,data,partition,dtrajs,fraction,distribution,debug=False):
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
        self.data = data
        self.partition = partition #hould be a descrete trajectory between macrostates.
        self.dtrajs = dtrajs # A descrete trajectory between microstates
        self.fraction  = fraction # list, that defines fractions of contacts,
                                   # that remains after mutation
        self.distribution = distribution
        self.debug = debug # debug flag
        if self.debug:
            print("fractions for mutations")
            print(fraction)
            print("Discrete trajectory in microstate space")
            print(dtrajs)
        print(data)

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
        for microstate, macrostate in zip(self.dtrajs,self.partition):
            if macrostate == index:
                microstates.append(microstate)
        microstates = set(microstates)
        return microstates


    def _compute_H(self,microstates,epsilons,compute_derivative=False):
        """
        The method computes unmutated hamiltonian values  and
        corresponding derivatives for each frame of
        each microstate. Returns a list dictionaries, one dictionary
        for each microstate.
         where the first key(['index']) corresponds to microstate index
         the second key ['H'] holds a list of floats, where each value represents
         H computed with particular epsilons
         the third key ['dH'] holds a list of floats, where each values represents
         H derivative, computed with particular epsilons
        """

        H_values = []
        for microstate in microstates:
            dict = {}
            hepsilons, depsilons  = self.model.get_potentials_epsilon(self.data[microstate]['data'])
            dict['index'] = microstate
            dict['H'] = hepsilons(epsilons)
            if compute_derivative:
                dict['dH'] = depsilons(epsilons)
            H_values.append(dict)
            if self.debug:
                print("Hamiltonian and hamiltonain derivatives")
                print(H_values)
        return H_values

    def _compute_mutated_H(self,microstates,epsilons,compute_derivative=False):
        """
        The function computes mutated Hamiltonian. Fore details, see _compute_H method
        """
        mutated_epsilons = np.multiply(epsilons,self.fraction)
        if self.debug:
            print("Mutated_epsilons")
            print(mutated_epsilons)
        H_values = self._compute_H(microstates,mutated_epsilons,compute_derivative=compute_derivative)
        if compute_derivative:
            #In compute_H, derivative is taken with respect to mutated_epsilons.
            # But we would like to get a derivative with respect to original epsilons.
            for microstate in H_values:
                for frame in microstate['dH']:
                    frame = np.multiply(frame,self.fraction)
            if self.debug:
                print("Mutated Hamiltonian derivative")
                print(H_values)
        return H_values

    def _compute_delta_H(self,H0,mutated_H,compute_derivative=False):
        """
        The function computes change in Hamiltonian upon mutation.
        Parameters
        ----------
        H0 : list of dicts
              Output of _compute_H function
        mutated_H : list of dicts
                    Output of _compute_mutated_H function

        """

        assert len(H0) == len(mutated_H)
        delta_H = []
        for microstate0, microstate_mutated in zip(H0,mutated_H):
            microstate_delta_H = {}
            assert microstate0['index'] == microstate_mutated['index']
            microstate_delta_H['index'] = microstate0['index']
            microstate_delta_H['delta_H'] = np.subtract(microstate_mutated['H'],microstate0['H'])
            delta_H.append(microstate_delta_H)
            if compute_derivative == True:
                microstate_delta_H['delta_dH']=np.subtract(microstate_mutated['dH'],microstate0['dH'])

        return delta_H

    def _compute_microstate_average(self,data,keys):
        """
        The method computes microstate average based on data list

        Parameters
        -----------
        data : list of dictionaries
               Each dictionary contains index key, that represents
               number of frames, and different keys with data. Each
               data entry is a list, each element of this list represents information
               for one frame.
               {'index' : <int, index of microstate>,
                 'key1' : <list, each entry on the list corresponds to one frame, that
                          belongs to that particular microstate>,
                 'key2" : <list, the same as key1
                          }
        keys :   list of  'str'
                 List of keys, that should be used for averaging.

        Returns
        --------

        aver_data : list of dictionaries with microstate averages.

        """
        aver_data = []
        for microstate in data:
            microstate_aver = {}
            microstate_aver['index'] = microstate['index']
            for key in keys:
                data_array = np.array(microstate[key])
                if self.debug:
                    if microstate['index'] == 0:
                        print(data_array)
                assert data_array.ndim <= 2, "arrays with more than two dimensions are not accepted as data"
                microstate_aver[key] = np.mean(data_array,axis=-1)
            aver_data.append(microstate_aver)
        if self.debug:
            print(aver_data)
        return (aver_data)


    def _compute_ensemble_average(self,data,keys):
        """
        The method computes ensemble averages for data in the dictionary.
        Parameters
        ----------
        data : list of dictionaries


        """
        aver_data = self._compute_microstate_average(data,keys)
        ensemble_averages = {}
        for key in keys:
            normalization = 0
            if np.array(aver_data[0][key]).ndim == 0:
                key_average = 0.0
                for microstate in aver_data:
                    key_average += self.distribution[microstate['index']]*microstate[key]
                    normalization += self.distribution[microstate['index']]
                key_average = key_average/normalization

            elif np.array(aver_data[0][key]).ndim == 1:
                key_average = np.zeros(np.array(aver_data[0][key]).shape[0])
                for microstate in aver_data:
                    key_average += np.multiply(self.distribution[microstate['index']],microstate[key])
                    normalization += self.distribution[microstate['index']]
                key_average = np.divide(key_average,normalization)
            ensemble_averages[key] = key_average
        return ensemble_averages


    def get_boltzman_weight(self,H):
        """The function computes boltzman weight for Hamiltonian, exp(-beta*H)
           Parameters
           ----------
           H : np.array
               Array of floats

            Returns
            -------
             np.array, the same size as H
             exponent = exp(H). H already includes beta.

        """

        #return np.exp(-1*self.model.beta*H)
        return np.exp(H)


    def add_new_function(self,dictionary,function,key_initial,key_final):
        """
        The method creates a new dictionary, computed based on
        values in existant keys.
        Parameters
        ----------
        dictionary : dictionary to use
        function :  function object
                    A function, that computes new values based on existing parameters.

        key_initial : str
                      Name of the key, that is used as parameters for function
        key_final :  str
                     Name of key for new value in dictionary
        """

        for microstate in dictionary:
            microstate[key_final] = function(microstate[key_initial])


    def _compute_delta_G(self,macrostate,epsilons,compute_derivative=False):
        """
        The function computes a delta_G of mutation for a particular macrostate.
        Parameters
        -----------
        marcostate : int
                     Index of a macrostate to use for computation

        compute_derivative : bool
                             If true, the function returns also derivative of
                             deltaG
        epsilon            : array of array-like object of float
                             Contains  parameters of the model
        """
        # Find all the  microstates, that correspond to a particular microstate
        microstates = self._get_microstates(macrostate)
        H0 = self._compute_H(microstates,epsilons,compute_derivative=compute_derivative)
        H_mutated = self._compute_mutated_H(microstates,epsilons,compute_derivative=compute_derivative)
        delta_H = self._compute_delta_H(H0,H_mutated,compute_derivative=compute_derivative)
        if self.debug:
            print(delta_H[0])
        self.add_new_function(delta_H,self.get_boltzman_weight,'delta_H','exp(-bH)')
        average = self._compute_ensemble_average(delta_H,['exp(-bH)'])
        if self.debug:
            print("Average value of expenent of  delta_H")
            print(average)
        delta_G = -np.log(average['exp(-bH)'])
        if compute_derivative == True:
            for index,microstate in enumerate(delta_H):
                delta_H[index]['exp_dH_mutated']=np.multiply(microstate['exp(-bH)'],H_mutated[index]['dH'])
            average_exp_times_derivative = self._compute_ensemble_average(delta_H,['exp_dH_mutated'])
            average_dH0 = self._compute_ensemble_average(H0,['dH'])
            d_delta_G = np.subtract(np.divide(average_exp_times_derivative['exp_dH_mutated'],average['exp(-bH)']),average_dH0['dH'])
            return(delta_G,d_delta_G)
        return(delta_G)


    def  compute_delta_delta_G(self,
                                epsilons,
                                state_dictionary = {'F':0,'U':1},
                                compute_derivative=False):
        """
        The function computes delta_delta_G based on epsilons.
        Parameters:
        -----------
        epsilons : numpy array of floats
                   Contains model parameters
        state_dictionary : dict
                           Contains two keys, each of which can have an integer value.
                           Key 'F' corresponds to the index of macrostate, that is considered
                           as Folded state in macrostate trajectory file
                           Key 'U' corresponsd to the index of macrostate, that is considered as
                           Unfolded state in macrostate trajectory file
        compute_derivateve : bool
                             Compute DeltaDeltaG derivatives, if True

        RETURNS delta_delta_G/kT!

        """

        G_folded = self._compute_delta_G(state_dictionary['F'],epsilons,compute_derivative=compute_derivative)
        G_unfolded = self._compute_delta_G(state_dictionary['U'],epsilons,compute_derivative=compute_derivative)
        if compute_derivative:
            ddG = G_folded[0]-G_unfolded[0]
            ddG_derivative = G_folded[1]-G_unfolded[1]
            return ddG, ddG_derivative
        return G_folded - G_unfolded


    def compute_observation(self):
        """
        The function returns observed values of delta_deltaG.
        """
        pass
