"""
The package contains classes needed for manipulation with delta_delta_G variable.
At this point, it is completely separated from classes, that already exist in pyODEM,
though it uses some of the helper functions.

"""
import numpy as np

class Likelihood():
    """
    Collects all the observables and corresponding derivatives and create
    Likelihood

    Attributes:
    -----------
    observable_object_list : list of objects
                            Each object represents 1 observable/group of
                            observables. Each object should have a method
                            `compute_observation`.

    observed_value_list : list of numpy arrays.
                         Each numpy array can be 0d or 1d and represents experimental
                         values for the observables, computed by the corresponding
                         observable object
    std_list : list of numpy arrays
               Each numpy array can be 0d or 1d and represents experimental
               errors, that correcpond observables

    num_of_observable_objects : int
                              Length of observable_object list.
                              Should be equal to length of std_list and
                              length of observed_value_list


    Methods:

    Class shoud hold all the methods
    """

    def __init__(self):
        """
        Initializaiton of the likelihood function
        """
        self.observed_value_list = []
        self.std_list = []
        self.observable_object_list = []
        self.num_of_observable_objects = 0

    def add_observable(self,observable_object,observed_values,error,scale=1.0):
        """
        The method fills all the fields for a particular observable ddG.

        Parameters
        ----------
        observable_object : object
                          Observable object, that is able to calculate
                          observable and their derivatives
        observed_values : float, list of floats or 1d-numpy array,
                           experimental values, that correspond to ones
                           calculated by observable object
        error : float, list of floats or 1d-numpy array.
                            experimental errors, that correspond to experimental
                            values in observed_values
        """
        type_consistent_observed_value = np.atleast_1d(np.array(observed_values))
        self.observed_value_list.append(type_consistent_observed_value)

        errors = np.atleast_1d(np.array(error)*scale)
        self.std_list.append(errors)

        self.observable_object_list.append(observable_object)
        self.num_of_observable_objects += 1
        assert self.num_of_observable_objects == len(self.observed_value_list)
        assert self.num_of_observable_objects == len(self.std_list)
        assert self.num_of_observable_objects == len(self.observable_object_list)
        return

    def _compute_z_score(self,calculated_value,std,observed_value):
        """
        Compute z-score

        Parameters
        -----------
        calculated_value : numpy ndarray, dtype=float
                           A sample value
        std              : numpy ndarray,  dtype=float
                          Standard deviation
        observed_value : numpy ndarray, dtype=float
                         Expectation value
        Demensionality of all the input parameters should be the same
        Returns
        --------
        z-score : numpy ndarray
        Z-score, dimensionality the same as for input arrays

        """
        z_score = np.divide(np_subtract(calculated_value,observed_value),std)
        return z_score

    def compute_ln_gauss(self,epsilons,grad_parameters=None):
        """

        Method copmutes -logQ and d(-logQ)/d(epsilon).
        For calculations, it is expected that all the values are numpy arrays

        Parameters
        ----------
        epsilons : 1D  numpy array of floats
         set of model parameters
        grad_parameters
        numpy array of integer indexes. For which derivative should
        be computed.
        """
        negative_lnQ = 0
        if grad_parameters is None:
            derivative_negative_lnQ = np.zeros(epsilons.shape[0])
        else:
            derivative_negative_lnQ = np.zeros(grad_parameters.shape[0])

        for observable_ndx in range(self.num_of_observable_objects):
            observed_values = self.observed_value_list[observable_ndx]
            std = self.std_list[observable_ndx]
            calculated_values, calculated_derivatives = self.observable_object_list[observable_ndx].compute_observation(epsilons)
            assert len(observed_values) == len(calculated_values), "Length of experimental and calculated observable values do not match"
            z_score = self.compute_z_score(calculated_values,std,observed_values)
            negative_lnQ +=  0.5*np.sum(np.square(z_score))
            derivative_negative_lnQ +=  np.multiply(np.divide(z_score,std),derivative)
        return negative_lnQ, derivative_negative_lnQ
