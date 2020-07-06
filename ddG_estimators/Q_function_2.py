"""
The package contains classes needed for manipulation with delta_delta_G variable.
At this point, it is completely separated from classes, that already exist in pyODEM,
though it uses some of the helper functions.

"""
import numpy as np

class Q_function():
    """
    Class collects all the observables and create Q function and corresponding
    derivatives.
    """

    def __init__(self):
        """
        Initializaiton of Q_function
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

        fraction : 1d numpy.ndarray of floats
                    Contains fraction of contacts, that remains after mutation

        """

        self.observed_value_list.append(np.array(observed_values))
        self.std_list.append(np.array(error)*scale)
        self.observable_object_list.append(observable)
        self.num_of_observable_objects += 1

    def _compute_z_score(self,calculated_value,std,observed_value):
        """
        Compute z-score for a massive
        """
        return (calculated_value-observed_value)/std

    def compute_log_Q(self,epsilons,grad_parameters=None):
        """
        Method copmutes -logQ and d(-logQ)/d(epsilon).
        At this point, only logarithmic function is implemented
        Parameters
        ----------
        epsilons : 1D  numpy array of floats
         set of model parameters
        grad_parameters
        numpy array of integer indexes. Each index
        """
        negative_lnQ = 0
        if grad_parameters is None:
            derivative_negative_lnQ = np.zeros(epsilons.shape[0])
        else:
             derivative_negative_lnQ = np.zeros(grad_parameters.shape[0])

        for observable_ndx in range(self.num_of_observable_objects):
            observed_values = self.observed_value_list[observable_ndx]
            std = self.std_list[observable_ndx]
            value, derivative = self.ddG_observables[observable_ndx].compute_delta_delta_G(epsilons,compute_derivative=True,grad_parameters=grad_parameters)
            z_score = self._compute_z_score(value,std, observed_value)
            negative_lnQ +=  0.5*(z_score)**2
            derivative_negative_lnQ +=  np.multiply(z_score/std,derivative)
        return negative_lnQ, derivative_negative_lnQ
