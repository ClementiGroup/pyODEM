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
        self.ddG_observed_values = []
        self.ddG_std = []
        self.ddG_observables = []
        self.num_of_obs = 0

    def add_ddG(self,observable,observed_value,error,scale=1.0):
        """
        The method fills all the fields for a particular observable ddG.

        Parameters
        ----------

        fraction : 1d numpy.ndarray of floats
                    Contains fraction of contacts, that remains after mutation

        """

        self.ddG_observed_values.append(observed_value)
        self.ddG_std.append(error*scale)
        self.ddG_observables.append(observable)
        self.num_of_obs += 1

    @property
    def num_of_ddG(self):
        return self.num_of_obs

    def _compute_z_score(self,calculated_value,std,observed_value):
        return (calculated_value-observed_value)/std

    def compute_log_Q(self,epsilons,data_points=None):
        """
        Method copmutes -logQ and d(-logQ)/d(epsilon).
        At this point, only logarithmic function is implemented
        Parameters
        ----------
        epsilons : 1D  numpy array of floats
         set of model parameters
        data_points : 1D numpy array of ints
        numpy array of indexes  Each index corresponds
        """
        negative_lnQ = 0
        derivative_negative_lnQ = np.zeros(epsilons.shape[0])
        if data_points is None:
            data_points = [i for i in range(self.num_of_obs)]
        for observable_ndx in range(self.num_of_obs):
            if observable_ndx in data_points:
                observed_value = self.ddG_observed_values[observable_ndx]
                #print("observable: {}".format(observable_ndx))
                std = self.ddG_std[observable_ndx]
                value, derivative = self.ddG_observables[observable_ndx].compute_delta_delta_G(epsilons,compute_derivative=True)
                #print("observable_value: {}".format(value))
                z_score = self._compute_z_score(value,std, observed_value)
                #print("z_score: {}".format(z_score))
                negative_lnQ +=  0.5*(z_score)**2
                derivative_negative_lnQ +=  np.multiply(z_score/std,derivative)
        return negative_lnQ, derivative_negative_lnQ
