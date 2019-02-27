""" Contains the object ExperimentalObservables
Meant to hold the classes and objects for collecting the observables and
computing collected features. This is where the Q values and derivatives
should be calculated for later use in the package.

For purposes of this module:
observable object refers to the class of objects.
observables refers to the actual observations each object makes.

"""


import numpy as np
from mpi4py import MPI


from pyODEM.observables.hist_analysis_pkg import HistogramO
import pyODEM.basic_functions as bf



class ExperimentalObservables(object):
    """ Object for holding observables and computing Q values

    Specific observables are added by calling the associated Methods. This will
    set up the necessary lists of functions.

    Attributes:
        observables (list of ObservableObjects): Used for computing the
            observables on a trajectory.
        q_functions (list of method): Functions compute Q value for each
            observable.
        dq_functions (list of method): Derivatives of q_functions.
        log_functions (list of method): Logarithm of each q_function
        dlog_functions (list of method): Derivative of the log_functions
        num_q_functions (float): Total number of q_functions.
        obs_seen (list of bool): True for q_functions to use when computing the
            Q value. Used internally to avoid unnecessary computations during
            the optimization procedure. Defaults to using all q_functions when
            computing during the optimization procedure. Defaults to

    Example:
        obs = ExperimentalObservables()
        obs.add_histogram("exp_data.dat")

    """
    def __init__(self):
        self.observables = []
        self.q_functions = []
        self.dq_functions = []
        self.log_functions = []
        self.dlog_functions = []
        self.num_q_functions = 0

    def add_average(self, obs_value, obs_std, errortype="gaussian", scale=1.0):

        """ Adds a single_value observable.

        Method will add appropriate objects to the list and generate the
        necessary functions for an observable, with is represented by a single value for the
        entire trajectory (like a spin-coupling constant for a particular pair of atoms)

        Args:
            obs_value (1d numpy array): Values of the observable.

            obs_std (1d numpy array): Corresponding standard deviation

            errortype (str): Type of error to use for the observables.

            scale (float): Scale the error values by this factor. Important for
                numerical efficiency. Defaults to 1.0 (no scaling).
        """
        observable = AverageO()
        self.observables.append(observable)

        #add the necessary Q functions.
        for i in range(np.shape(obs_value)[0]):
	        self.num_q_functions += 1
	        mean = obs_value[i]
	        std = obs_std[i]*scale
	        if std == 0:
	            std = 1.0 #std of zero could lead to divide by zero exception. Set to 1 if it's zero (due to no sampling)
	        self.q_functions.append(bf.statistical.wrapped_gaussian(mean,std))
	        self.dq_functions.append(bf.statistical.wrapped_derivative_gaussian(mean,std))
	        self.log_functions.append(bf.log_statistical.wrapped_harmonic(mean,std))
	        self.dlog_functions.append(bf.log_statistical.wrapped_derivative_harmonic(mean,std))

        #Default: all observables are by default seen.
        self.prep_True()
        assert len(self.obs_seen) == self.num_q_functions
        assert len(self.q_functions) == self.num_q_functions
        assert len(self.dq_functions) == self.num_q_functions
        assert len(self.log_functions) == self.num_q_functions
        assert len(self.dlog_functions) == self.num_q_functions


    def add_histogram(self, exp_file, compute=False, nbins=None, histrange=None, spacing=None, edges=None, errortype="gaussian", scale=1.0):
        """ Adds a Histogram observable from HistogramO

        Method will add appropriate objects to the list and generate the
        necessary functions for a histogram observable such as FRET pair
        distance or position distribution. Can either give the histogram results
        and edges, or a trace of data and parameters for histogramming the data.


        Args:
            exp_file (str): Data file name. Formatted as either 1) two columns,
                representing value and std, or 2) a trace of numbers to be
                histogrammed.
            compute (bool): True if the data file is a trace of numbers to
                histogram. Defaults to False.
            nbins (int): Number of bins if data file is a trace.
            histrange(tuple): Range to histogram over if data file is a trace.
            spacing (float): Spacing to histogram the data over if data file is
                a trace.
            edges (array): 1-D column of edges for histogram bins.
            errortype (str): Type of error to use for the observables.
            scale (float): Scale the error values by this factor. Important for
                numerical efficiency. Defaults to 1.0 (no scaling).

            Histogram parameters default to None.
            Must give a valid set of parameters:
            1. nbins, histrange
            2. spacing, histrange
            3. edges

        """
        #add observable object if compute is False.
        #Otherwise, compute the actual observed values.
        if compute == False:
            observable = HistogramO(nbins, histrange, spacing, edges)
            self.observables.append(observable)
        else:
        	pass #implement a method for analyzing a data file based on the parameters for the histogram given here.

        exp_data = np.loadtxt(exp_file)

        #add the necessary Q functions.
        for i in range(np.shape(exp_data)[0]):
            self.num_q_functions += 1
            mean = exp_data[i,0]
            std = exp_data[i,1]*scale
            if std == 0:
                std = 1.0 #std of zero could lead to divide by zero exception. Set to 1 if it's zero (due to no sampling)
            self.q_functions.append(bf.statistical.wrapped_gaussian(mean,std))
            self.dq_functions.append(bf.statistical.wrapped_derivative_gaussian(mean,std))
            self.log_functions.append(bf.log_statistical.wrapped_harmonic(mean,std))
            self.dlog_functions.append(bf.log_statistical.wrapped_derivative_harmonic(mean,std))

        #Default: all observables are by default seen.
        self.prep_True()
        assert len(self.obs_seen) == self.num_q_functions
        assert len(self.q_functions) == self.num_q_functions
        assert len(self.dq_functions) == self.num_q_functions
        assert len(self.log_functions) == self.num_q_functions
        assert len(self.dlog_functions) == self.num_q_functions

    def prep(self):
        """ Sets obs_seen to False for all observables. """

        self.obs_seen = [False for i in range(self.num_q_functions)]

    def prep_True(self):
        """ Sets obs_seen to True for all observables. """

        self.obs_seen = [True for i in range(self.num_q_functions)]

    def synchronize_obs_seen(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        this_seen = [False for i in range(self.num_q_functions)]
        if rank == 0:
            for i in range(self.num_q_functions):
                if self.obs_seen[i]:
                    this_seen[i] = True
            for i_thread in range(1, size):
                that_seen = comm.recv(source=i_thread, tag=3)
                for i in range(self.num_q_functions):
                    if that_seen[i]:
                        this_seen[i] = True
        else:
            comm.send(self.obs_seen, dest=0, tag=3)
            this_seen = None
        this_seen = comm.bcast(this_seen, root=0)
        self.obs_seen = this_seen

    def compute_observations(self, data, weights=None, all=False):
        """ Compute the observables from a data set.

        Assumes the data set is formulated correctly for the observables to
        interpret.

        Args:
            data (list of arrays): List of data sets. For each entry, the first
                index corresponds to the frames and the second index and above
                are for coordinates.
            weights (array of float): Weight values for each frame. 1-D array,
                same size as first dimension of data.

        Returns:
            all_obs (list of float): Value of observables for every
                ObservableObject.
            all_std (list of float): Value of std for each value in all_obs.
        """
        if weights == None:
            weights = np.ones(np.shape(data[0])[0])

        all_obs = np.array([])
        all_std = np.array([])
        seen_this_time = []
        for idx,observable in enumerate(self.observables):
            obs, std, seen = observable.compute_observed(data[idx], weights)
            all_obs = np.append(all_obs, obs)
            all_std = np.append(all_std, std)
            seen_this_observable = seen
            if all:
                for TorF in seen_this_observable:
                    seen_this_time.append(True)
            else:
                for TorF in seen_this_observable:
                    seen_this_time.append(TorF)
        self.obs_seen = [old_seen or new_seen for old_seen, new_seen in zip(self.obs_seen,seen_this_time)]

        assert len(self.obs_seen) == self.num_q_functions
        return all_obs, all_std

    def get_q_functions(self):
        """ Returns the q_function and the derivative of the q_function

        Returns:
            q_simple (method): Computes q value based on some observabled.
            dq_simple (method): Computes derivative of q_values.

        """
        def q_simple(observations):
            if not np.shape(observations)[0] == self.num_q_functions:
                #check to see if number of observations match
                raise IOError("Number of observations not equal to number of q_functions. This is a problem")
            q_value = 1.0
            for i in range(self.num_q_functions):
                if self.obs_seen[i]:
                    q_value *= self.q_functions[i](observations[i])

            return q_value

        def dq_simple(observations, derivative_observed):
            if not np.shape(observations)[0] == self.num_q_functions:
                #check to see if number of observations match
                raise IOError("Number of observations not equal to number of q_functions. This is a problem")
            dq_value = 0.0
            for i in range(self.num_q_functions):
                if self.obs_seen[i]:
                    dq_value += (self.dq_functions[i](observations[i]) * derivative_observed[i]) / self.q_functions[i](observations[i])

            return dq_value

        return q_simple, dq_simple

    def get_log_q_functions(self):
        """ Returns the logarithm of q_function and its derivative

        Returns:
            q_simple (method): Computes logarithm of q.
            dq_simple (method): Computes derivative of log(q).

        """
        def q_simple(observations):
            if not np.shape(observations)[0] == self.num_q_functions:
                #check to see if number of observations match
                raise IOError("Number of observations not equal to number of q_functions. This is a problem")
            q_value = 0.0
            for i in range(self.num_q_functions):
                if self.obs_seen[i]:
                    q_value += self.log_functions[i](observations[i])

            return q_value

        def dq_simple(observations, derivative_observed):
            if not np.shape(observations)[0] == self.num_q_functions:
                #check to see if number of observations match
                raise IOError("Number of observations not equal to number of q_functions. This is a problem")
            dq_value = 0.0
            for i in range(self.num_q_functions):
                if self.obs_seen[i]:
                    dq_value += self.dlog_functions[i](observations[i]) * derivative_observed[i]

            return dq_value

        return q_simple, dq_simple
