""" Contains the object ExperimentalObservables
Meant to hold the classes and objects for collecting the observables and 
computing collected features. This is where the Q values and derivatives 
should be calculated for later use in the package.

For purposes of this module: 
observable object refers to the class of objects.
observables refers to the actual observations each object makes.

"""


import numpy as np

from pyfexd.observables.hist_analysis_pkg import HistogramO
import pyfexd.basic_functions as bf



class ExperimentalObservables(object):
    """ Object for holding observables and computing Q values
    
    Specific observables are added by calling the associated Methods. 
    This will set up the necessary lists of functions.
    
    Attributes:
        observables(list): List of ObservableObjects. Used for 
            computing the observables on a trajectory.
        q_functions(list): List of q_function associated with each value 
            of all the observables.
        dq_functions(list): List of derivatives of each q_function.
        log_funtions(list): Logarithm of each q_function
        dlog_functions(list): derivative of the log_functions
        num_q_functions(float): Number of q_functions added thus far.
        obs_seen(list): List of bool. True for observables that were 
            seen in the simulation. Used internally in other methods.
    
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
        
    def add_histogram(self, exp_file, compute=False, nbins=None, histrange=None, spacing=None, edges=None, errortype="gaussian"):
        """ Adds a Histogram observable from HistogramO
        
        Method will add appropriate objects to the list and generate the 
        necessary functions for a histogram observable such as FRET pair 
        distance or position distribution.
        
        Args:
            exp_file(str): Properly formatted data file. Either in two 
                columns, representing value and std, or a trace of 
                numbers to be histogrammed.
            compute(bool): True if the data file is a trace of numbers 
                to histogram. Default to False.
            nbins(int): Specify number of bins for histogramming.
            histrange(tuple): Range to histogram over.
            spacing(float): Spacing to histogram the data over.
            edges(array): 1-D column of edges for histogram bins.
            errortype(str): Type of error to use for the observables.
            
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
	        std = exp_data[i,1]
	        if std == 0:
	            std = 1.0 #std of zero could lead to divide by zero exception. Set to 1 if it's zero (due to no sampling) 
	        self.q_functions.append(bf.statistical.wrapped_gaussian(mean,std))
	        self.dq_functions.append(bf.statistical.wrapped_derivative_gaussian(mean,std))
	        self.log_functions.append(bf.log_statistical.wrapped_harmonic(mean,std))
	        self.dlog_functions.append(bf.log_statistical.wrapped_derivative_harmonic(mean,std))
	    
	    #Default: all observables are by default seen.     
        self.prep_True()
	    
    
    def prep(self):
        """ Sets obs_seen to False for all observables. """

        self.obs_seen = [False for i in range(self.num_q_functions)]
        
    def prep_True(self):
        """ Sets obs_seen to True for all observables. """
        
        self.obs_seen = [True for i in range(self.num_q_functions)]

    def compute_observations(self, data, weights=None):
        """ Compute the observables from a data set. 
        
        Assumes the data set is formulated correctly for the observables 
        to interpret.
        
        Args:
            data(array): Typically array of floats. First index frames, 
                second index and above are for coordiantes.
            weights(array): Weight values for each frame. 1-D array, 
                same size as first dimension of data.
        
        Returns:
            all_obs(list): Value of observables for every 
                ObservableObject.  
            all_std(list): Value of std for each observable in all_obs.        
        """
        if weights == None:
            weights = np.ones(np.shape(data)[0])
        
        all_obs = np.array([])
        all_std = np.array([])
        for observable in self.observables:
            obs, std, seen = observable.compute_observed(data, weights)
            all_obs = np.append(all_obs, obs)
            all_std = np.append(all_std, std)
            self.obs_seen = [seent or nott for seent, nott in zip(seen, self.obs_seen)] 
        return all_obs, all_std
    
    def get_q_functions(self):
        """ Returns the q_function and the derivative of the q_function
        
        Returns:
            q_simple(method): Computes q value based on some observabled.
            dq_simple(method): Computes derivative of q_values.
        
        """
        def q_simple(observations):
            if not np.shape(observations)[0] == self.num_q_functions:
                #check to see if number of observations match
                #NOTE: Depending on how values are arranged and passed in this wrapper  function, we can add q_functions later in a script and cause this to fail when trying ot use the same "old" function.
                raise IOError("Number of observations not equal to number of q_functions. This is a problem")
            q_value = 1.0
            for i in range(self.num_q_functions):
                if self.obs_seen[i]:
                    q_value *= self.q_functions[i](observations[i])
            
            return q_value
        
        def dq_simple(observations, derivative_observed):
            if not np.shape(observations)[0] == self.num_q_functions:
                #check to see if number of observations match
                #NOTE: Depending on how values are arranged and passed in this wrapper  function, we can add q_functions later in a script and cause this to fail when trying ot use the same "old" function.
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
            q_simple(method): Computes logarithm of q.
            dq_simple(method): Computes derivative of log(q).
        
        """
        def q_simple(observations):
            if not np.shape(observations)[0] == self.num_q_functions:
                #check to see if number of observations match
                #NOTE: Depending on how values are arranged and passed in this wrapper  function, we can add q_functions later in a script and cause this to fail when trying ot use the same "old" function.
                raise IOError("Number of observations not equal to number of q_functions. This is a problem")
            q_value = 0.0
            for i in range(self.num_q_functions):
                if self.obs_seen[i]:
                    q_value += self.log_functions[i](observations[i])
            
            return q_value
        
        def dq_simple(observations, derivative_observed):
            if not np.shape(observations)[0] == self.num_q_functions:
                #check to see if number of observations match
                #NOTE: Depending on how values are arranged and passed in this wrapper  function, we can add q_functions later in a script and cause this to fail when trying ot use the same "old" function.
                raise IOError("Number of observations not equal to number of q_functions. This is a problem")
            dq_value = 0.0
            for i in range(self.num_q_functions):
                if self.obs_seen[i]:
                    dq_value += self.dlog_functions[i](observations[i]) * derivative_observed[i]
                    
            return dq_value
    
        return q_simple, dq_simple
            
        
        
