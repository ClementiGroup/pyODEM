""" Contains EstimatorsObject """

import numpy as np
import time

np.seterr(over="raise")

class EstimatorsObject(object):
    """ Contains inputted formatted data and results from analysis 
    
    This object contains the necessary data formatted appropriately, as
    well as the resultant Q functions and the results of any 
    optimization routine.
    
    """
    def __init__(self, data, data_sets, observables, model, obs_data=None, stationary_distributions=None, K_shift=0, K_shift_step=100, Max_Count=10):
        """ initialize object and process all the inputted data 
        
        Args:
            data (array): First index is the frame, the other indices 
                are the data for the frame. Should be the data loaded 
                from model.load_data()
            data_sets (list): Each entry is an array with the frames 
                corresponding to that equilibrium state.      
            observables (ExperimentalObservables): See object in
                pyfexd.observables.exp_observables.ExperimentalObservables
            model (ModelLoader): See object in the module
                 pyfexd.model_loaders.X for the particular model.
            obs_data (list): Use if data set for computing observables 
                is different from data for computing the energy. List 
                contains arrays where each array-entry corresponds to 
                the observable in the ExperimentalObservables object. 
                Arrays are specified with first index corresponding to 
                the frame and second index to the data. Default: Use the 
                array specified in data for all observables.
            stationary_distributions (list): List of values for pi for 
                each stationary distribution. Must be same size as 
                data_sets. Default will compute the distribution based 
                upon the weighting of each data_sets. 
            K_shift (float): Value to shift exponents by. Default 0.
            K_shift_step (float): Value to increase the K_shift by in 
                event exponential evaluation fails. Default 100.
            Max_Count (int): Number of attempts of automatic re-scaling 
                to attempt before the method will give up and raise an 
                error.    
        """
        print "Initializing EstimatorsObject"
        t1 = time.time()
        #set k-shift parameters
        self.K_shift = K_shift
        self.K_shift_step = K_shift_step
        self.Max_Count = Max_Count
        #observables get useful stuff like value of beta
        self.beta = model.beta
        self.number_equilibrium_states = len(data_sets)
        self.observables = observables
        self.model = model
        
        self.observables.prep()
        
        self.current_epsilons = model.get_epsilons()
        self.number_params = np.shape(self.current_epsilons)[0]
        self.Q_function, self.dQ_function = observables.get_q_functions()
        self.log_Q_function, self.dlog_Q_function = observables.get_log_q_functions()
        
        #calculate average value of all observables and associated functions 
        self.expectation_observables = []
        self.epsilons_functions = []
        self.derivatives_functions = []
        
        
        self.h0 = []
        
        self.pi = []
        self.ni = []
        
        if obs_data is None: #use sim data for calculating observables
            obs_data = [data for i in range(len(self.observables.observables))]
        else:
            pass
            
        #load data for each set, and compute energies and observations
        for i in data_sets:
            use_data = data[i]
            epsilons_function, derivatives_function = model.get_potentials_epsilon(use_data)
            #process obs_data so that only the desired frames are passed
            use_obs_data = []
            for obs_dat in obs_data:
                use_obs_data.append(obs_dat[i])
            observed, obs_std = observables.compute_observations(use_obs_data)
            num_in_set = np.shape(use_data)[0]
            
            self.expectation_observables.append(observed)
            self.epsilons_functions.append(epsilons_function)
            self.derivatives_functions.append(derivatives_function)
            
            self.h0.append(epsilons_function(self.current_epsilons))
            
            self.ni.append(num_in_set)
            self.pi.append(num_in_set)
            
        ##number of observables
        self.num_observable = np.shape(observed)[0]   
        self.pi =  np.array(self.pi).astype(float)
        self.pi /= np.sum(self.pi)
        
        if not stationary_distributions is None:
            print "Using Inputted Stationary Distribution"
            if np.shape(stationary_distributions)[0] == len(self.ni):
                print "Percent Difference of Selected Stationary Distribution from expected"
                diff = self.pi - stationary_distributions
                print np.abs(diff/self.pi)
                self.pi = stationary_distributions
            else:
                raise IOError("Inputted stationary distributions does not number of equilibrium states.")
        
        ##Compute factors that don't depend on the re-weighting
        self.state_prefactors = []
        for i in range(self.number_equilibrium_states):
            state_prefactor = self.pi[i] * self.expectation_observables[i]   
            self.state_prefactors.append(state_prefactor)
        
        t2 = time.time()
        total_time = (t2-t1) / 60.0
        print "Initializaiton Completed: %f Minutes" % total_time 
    def get_reweighted_observable_function(self):
        return self.calculate_observables_reweighted
        
    def calculate_observables_reweighted(self, epsilons):
        """ Calculates the observables using a set of epsilons
        
        Takes as input a new set of epsilons (model parameters). 
        Calculates a new set of observables using self.observables and 
        outputs the observables as an array.
        
        Args:
            epsilons (array): Model parameters
        
        Returns:
            next_observed (array): Values for all the observables.
        
        """
        
        #initiate value for observables:
        next_observed = np.zeros(self.num_observable)
        

        #add up all re-weighted terms for normalizaiton
        total_weight = 0.0
        #calculate re-weighting for all terms 
        for i in range(self.number_equilibrium_states):
            next_weight = np.sum(np.exp(self.epsilons_functions[i](epsilons) - self.h0[i])) / self.ni[i]
            next_observed += next_weight * self.state_prefactors[i]
            total_weight += next_weight * self.pi[i]
        
        #normalize so total re-weighted probability is = 1.0
        next_observed /= total_weight
        
        return next_observed
    
    def save_solutions(self, new_epsilons):
        """ Save the solutions and store the new and old Q """
        
        self.new_epsilons = new_epsilons
        self.oldQ = self.Qfunction_epsilon(self.current_epsilons)
        self.newQ = self.Qfunction_epsilon(self.new_epsilons)
        self.old_epsilons = self.current_epsilons
           
    def get_function(self, derivatives, logq):
        """ Returns the necessary function for later use """
        
        if derivatives:
            if logq:
                func = self.derivatives_log_Qfunction_epsilon
            else:
                func = self.derivatives_Qfunction_epsilon
        else:
            if logq:
                func = self.log_Qfunction_epsilon
            else:
                func = self.Qfunction_epsilon
            
        return func
        
    def Qfunction_epsilon(self, epsilons, K_shift=None, Count=0):
        #initiate value for observables:
        
        next_observed, total_weight, boltzman_weights = self.get_reweights(epsilons)
        
        #Minimization, so make maximal value a minimal value with a negative sign.
        Q = -1.0 * self.Q_function(next_observed) 

        return Q
        
    def log_Qfunction_epsilon(self, epsilons, K_shift=None, Count=0):
        next_observed, total_weight, boltzman_weights = self.get_reweights(epsilons)
        
        #Minimization, so make maximal value a minimal value with a negative sign.
        Q = self.log_Q_function(next_observed) 
        #print epsilons
        return Q

    def derivatives_Qfunction_epsilon(self, epsilons, K_shift=None, Count=0):
        next_observed, total_weight, boltzman_weights = self.get_reweights(epsilons)
        
        Q = self.Q_function(next_observed)
        derivative_observed_first, derivative_observed_second = self.get_derivative_pieces(epsilons, boltzman_weights)
        
        dQ_vector = []
        for j in range(self.number_params):
            derivative_observed = (derivative_observed_first[j]  - (next_observed * derivative_observed_second[j])) / total_weight 
            dQ = self.dQ_function(next_observed, derivative_observed) * Q
            dQ_vector.append(dQ)
            
        dQ_vector = -1. * np.array(dQ_vector)
        Q *= -1.
        return Q, dQ_vector

    def derivatives_log_Qfunction_epsilon(self, epsilons, K_shift=None, Count=0):
        next_observed, total_weight, boltzman_weights = self.get_reweights(epsilons)
        
        Q = self.log_Q_function(next_observed)
        derivative_observed_first, derivative_observed_second = self.get_derivative_pieces(epsilons, boltzman_weights)
        
        dQ_vector = []
        for j in range(self.number_params):
            derivative_observed = (derivative_observed_first[j]  - (next_observed * derivative_observed_second[j])) / total_weight 
            dQ = self.dlog_Q_function(next_observed, derivative_observed) 
            dQ_vector.append(dQ)
        
        dQ_vector = np.array(dQ_vector)
        
        return Q, dQ_vector
    
    def get_derivative_pieces(self, epsilons, boltzman_weights):
        derivative_observed_first = [np.zeros(self.num_observable) for j in range(self.number_params)]
        derivative_observed_second = [np.zeros(self.num_observable) for j in range(self.number_params)]
        for i in range(self.number_equilibrium_states):
            for j in range(self.number_params):
                next_weight_derivatives = np.sum(boltzman_weights[i] * self.derivatives_functions[i](epsilons)[j]) / self.ni[i]
                derivative_observed_first[j] += next_weight_derivatives * self.state_prefactors[i]
                derivative_observed_second[j] += next_weight_derivatives * self.pi[i]
        
        return derivative_observed_first, derivative_observed_second
        
    def get_reweights(self, epsilons):
        #initialize final matrices
        next_observed = np.zeros(self.num_observable)
        
        #add up all re-weighted terms for normalizaiton
        total_weight = 0.0
        
        #Calculate the reweighting for all terms
        boltzman_weights = self.get_boltzman_weights(epsilons)

        for i in range(self.number_equilibrium_states):
            next_weight = np.sum(boltzman_weights[i]) / self.ni[i]
            next_observed += next_weight * self.state_prefactors[i]
            total_weight += next_weight * self.pi[i]
        next_observed /= total_weight
        
        return next_observed, total_weight, boltzman_weights
        
    def get_boltzman_weights(self, epsilons):
        #calculate the boltzman weights. 
        #If OverflowExcpetion, recompute with shift to all values
        #shift is simply -max_val+K_Shift
        K_shift = 700
        try:
            boltzman_weights = []
            for i in range(self.number_equilibrium_states):
                boltzman_wt = np.exp(self.epsilons_functions[i](epsilons) - self.h0[i])
                boltzman_weights.append(boltzman_wt)
        except:
            print "Exponential Function Failed"
            exponents = []
            boltzman_weights = []
            max_val = 0
            for i in range(self.number_equilibrium_states):
                exponent = self.epsilons_functions[i](epsilons) - self.h0[i]
                max_exponent = np.max(exponent)
                if max_exponent > max_val:
                    max_val = max_exponent
                exponents.append(exponent)
            for i in range(self.number_equilibrium_states):
                boltzman_wt = np.exp(exponents[i] - max_val + K_shift)
                boltzman_weights.append(boltmzan_wt)
        
        assert len(boltzman_weights) == self.number_equilibrium_states
        return boltzman_weights 
        
def save_error_files(diff, epsilons, h0, kshift):
    np.savetxt("ERROR_EXPONENTIAL_FUNCTION", diff)
    np.savetxt("ERROR_EPSILONS", epsilons)
    np.savetxt("ERROR_H0", h0)
    f = open("ERROR_LOG_EXPONENTIAL_FUNCTION", "w")
    f.write("\n\nCurrent Shift: %f\n" % kshift)
    f.close()    


