""" Contains EstimatorsObject """

import numpy as np

class EstimatorsObject(object):
    """ Contains inputted formatted data and results from analysis 
    
    This object contains the necessary data formatted appropriately, as
    well as the resultant Q functions and the results of any 
    optimization routine.
    
    """
    def __init__(self, data, data_sets, observables, model, obs_data=None):
        """ initialize object and process all the inputted data 
        
        Args:
            data (array): First index is the frame, the other indices are 
                the data for the frame.
            data_sets (list): Each entry is an array with the frames 
                corresponding to that equilibrium state.      
            observables (ExperimentalObservables): See object in the 
                module <pyfexd/observables>.
            model (ModelLoader): See object in the module
                 <pyfexd/model_loaders>.
                
        """
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
        
        ##Compute factors that don't depend on the re-weighting
        self.state_prefactors = []
        for i in range(self.number_equilibrium_states):
            state_prefactor = self.pi[i] * self.expectation_observables[i]   
            self.state_prefactors.append(state_prefactor)
    
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
        
    def Qfunction_epsilon(self, epsilons):
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
        
        #Minimization, so make maximal value a minimal value with a negative sign.
        Q = -1.0 * self.Q_function(next_observed) 

        return Q
        
    def log_Qfunction_epsilon(self, epsilons):
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
        
        #Minimization, so make maximal value a minimal value with a negative sign.
        Q = self.log_Q_function(next_observed) 
        #print epsilons
        return Q

    def derivatives_Qfunction_epsilon(self, epsilons):
        #initialize final matrices
        next_observed = np.zeros(self.num_observable)
        derivative_observed_first = [np.zeros(self.num_observable) for j in range(self.number_params)]
        derivative_observed_second = [np.zeros(self.num_observable) for j in range(self.number_params)]
        dQ_vector = []
        #add up all re-weighted terms for normalizaiton
        total_weight = 0.0
        
        #Calculate the reweighting for all terms: Get Q, and next_observed and derivative terms
        for i in range(self.number_equilibrium_states):
            #terms for Q
            boltzman_weights = np.exp(self.epsilons_functions[i](epsilons) - self.h0[i])
            next_weight = np.sum(boltzman_weights) / self.ni[i]
            next_observed += next_weight * self.state_prefactors[i]
            total_weight += next_weight * self.pi[i]
            #terms for dQ/de
            for j in range(self.number_params):
                next_weight_derivatives = np.sum(boltzman_weights * self.derivatives_functions[i](epsilons)[j]) / self.ni[i]
                derivative_observed_first[j] += next_weight_derivatives * self.state_prefactors[i]
                derivative_observed_second[j] += next_weight_derivatives * self.pi[i]
                
        #normalize so total re-weighted probability is = 1.0
        next_observed /= total_weight
        
        #Minimization, so make maximal value a minimal value with a negative sign.
        Q = self.Q_function(next_observed)
        
        for j in range(self.number_params):
            derivative_observed = (derivative_observed_first[j]  - (next_observed * derivative_observed_second[j])) / total_weight 
            dQ = self.dQ_function(next_observed, derivative_observed) * Q
            dQ_vector.append(dQ)

        
        return Q, np.array(dQ_vector)

    def derivatives_log_Qfunction_epsilon(self, epsilons):
        #initialize final matrices
        next_observed = np.zeros(self.num_observable)
        derivative_observed_first = [np.zeros(self.num_observable) for j in range(self.number_params)]
        derivative_observed_second = [np.zeros(self.num_observable) for j in range(self.number_params)]
        dQ_vector = []
        #add up all re-weighted terms for normalizaiton
        total_weight = 0.0
        
        #Calculate the reweighting for all terms: Get Q, and next_observed and derivative terms
        for i in range(self.number_equilibrium_states):
            #terms for Q
            boltzman_weights = np.exp(self.epsilons_functions[i](epsilons) - self.h0[i])
            next_weight = np.sum(boltzman_weights) / self.ni[i]
            next_observed += next_weight * self.state_prefactors[i]
            total_weight += next_weight * self.pi[i]
            #terms for dQ/de
            for j in range(self.number_params):
                next_weight_derivatives = np.sum(boltzman_weights * self.derivatives_functions[i](epsilons)[j]) / self.ni[i]
                derivative_observed_first[j] += next_weight_derivatives * self.state_prefactors[i]
                derivative_observed_second[j] += next_weight_derivatives * self.pi[i]
                
        #normalize so total re-weighted probability is = 1.0
        next_observed /= total_weight
        
        #Minimization, so make maximal value a minimal value with a negative sign.
        Q = self.log_Q_function(next_observed)
        
        for j in range(self.number_params):
            derivative_observed = (derivative_observed_first[j]  - (next_observed * derivative_observed_second[j])) / total_weight 
            dQ = self.dlog_Q_function(next_observed, derivative_observed) 
            dQ_vector.append(dQ)

        #print epsilons
        return Q, np.array(dQ_vector)
    
    
        
