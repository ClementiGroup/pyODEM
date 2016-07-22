""" Contains EstimatorsObject """

import numpy as np
import time
import mdtraj as md

np.seterr(over="raise")

class EstimatorsObject(object):
    """ Contains inputted formatted data and results from analysis 
    
    This object contains the necessary data formatted appropriately, as
    well as the resultant Q functions and the results of any 
    optimization routine.
    
    """
    def __init__(self, data, data_sets, observables, model, obs_data=None, stationary_distributions=None, model_state=None):
        """ initialize object and process all the inputted data 
        
        Args:
            data (array): First index is the frame, the other indices 
                are the data for the frame. Should be the data loaded 
                from model.load_data()
            data_sets (list): Each entry is an array with the frames 
                corresponding to that equilibrium state.      
            observables (ExperimentalObservables): See object in
                pyfexd.observables.exp_observables.ExperimentalObservables
            model (ModelLoader/list): See object in the module
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
            model_state (list): List which model object to use when 
                model is a list. Default None.
        """
        print "Initializing EstimatorsObject"
        t1 = time.time()
        #observables get useful stuff like value of beta
        self.number_equilibrium_states = len(data_sets)
        self.observables = observables

        
        self.observables.prep()
        
        self.Q_function, self.dQ_function = observables.get_q_functions()
        self.log_Q_function, self.dlog_Q_function = observables.get_log_q_functions()
        
        #calculate average value of all observables and associated functions 
        self.expectation_observables = []
        self.epsilons_functions = []
        self.derivatives_functions = []
        
        
        self.h0 = []
        
        self.pi = []
        self.ni = []
        
        ##calculate the number of frames data has
        if type(data) is md.core.trajectory.Trajectory:
            total_data_frames = data.n_frames
        else: #its an array
            total_data_frames = np.shape(data)[0]
            
        ####Format Inputs####
        
        ##Check obs_data
        if obs_data is None: #use sim data for calculating observables
            obs_data = [data for i in range(len(self.observables.observables))]
        else:
            pass
        
        ##check if model is a list or not
        if isinstance(model, list):
            if model_state is None:
                raise IOError("model_state must be specified if model is a list")
                
        else: #convert model to be a list, construct model_state
            model = [model]
            if model_state is None:
                model_state = np.array([0 for i in range(total_data_frames)])
        self.num_models = len(model)
        self.model = model
        self.current_epsilons = model[0].get_epsilons() #assumes the first model is the one you want
        self.number_params = np.shape(self.current_epsilons)[0]
        
        #check model inputs
        if not np.max(model_state) < len(model):
            raise IOError("model_state formatted incorrectly. Values should be 0-X, where length of model is X+1") 
        
        if not total_data_frames == np.shape(model_state)[0]:
            raise IOError("shape of model_state and data do not match")
            
        #load data for each set, and compute energies and observations
        count = -1
        self.state_size = []
        self.state_ham_functions = []#debugging 
        for state_indices in data_sets:
            count += 1
            use_data = data[state_indices]
            num_in_set = np.shape(state_indices)[0]
            self.state_size.append(num_in_set)
            which_model = model_state[state_indices]
            ##assumes order does not matter, so long as relative order is preserved.
            ##since the epsilons_function and derivatives are summed up later
            this_epsilons_function = []
            this_derivatives_function = []
            total_h0 = 0
            for idx in range(self.num_models):
                if idx in which_model:
                    this_indices = np.where(which_model == idx)
                    this_data = use_data[this_indices]
                    epsilons_function, derivatives_function = model[idx].get_potentials_epsilon(this_data)
                    size_array = np.shape(epsilons_function(self.current_epsilons))[0]
                    for test in derivatives_function(self.current_epsilons):
                        assert np.shape(test)[0] == size_array 
                    this_epsilons_function.append(epsilons_function)
                    this_derivatives_function.append(derivatives_function)
                    this_h0 = epsilons_function(model[idx].get_epsilons())
                    try:
                        total_h0 = np.append(total_h0, this_h0, axis=0)
                    except:
                        total_h0 = this_h0
                assert len(this_epsilons_function) == len(this_derivatives_function)
            num_functions = len(this_epsilons_function)
            
            ##define new wrapper functions to wrap up the computation of several hamiltonians from different models
            ham_calc = HamiltonianCalculator(this_epsilons_function, this_derivatives_function, self.number_params, num_in_set, count)
            
            self.epsilons_functions.append(ham_calc.epsilon_function)
            self.derivatives_functions.append(ham_calc.derivatives_function)
            self.h0.append(total_h0)
            self.state_ham_functions.append(ham_calc)
            #process obs_data so that only the desired frames are passed
            use_obs_data = []
            for obs_dat in obs_data:
                use_obs_data.append(obs_dat[state_indices])
            observed, obs_std = observables.compute_observations(use_obs_data)
            
            self.expectation_observables.append(observed)

            self.ni.append(num_in_set)
            self.pi.append(num_in_set)
        
        ##check the assertion. make sure everything same size
        for i in range(self.number_equilibrium_states):
            print "For state %d" % i
            print np.shape(self.epsilons_functions[i](self.current_epsilons))[0]
            print np.shape(self.h0[i])[0]
            assert np.shape(self.epsilons_functions[i](self.current_epsilons))[0] == np.shape(self.h0[i])[0]
            size = np.shape(self.epsilons_functions[i](self.current_epsilons))[0] 
            for arrr in self.derivatives_functions[i](self.current_epsilons):
                assert np.shape(arrr)[0] == size
           
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
        
        ##debugging options below
        self.count_Qcalls = 0
        self.count_hepsilon = 0
        self.count_dhepsilon = 0
        self.trace_Q_values= []
        self.trace_loq_Q_values= []
        
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
        
    def Qfunction_epsilon(self, epsilons, Count=0):
        #initiate value for observables:
        
        next_observed, total_weight, boltzman_weights = self.get_reweights(epsilons)
        
        #Minimization, so make maximal value a minimal value with a negative sign.
        Q = -1.0 * self.Q_function(next_observed) 
        
        ##debug
        self.count_Qcalls += 1
        self.trace_Q_values.append(Q)
        
        return Q
        
    def log_Qfunction_epsilon(self, epsilons, Count=0):
        next_observed, total_weight, boltzman_weights = self.get_reweights(epsilons)
        
        #Minimization, so make maximal value a minimal value with a negative sign.
        Q = self.log_Q_function(next_observed) 
        #print epsilons
        
        ##debug
        self.count_Qcalls += 1
        self.trace_log_Q_values.append(Q)
        
        return Q

    def derivatives_Qfunction_epsilon(self, epsilons, Count=0):
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
        
        self.trace_Q_values.append(Q)
        return Q, dQ_vector

    def derivatives_log_Qfunction_epsilon(self, epsilons, Count=0):
        next_observed, total_weight, boltzman_weights = self.get_reweights(epsilons)
        
        Q = self.log_Q_function(next_observed)
        derivative_observed_first, derivative_observed_second = self.get_derivative_pieces(epsilons, boltzman_weights)
        
        dQ_vector = []
        for j in range(self.number_params):
            derivative_observed = (derivative_observed_first[j]  - (next_observed * derivative_observed_second[j])) / total_weight 
            dQ = self.dlog_Q_function(next_observed, derivative_observed) 
            dQ_vector.append(dQ)
        
        dQ_vector = np.array(dQ_vector)
        
        self.trace_log_Q_values.append(Q)
        return Q, dQ_vector
    
    def get_derivative_pieces(self, epsilons, boltzman_weights):
        derivative_observed_first = [np.zeros(self.num_observable) for j in range(self.number_params)]
        derivative_observed_second = [np.zeros(self.num_observable) for j in range(self.number_params)]
        for i in range(self.number_equilibrium_states):
            deriv_function = self.derivatives_functions[i](epsilons)
            self.count_dhepsilon += 1
            for j in range(self.number_params):
                next_weight_derivatives = np.sum(boltzman_weights[i] * deriv_function[j]) / self.ni[i]
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
                self.count_hepsilon += 1
                boltzman_weights.append(boltzman_wt)
        except:
            print "Exponential Function Failed"
            exponents = []
            boltzman_weights = []
            max_val = 0
            for i in range(self.number_equilibrium_states):
                exponent = self.epsilons_functions[i](epsilons) - self.h0[i]
                self.count_hepsilon += 1
                max_exponent = np.max(exponent)
                if max_exponent > max_val:
                    max_val = max_exponent
                exponents.append(exponent)
            for i in range(self.number_equilibrium_states):
                boltzman_wt = np.exp(exponents[i] - max_val + K_shift)
                boltzman_weights.append(boltmzan_wt)
        
        assert len(boltzman_weights) == self.number_equilibrium_states
        for idx,state in enumerate(boltzman_weights):
            assert np.shape(state)[0] == self.state_size[idx]
        return boltzman_weights 
    
    def save_debug_files(self):
        """ save debug files
        
        file1: counts for functions calls.
        file2: trace of Q values during optmization
        
        """
        
        f = open("function_calls.dat", "w")
        f.write("Number of times Q was computed: %d\n" % self.count_Qcalls)
        f.write("Number of times the Hamiltonian was computed: %d\n" % self.count_hepsilon)
        f.write("Number of times the derivative of the Hamiltonian was computed: %d\n" % self.count_dhepsilon)
        
        np.savetxt("trace_Q_values.dat", self.trace_Q_values)
        np.savetxt("trace_log_Q_values.dat", self.trace_log_Q_values)
        
        
        
class HamiltonianCalculator(object):
    def __init__(self, hamiltonian_list, derivative_list, number_params, size, state):
        self.hamiltonian_list = hamiltonian_list
        self.derivative_list = derivative_list
        assert len(self.hamiltonian_list) == len(self.derivative_list)
        self.num_functions = len(self.hamiltonian_list)
        self.number_params = number_params
        self.size = size
        self.state = state
        
    def epsilon_function(self,epsilons):
        except_count = 0
        for idx in range(self.num_functions):
            this_array = self.hamiltonian_list[idx](epsilons)
            try:
                total_array = np.append(total_array, this_array, axis=0)
            except:
                except_count += 1
                total_array = this_array
            assert except_count == 1
        return total_array
    
    def derivatives_function(self,epsilons):
        count = 0
        except_count = 0
        for idx in range(self.num_functions):
            this_list = self.derivative_list[idx](epsilons)
            count += np.shape(this_list[0])[0]    
            try:
                for j in range(self.number_params):
                    total_list[j] = np.append(total_list[j], this_list[j], axis=0)
            except:
                #must duplicate list
                #if not duplicated, errors will result
                #derivative_list functions might return a stored list
                #List pass by reference, so python points to that list
                #when it appends, it appends to the internal list as well
                #this causes total_list to grow every function call
                #this is not a problem, with arrays
                #Arrays are pass by reference as well
                #But np.append duplicates the array automatically
                total_list = list(this_list)
                
        return total_list
    
    
def save_error_files(diff, epsilons, h0, kshift):
    np.savetxt("ERROR_EXPONENTIAL_FUNCTION", diff)
    np.savetxt("ERROR_EPSILONS", epsilons)
    np.savetxt("ERROR_H0", h0)
    f = open("ERROR_LOG_EXPONENTIAL_FUNCTION", "w")
    f.write("\n\nCurrent Shift: %f\n" % kshift)
    f.close()    

