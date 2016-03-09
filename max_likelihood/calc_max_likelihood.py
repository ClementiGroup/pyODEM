""" Solves using the max_likelihood_method"""

import numpy as np

import scipy.optimize as optimize

class SolutionHolder(object):
    pass

def solve_simplex(func, x0):
    optimal = optimize.minimize(func, x0, method="Nelder-Mead")
    print optimal.message
    
    if not optimal.success == True:
        raise IOError("Minimization failed to find a local minima using the simplex method") 
            
    return optimal.x
    
def solve_simplex_global(func, x0):
    out_eps = solve_simplex(func, x0)
        
    num_eps = np.shape(x0)[0]
    if np.sum(np.abs(x0-out_eps)) < num_eps*0.1:
        #check for global minima since near a local minima
        out_Q  = func(out_eps)
        for i in range(100):
            rand = np.random.rand(num_eps)
            for j in range(num_eps):
                if np.random.rand(1)[0] < 0.5:
                    rand[j] *= -1.0

            optime = solve_simplex(func, x0 + (rand))
            optimQ = func(optime)
            
            if out_Q > optimQ:
                out_eps = optime
                print "found better global Q"
                print "new starting epsilons:"
                print x0 + rand
                print "new finished epsilons:"
                print optime
                print "new Q"
                print optimQ
    return out_eps
             

def estimate_new_epsilons(data, data_sets, observables, model):
    
    #observables get useful stuff like value of beta
    beta = model.beta
    number_equilibrium_states = len(data_sets)
    
    observables.prep()

    #first calculate average value of all observables and associated functions 
    
    expectation_observables = []
    epsilons_functions = []
    
    current_epsilons = model.get_epsilons()
    h0 = []
    
    pi = []
    ni = []
    
    Q_function = observables.get_q_function()
    
    
    
    for i in data_sets:
        use_data = data[i]
        epsilons_function = model.get_potentials_epsilon(use_data)
        observed, obs_std = observables.compute_observations(use_data) #outputs a std in case people want to plot with error bars
        num_in_set = np.shape(use_data)[0]
        
        expectation_observables.append(observed)
        epsilons_functions.append(epsilons_function)
        
        h0.append(epsilons_function(current_epsilons))
        
        ni.append(num_in_set)
        pi.append(num_in_set)
        
        
        
    num_observable = np.shape(observed)[0] ## Calculate the number of observables

        
    pi =  np.array(pi).astype(float)
    pi /= np.sum(pi)
    state_prefactors = []
    
    for i in range(number_equilibrium_states):
        state_prefactor = pi[i] * expectation_observables[i]##these are the parts that don't depend on the re-weighting   
        state_prefactors.append(state_prefactor)

    #then wrap up a funciton that takes only epsilons, and outputs a value for Q
    def Qfunction_epsilon(epsilons):
        #initiate value for observables:
        next_observed = np.zeros(num_observable)
        
        #calculate re-weighting for all terms 
        total_weight = 0.0 #add up all re-weighted terms and normalize to keep probability equal to 1
        for i in range(number_equilibrium_states):
            #exp(-beat dH) weighted for this state is:
            next_weight = np.sum(np.exp(epsilons_functions[i](epsilons) - h0[i])) / ni[i]
            next_observed += next_weight * state_prefactors[i]
            total_weight += next_weight
        
        next_observed /= total_weight
        
        Q = -1.0 * Q_function(next_observed) #Minimization, so make maximal value a minimal value with a negative sign.

        return Q
        
    #Then run the solver
    new_epsilons = solve_simplex_global(Qfunction_epsilon, current_epsilons) #find local minima of this current set of epsilons

    #then return a new set of epsilons... consider adding a method to the model object to automatically udpate epsilons
    sh = SolutionHolder()
    sh.state_prefactors = state_prefactors
    sh.new_epsilons = new_epsilons
    sh.old_epsilons = current_epsilons
    sh.Qfunction_epsilon = Qfunction_epsilon
    
    return sh
    
    
    
    
    
    
    
    
