""" Optimizes model parameters using a maximum likelihood approach

This module has the functions for optimizing the a quality funciton Q,
referred to as Qfunc or Q throughout. Q has many components, but mainly
you seek to optimize some parameters epsilons, hereby referred to as
epsilons, x0 or x. Epsilons are model dependent. 

"""

import numpy as np
import scipy.optimize as optimize
from solutions import SolutionHolder

def solve_simplex(Qfunc, x0):
    """ Optimizes a function using the scipy siplex method.
    
    This method does not require computing the Jacobian and works in 
    arbitrarily high dimensions. This method works by computing new 
    values of the funciton to optimizie func in the vicinity of x0 and 
    moves in the direction of minimizing func.
    
    Args:
        Qfunc(x) (method): func returns a float and takes a list or 
            array x of inputs
        x0 (list/array floats): Initial guess of for optimal parameters
        
    Returns:
        array(float): optimal value of x if found. 
    
    Raises:
        IOerror: if optimization fails 
        
    """
    
    optimal = optimize.minimize(Qfunc, x0, method="Nelder-Mead")
    
    if not optimal.success == True:
        print optimal.message
        #raise IOError("Minimization failed to find a local minima using the simplex method") 
        return x0
           
    return optimal.x

    
def solve_simplex_global(Qfunc, x0, ntries=0):
    """ Attempts to find global minima using solve_simplex()
    
    Uses the solve_simplex() method by computing multiple random
    frontier points nearby the local minima closest to starting epsilons
    x0. Then optimizes each point and takes the most optimal result.
    
    Args:
        ntries (int): Number of random startung epsilons to generate 
            in case solution. Defaults to 0.
    
    Returns:
        array(float): Array of optimal values of epsilons found.
        
    """
    #best found epsilons so far
    out_eps = solve_simplex(Qfunc, x0)
    #best found Q so far
    out_Q  = Qfunc(out_eps)
    #starting minima to search around.
    start_eps = np.copy(out_eps)
     
    #check for global minima near found minima if ntries >0
    if not ntries <= 0:
        #generate random +/- perturbation to local minima found with 
        #uniform distribution from -1 to 1
        num_eps = np.shape(x0)[0] 
        for i in range(ntries):
            rand = np.random.rand(num_eps)
            for j in range(num_eps):
                if np.random.rand(1)[0] < 0.5:
                    rand[j] *= -1.0

            optime = solve_simplex(Qfunc, start_eps+rand)
            optimQ = Qfunc(optime)
            
            #save if new found func Q value is better than the rest
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

def solve_annealing(Qfunc, x0, ntries=1000, scale=0.2):
    numparams = np.shape(x0)[0]
    def take_step_custom(x):
        perturbation = np.random.randn(numparams)
        perturbation *= 0.2/np.sum(perturbation)
        return x + perturbation
    
    def test_bounds(f_new=None, x_new=None, f_old=None, x_old=None):
        if np.max(np.abs(x_new)) > 5:
            return False
        else:
            return True
            
    optimal = optimize.basinhopping(Qfunc, x0, niter=1000, T=0.5, stepsize=scale, accept_test=test_bounds, take_step=take_step_custom)  
    
    return optimal.x
    
def solve_cg(Qfunc, x0, ntries=1):
    
    print "solve using CG"
    optimal = optimize.minimize(Qfunc, x0, jac=True, method="CG")
    print optimal.message
    if not optimal.success == True:
        
        raise IOError("Minimization failed to find a local minima using the simplex method") 
        
    return optimal.x
    
def max_likelihood_estimate(data, data_sets, observables, model, ntries=0, solver="simplex", derivative="False"):
    """ Optimizes model's paramters using a max likelihood method
    
    Args:
        data (array): Contains all the data for a system, First index 
            should be for the frame number, remaining indices are up to 
            the user to handle using the ExperimentalObservables object.
        data_sets (list): List of arrays, where each array contains the 
            indices in the data for each equilibrium state. 
        observables (ExperimentalObservables): An object that is used
            for computing Q function value as well as the observed
            quantities. See: pyfexd/observables/exp_observables for
            full description.
        model (ModelLoader): Object that is used to load data and
            compute potential energies for the data set. See: 
            pyfexd/model_loaders/super_model for full description.
        ntries (int): Number of random startung epsilons to generate 
            in case solution. Defaults to 0.
        solver (str): Optimization procedures. Defaults to Simplex. 
        
    Returns:
        sh (SolutionHolder): new epsilons found as well as the Q
            function used.
            
    """
    #observables get useful stuff like value of beta
    beta = model.beta
    number_equilibrium_states = len(data_sets)
    
    observables.prep()

    #calculate average value of all observables and associated functions 
    
    expectation_observables = []
    epsilons_functions = []
    derivatives_functions = []
    
    current_epsilons = model.get_epsilons()
    number_params = np.shape(current_epsilons)[0]
    h0 = []
    
    pi = []
    ni = []
    
    Q_function, dQ_function = observables.get_q_functions()
    
    
    
    #load data for each set, and compute energies and observations
    for i in data_sets:
        use_data = data[i]
        epsilons_function, derivatives_function = model.get_potentials_epsilon(use_data)
        observed, obs_std = observables.compute_observations(use_data)
        num_in_set = np.shape(use_data)[0]
        
        expectation_observables.append(observed)
        epsilons_functions.append(epsilons_function)
        derivatives_functions.append(derivatives_function)
        
        h0.append(epsilons_function(current_epsilons))
        
        ni.append(num_in_set)
        pi.append(num_in_set)
    ##number of observables
    num_observable = np.shape(observed)[0]   
    pi =  np.array(pi).astype(float)
    pi /= np.sum(pi)
    
    ##Compute factors that don't depend on the re-weighting
    state_prefactors = []
    for i in range(number_equilibrium_states):
        state_prefactor = pi[i] * expectation_observables[i]   
        state_prefactors.append(state_prefactor)

    #then wrap up a function that takes only epsilons, and outputs Q value
    if derivative == False:
        def Qfunction_epsilon(epsilons):
            #initiate value for observables:
            next_observed = np.zeros(num_observable)
            

            #add up all re-weighted terms for normalizaiton
            total_weight = 0.0
            #calculate re-weighting for all terms 
            for i in range(number_equilibrium_states):
                next_weight = np.sum(np.exp(epsilons_functions[i](epsilons) - h0[i])) / ni[i]
                next_observed += next_weight * state_prefactors[i]
                total_weight += next_weight * pi[i]
            
            #normalize so total re-weighted probability is = 1.0
            next_observed /= total_weight
            
            #Minimization, so make maximal value a minimal value with a negative sign.
            Q = -1.0 * Q_function(next_observed) 

            return Q
        
    else:
        def Qfunction_epsilon(epsilons):
            next_observed = np.zeros(num_observable)
            dQ_vector = []
            #add up all re-weighted terms for normalizaiton
            total_weight = 0.0
            
            #Calculate the reweighting for all terms: Get Q, and next_observed
            for i in range(number_equilibrium_states):
                next_weight = np.sum(np.exp(epsilons_functions[i](epsilons) - h0[i])) / ni[i]
                next_observed += next_weight * state_prefactors[i]
                total_weight += next_weight * pi[i]
            
            #normalize so total re-weighted probability is = 1.0
            next_observed /= total_weight
            
            #Minimization, so make maximal value a minimal value with a negative sign.
            Q = -1.0 * Q_function(next_observed) 
            
            
            #Now calculate the derivatives 
            for j in range(number_params):
                derivative_observed = 0.0
                for i in range(number_equilibrium_states):
                    boltzman_weights = np.exp(epsilons_functions[i](epsilons) - h0[i])
                    next_weight_derivatives = np.sum(boltzman_weights * derivatives_functions[i](epsilons)[j]) / ni[i]
                    derivative_observed += next_weight_derivatives * state_prefactors[i]
        
                #normalize with same factor from calculating re-weighted Qs
                #1-D array of derivatives with respect to parameter j
                derivative_observed /= total_weight
                
                #Minimization, so make maximal value a minimal value with a negative sign.
                dQ = Q * dQ_function(next_observed, derivative_observed) 
                dQ_vector.append(dQ)
            
            return Q, np.array(dQ_vector)
        
    #Then run the solver
    dict_solvers = {"simplex":solve_simplex_global, "annealing":solve_annealing, "cg":solve_cg}
    dict_values = {"simplex":0, "annealing":1, "cg":2}
    
    new_epsilons = dict_solvers[solver](Qfunction_epsilon, current_epsilons, ntries=ntries)
    
    #new_epsilons = solve_simplex_global(Qfunction_epsilon, current_epsilons, ntries=ntries)

    #then return a new set of epsilons inside SolutionHolder
    sh = SolutionHolder()
    sh.state_prefactors = state_prefactors
    sh.new_epsilons = new_epsilons
    sh.old_epsilons = current_epsilons
    sh.Qfunction_epsilon = Qfunction_epsilon
    
    return sh
    
    
    
    
    
    
    
    
