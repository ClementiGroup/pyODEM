""" Optimizes model parameters using a maximum likelihood approach

This module has the functions for optimizing the a quality funciton Q,
referred to as Qfunc or Q throughout. Q has many components, but mainly
you seek to optimize some parameters epsilons, hereby referred to as
epsilons, x0 or x. Epsilons are model dependent. 

"""
import random
import numpy as np
import scipy.optimize as optimize
from estimators_class import EstimatorsObject

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
        print "Searching for a global minima. Trying %d times." % ntries
        num_eps = np.shape(x0)[0] 
        for i in range(ntries):
            #generate random +/- perturbation to local minima found with 
            #uniform distribution from -1 to 1'
            rand = np.random.rand(num_eps)
            for j in range(num_eps):
                if np.random.rand(1)[0] < 0.5:
                    rand[j] *= -1.0
            
            print "new starting epsilons:"
            print start_eps + rand
            
            optime = solve_simplex(Qfunc, start_eps+rand)
            optimQ = Qfunc(optime)
            
            #save if new found func Q value is better than the rest
            if out_Q > optimQ:
                out_eps = optime
                print "found better global Q"
                print "new finished epsilons:"
                print optime
                print "new Q"
                print optimQ
                    
    return out_eps

def solve_annealing(Qfunc, x0, ntries=1000, scale=0.2, logq=False):
    
    def test_bounds(f_new=None, x_new=None, f_old=None, x_old=None):
        if np.max(np.abs(x_new)) > 5:
            return False
        else:
            return True
    if logq:
        Tbarrier = 200
    else:
        Tbarrier = 0.5        
    optimal = optimize.basinhopping(Qfunc, x0, niter=ntries, T=Tbarrier, stepsize=scale, accept_test=test_bounds, interval=100)  
    
    return optimal.x

def solve_annealing_experimental(Qfunc, x0, ntries=1000, scale=0.2, logq=False):
    numparams = np.shape(x0)[0]
    def take_step_custom(x):
        perturbation = np.array([random.choice([-0.1, 0.1]) for i in range(numparams)])
        perturbation[np.where(np.random.rand(numparams)<0.8)] = 0
        return x + perturbation
    
    def test_bounds(f_new=None, x_new=None, f_old=None, x_old=None):
        if np.max(np.abs(x_new)) > 5:
            return False
        else:
            return True
    if logq:
        Tbarrier = 200
    else:
        Tbarrier = 0.5   
             
    optimal = optimize.basinhopping(Qfunc, x0, niter=ntries, T=Tbarrier, accept_test=test_bounds, take_step=take_step_custom)  
    
    return optimal.x

def solve_annealing_custom(Qfunc, x0,  ntries=1000, scale=0.2, logq=False):
    numparams = np.shape(x0)[0]
    #current vals
    Qval = Qfunc(x0)
    xval = x0
    #global vals
    minima = x0
    minQ = Qval
    for i in range(ntries):
        perturbation = np.array([random.choice([-0.1, 0.1]) for i in range(numparams)])
        perturbation[np.where(np.random.rand(numparams)<0.8)] = 0
        xnext = xval+perturbation
        Qnext = Qfunc(xnext)
        
        if Qnext < Qval:
            xval = xnext
            Qval = Qnext
        
        else:
            pass
        if Qval < minQ:
            minima = xval 
    
    for i in range(ntries/10):
        perturbation = np.array([random.uniform(-0.1, 0.1) for i in range(numparams)])
        perturbation[np.where(np.random.rand(numparams)<0.8)] = 0
        xnext = xval+perturbation
        Qnext = Qfunc(xnext)
        
        if Qnext < Qval:
            xval = xnext
            Qval = Qnext
        
        else:
            pass
        if Qval < minQ:
            minima = xval 
    
    return minima   
    
       
def solve_cg(Qfunc, x0):
    
    optimal = optimize.minimize(Qfunc, x0, jac=True, method="CG")
    print optimal.message
    if not optimal.success == True:
        
        raise IOError("Minimization failed to find a local minima using the simplex method") 
        
    return optimal.x
    
def max_likelihood_estimate(data, data_sets, observables, model, ntries=0, solver="simplex", logq="False", x0=None):
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
    eo = EstimatorsObject(data, data_sets, observables, model)

    if solver in ["cg"]:
        if logq:
            Qfunction_epsilon = eo.get_log_Q_function_derivatives()
        else:
            Qfunction_epsilon = eo.get_Q_function_derivatives()
    else:
        if logq:
            Qfunction_epsilon = eo.get_log_Q_function()
        else:
            Qfunction_epsilon = eo.get_Q_function()
    '''
    if logq:
        Qfunction_epsilon = eo.get_log_Q_function()
    else:
        Qfunction_epsilon = eo.get_Q_function()
    '''
    
    if x0 is None:
        current_epsilons = eo.current_epsilons
    else:
        current_epsilons = x0
    
    print "Starting Optimization"
    #Then run the solver
    if solver == "simplex":
        new_epsilons = solve_simplex_global(Qfunction_epsilon, current_epsilons, ntries=ntries)
    elif solver == "anneal":
        new_epsilons = solve_annealing(Qfunction_epsilon, current_epsilons,ntries=ntries, scale=0.2, logq=logq)
    elif solver == "cg":
        new_epsilons = solve_cg(Qfunction_epsilon, current_epsilons)
    elif solver == "anneal_exp":
        new_epsilons = solve_annealing_experimental(Qfunction_epsilon, current_epsilons, ntries=ntries, scale=0.2, logq=logq)
    elif solver == "custom":
        new_epsilons = solve_annealing_custom(Qfunction_epsilon, current_epsilons, ntries=ntries, scale=0.2, logq=logq)
    else:
        raise IOError("invalid solver, please select either: ...")
    
    
    #then return a new set of epsilons inside the EstimatorsObject
    
    eo.save_solutions(new_epsilons)
    return eo
    
    
    
    
    
    
    
    
