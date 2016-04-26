""" Optimizes model parameters using a maximum likelihood approach

This module has the functions for optimizing the a quality funciton Q,
referred to as Qfunc or Q throughout. Q has many components, but mainly
you seek to optimize some parameters epsilons, hereby referred to as
epsilons, x0 or x. Epsilons are model dependent. 

"""
import random
import time
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
    """ Use the scipy basinhopping method """
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
    """ Experimental annealing methods for testing purposes only"""
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

def solve_annealing_custom(Qfunc, x0,  ntries=1000, scale=0.2, logq=False, stuck=100, bounds=[-10,10]):
    """ Custom stochastic method 
    
    Currently perturbs randomly and steps downhill.
    """
    
    numparams = np.shape(x0)[0]
    
    #sort bounds so the smallest is first
    if not np.shape(bounds)[0] == 2:
        bounds = [-10, 10]
        print "Invalid bounds, setting to default of (-10,10)"
    
    use_bounds = []
    use_bounds.append(np.min(bounds))
    use_bounds.append(np.max(bounds))
    
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
        xnext[xnext<use_bounds[0]] = use_bounds[0]
        xnext[xnext>use_bounds[1]] = use_bounds[1]
        
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
        xnext[xnext<use_bounds[0]] = use_bounds[0]
        xnext[xnext>use_bounds[1]] = use_bounds[1]
        Qnext = Qfunc(xnext)
        
        if Qnext < Qval:
            xval = xnext
            Qval = Qnext
        
        else:
            pass
        if Qval < minQ:
            minima = xval 
    
    return minima   

def solve_newton_step_custom(Qfunc, x0, stepsize=1.0, logq=False, maxiters=200, proximity=1.0, qstop=1.0):
    """ Solve by taking a newton step"""
    
    xval = x0
    go = True
    count = 0
    qval, qderiv = Qfunc(xval)
    while go:
        count += 1
        target =  -qderiv
        step = target - xval
        if np.linalg.norm(step) > stepsize:
            step *= (stepsize/np.linalg.norm(step))
        
        qold = qval
        qoldest = qval
        xold = xval
        xval = xval + step
        print "Current Q Value:"
        print qval
        #print "moving to:"
        #print xval
        
        #Find an optimal step size
        go_find_step = True
        go_find_step_count = 0
        while go_find_step:
            qval, qderiv = Qfunc(xval)
            if qval > qold:
                print "Scaling down the step"
                step *= 0.1
                xval = xold + step
                go_find_step_count += 1
                if go_find_step_count == 4:
                    print "Failed to find a minima within 1/1000 of the step"
                    print "Exiting the Optimizer"
                    go = False
                    go_find_step = False
                    qval = qold
            else:
                print "Step is OKay now"
                go_find_step = False
                
        go_along_line = True
        while go_along_line:
            print "Going along line"
            qval, qderiv = Qfunc(xval)
            if qval > qold:
                go_along_line = False #started going up hill
                xval -= step
            else:
                qold = qval
                xval += step
        
        qval, qderiv = Qfunc(xval)
                
        diffq = qoldest - qval
        if diffq > 0 and diffq < qstop:
            print "Stopping as dq is within qstop"
            go = False
        if count > maxiters:
            print "Reached the maximum number of iterations."
            go = False
            
    return xval
    
def solve_cg(Qfunc, x0, norm=1):
    """ use the scipy.optimize.minimize, method=CG """
    def func(x):
        return Qfunc(x)[0]
    def dfunc(x):
        return -Qfunc(x)[1]
    optimal = optimize.fmin_cg(func, x0, fprime=dfunc, norm=norm)
    #print optimal
    #if not optimal[4] == 0:
        
        #raise IOError("Minimization failed to find a local minima. Code %d" % optimal[4]) 
        
    return optimal

def solve_bfgs(Qfunc, x0, bounds=None):
    """ Use the scipy.optimize.minimize (bfgs) method"""
    if bounds is None:
        optimal = optimize.minimize(Qfunc, x0, method="L-BFGS-B", jac=True)
    else:
        optimal = optimize.minimize(Qfunc, x0, method="L-BFGS-B", jac=True, bounds=bounds)
    if not optimal.success == True:
        print optimal.message
        #raise IOError("Minimization failed to find a local minima using the simplex method") 
        return x0
           
    return optimal.x

def solve_one_step(Qfunc, x0, stepsize=1.0, bounds=None):
    """ take steps in steepest decent until reaches the smallest value"""    
    xval = x0
    go = True
    count = 0
    qval, qderiv = Qfunc(xval)
    qold = qval
    target =  -qderiv
    step = target - xval
    if np.linalg.norm(step) > stepsize:
        step *= (stepsize/np.linalg.norm(step))
    
    #detemrine bounds
    bound_terms = []
    if bounds is None:
        for i in xval:
            bound_terms.append([0,10])
    else:
        bound_terms = bounds
        
    #Find an optimal step size
    go_find_step = True
    go_find_step_count = 0
    while go_find_step:
        qval, qderiv = Qfunc(xval)
        if qval > qold:
            print "Scaling down the step"
            step *= 0.1
            xval = xold + step
            go_find_step_count += 1
            if go_find_step_count == 4:
                print "Failed to find a minima within 1/1000 of the step"
                print "Exiting the Optimizer"
                go = False
                go_find_step = False
                qval = qold
        else:
            print "Step is OKay now"
            go_find_step = False
            
    go_along_line = True
    while go_along_line:
        print "Going along line"
        qval, qderiv = Qfunc(xval)
        if qval > qold and check_bounds(xval,bound_terms):
            go_along_line = False #started going up hill
            xval -= step
        else:
            qold = qval
            xval += step
    
    return xval

def check_bounds(eps, bounds):
    good = True
    for i in eps:
        if eps < bounds[0] or eps > bounds[1]:
            good = False
    
    return good  
        
def max_likelihood_estimate(data, data_sets, observables, model, obs_data=None, solver="simplex", logq=False, x0=None, kwargs={}):
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
        obs_data (list): Use if data set for computing observables is 
            different from data for computing the energy. List  contains 
            arrays where each array-entry corresponds to the observable 
            in the ExperimentalObservables object. Arrays are specified 
            with first index corresponding to the frame and second index 
            to the data. Default: Use the array specified in data for 
            all observables. 
        ntries (int): Number of random starting epsilons to generate in 
            case solution. Defaults to 0.
        solver (str): Optimization procedures. Defaults to Simplex. 
            Available methods include: simplex, anneal, cg, custom.
        logq (bool): Use the logarithmic Q functions. Default: False.
        x0 (array): Specify starting epsilons for optimization methods. 
            Defaults to current epsilons from the model.
            
    Returns:
        eo (EstimatorsObject): Object that contains the data used for 
            the computation and the results.
            
    """
    
    eo = EstimatorsObject(data, data_sets, observables, model, obs_data=obs_data)

    if solver in ["cg", "newton", "bfgs", "one"]:
        derivative = True
    else:
        derivative = False
    Qfunction_epsilon = eo.get_function(derivative, logq)
    
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
    t1 = time.time()
    #Then run the solver
    
    ##add keyword args thatn need to be passed
    #kwargs["logq"] = logq
    
    function_dictionary = {"simplex":solve_simplex_global}
    function_dictionary["anneal"] = solve_annealing
    function_dictionary["cg"] = solve_cg
    function_dictionary["anneal_exp"] = solve_annealing_experimental
    function_dictionary["custom"] = solve_annealing_custom
    function_dictionary["newton"] = solve_newton_step_custom
    function_dictionary["bfgs"] = solve_bfgs
    function_dictionary["one"] = solve_one_step
    if solver not in function_dictionary:
        raise IOError("Invalid Solver. Please specify a valid solver")
    func_solver = function_dictionary[solver]
    
    new_epsilons = func_solver(Qfunction_epsilon, current_epsilons, **kwargs)
    
    
    t2 = time.time()
    total_time = (t2-t1) / 60.0
    print "Optimization Complete: %f minutes" % total_time
    
    #then return a new set of epsilons inside the EstimatorsObject
    eo.save_solutions(new_epsilons)
    return eo
    
    
    
    
    
    
    
    
