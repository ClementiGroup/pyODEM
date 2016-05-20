""" Optimizers implemented for pyfexd package

This module contains several optimization routines. In particular, I 
found these to work well under certain conditions for applications 
pyfexd was intended. It should be noted that the user is not limited to 
only these functions. Users should feel free to add their own functions 
either by adding it to this package or by passing the optimizer directly 
to the estimators. 

Positional Arguments for all optimizers:
    Qfunc (method): Q function to be minimized.
    x0 (array): Array of starting epsilons.

Keyword Arguments are explained in the relevant method as each optimizer 
has a unique set of user options.

There are two sections for this module: 
1) scipy optimizer wrappers methods 
2) custom methods

The scipy wrappers are meant to wrap a sciyp optimizer to be called from 
the script directly and output only the necessary information. Internal 
checks can be built in to make sure it doesn't fail. The custom methods 
are methods written specifically for this package. These are not 
necessarily new methods, but are rather implementations for which no 
good scipy or numpy implementation exists. Note to developers: Please 
keep this module organized.     

"""

import random
import time
import numpy as np
import scipy.optimize as optimize

#### CUSTOM ERROR MESSAGES ####
class FailedToOptimizeException(Exception):
    def __init__(self, method, relevant_parameters):
        message = "%s failed to optimize." % method
        for key in relevant_parameters:
            message += "\n%s: %s" % (key, relevant_parameters[key])
        
        super(FailedToOptimizeException, self).__init__(message)

#### SCIPY OPTIMIZER WRAPPERS ####
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
        terms = {"message":optimal.message}
        raise FailedToOptimizeException("simplex", terms) 
           
    return optimal.x
    
def solve_cg(Qfunc, x0, norm=1, gtol=10**-5):
    """ use the scipy.optimize.minimize, method=CG """
    
    optimal = optimize.minimize(Qfunc, x0, jac=True, method="CG", options={'norm':norm, 'gtol':gtol})
    
    if not optimal.success == True:
        terms = {"message":optimal.message}
        raise FailedToOptimizeException("Conjugate-Gradient", terms) 
        
    return optimal.x

def solve_bfgs(Qfunc, x0, bounds=None, gtol=10**-5):
    """ Use the scipy.optimize.minimize (bfgs) method"""
    if bounds is None:
        optimal = optimize.minimize(Qfunc, x0, method="L-BFGS-B", jac=True, options={'gtol':gtol})
    else:
        optimal = optimize.minimize(Qfunc, x0, method="L-BFGS-B", jac=True, bounds=bounds, options={'gtol':gtol})
    if not optimal.success == True:
        terms = {"message":optimal.message}
        raise FailedToOptimizeException("BFGS", terms) 
           
    return optimal.x
    
def solve_annealing(Qfunc, x0, ntries=1000, scale=0.2, Tbarrier=200):
    """ Use the scipy basinhopping method """
    def test_bounds(f_new=None, x_new=None, f_old=None, x_old=None):
        if np.max(np.abs(x_new)) > 5:
            return False
        else:
            return True
    
    optimal = optimize.basinhopping(Qfunc, x0, niter=ntries, T=Tbarrier, stepsize=scale, accept_test=test_bounds, interval=100)  
    
    return optimal.x

#### CUSTOM METHODS ####
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
            
            try:
                optime = solve_simplex(Qfunc, start_eps+rand)
                optimQ = Qfunc(optime)
            except FailedToOptimizeException:
                print "Epsilons failed to optimize. Try another."
            
            #save if new found func Q value is better than the rest
            if out_Q > optimQ:
                out_eps = optime
                print "found better global Q"
                print "new finished epsilons:"
                print optime
                print "new Q"
                print optimQ
                out_Q = optimQ
                    
    return out_eps
    
def solve_annealing_experimental(Qfunc, x0, ntries=1000, scale=0.2, Tbarrier=200):
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
             
    optimal = optimize.basinhopping(Qfunc, x0, niter=ntries, T=Tbarrier, accept_test=test_bounds, take_step=take_step_custom)  
    
    return optimal.x

def solve_annealing_custom(Qfunc, x0,  ntries=1000, scale=0.2, stuck=100, bounds=[-10,10]):
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

def solve_newton_step_custom(Qfunc, x0, stepsize=1.0, maxiters=200, proximity=1.0, qstop=1.0):
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


def solve_one_step(Qfunc, x0, stepsize=1.0, bounds=None):
    """ take steps in steepest decent until reaches the smallest value"""    
    xval = x0
    xold = x0
    go = True
    count = 0
    qval, qderiv = Qfunc(xval)
    print "Starting Value for Q:"
    print qval
    #print np.min(np.abs(qderiv)) 
    #print np.max(np.abs(qderiv))
    qold = qval
    target = -qderiv
    step = target - xval
    if np.linalg.norm(step) > stepsize:
        step *= (stepsize/np.linalg.norm(step))
    #print np.min(np.abs(step)) 
    #print np.max(np.abs(step))
    #detemrine bounds
    bound_terms = []
    if bounds is None:
        for i in xval:
            bound_terms.append([0,10])
    else:
        bound_terms = bounds
     
    #go along line until it reaches a minima               
    go_along_line = True
    while go_along_line:
        print "Going along line"
        xval = enforce_bounds(xold+step, bound_terms)
        qval, qderiv = Qfunc(xval)
        print qval
        print np.max(np.abs(xval-xold))
        if qval > qold:
            print "Started going uphill. Terminating"
            go_along_line = False #started going up hill
        elif check_bounds(xval,bound_terms):
            print "Hit the bounded wall"
            go_along_line = False #started going up hill
        else:
            qold = qval
            xold = xval
    
    return xold

def enforce_bounds(eps, bounds):
    for idx,i in enumerate(eps):
        if i < bounds[idx][0]:
            eps[idx] = bounds[idx][0]
        elif i > bounds[idx][1]:
            eps[idx] = bounds[idx][1]         
    return eps
    
def check_bounds(eps, bounds):
    bad = False
    for idx,i in enumerate(eps):
        if i < bounds[idx][0] or i > bounds[idx][1]:
            bad = True
    
    return bad 

#### dictionary of all the functions ####
function_dictionary = {"simplex":solve_simplex}
function_dictionary["simplex_global"] = solve_simplex_global
function_dictionary["anneal"] = solve_annealing
function_dictionary["cg"] = solve_cg
function_dictionary["anneal_exp"] = solve_annealing_experimental
function_dictionary["custom"] = solve_annealing_custom
function_dictionary["newton"] = solve_newton_step_custom
function_dictionary["bfgs"] = solve_bfgs
function_dictionary["one"] = solve_one_step

