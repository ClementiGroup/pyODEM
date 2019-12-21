""" Optimizers implemented for pyODEM package

This module contains several optimization routines. In particular, I found these
to work well under certain conditions for which pyODEM was intended. It should
be noted that the user is not limited to only these functions. Users should feel
free to add their own functions either by adding it to this package or by
passing the optimizer directly to the estimators.

Positional Arguments for all optimizers:
    Qfunc (method): Q function to be minimized.
    x0 (array): Array of starting epsilons.

Keyword Arguments are explained in the relevant method as each optimizer has a
unique set of user options.

There are two sections for this module:
1) scipy optimizer wrappers methods
2) custom methods

The scipy wrappers are meant to wrap a sciyp optimizer to be called from the
script directly and output only the necessary information. Internal checks can
be built in to make sure it doesn't fail. The custom methods are methods written
specifically for this package. These are not necessarily new methods, but are
rather implementations for which no good scipy or numpy implementation exists.

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
def solve_simplex(Qfunc, x0, tol=None):
    """ Optimizes a function using the scipy siplex method.

    This method does not require computing the Jacobian and works in arbitrarily
    high dimensions. This method works by computing new values of the funciton
    to optimizie func in the vicinity of x0 and moves in the direction of
    minimizing func.

    Args:
        Qfunc (method): func returns a float and takes a list or array x of
            inputs.
        x0 (list/array floats): Initial guess of for optimal parameters

    Returns:
        array(float): optimal value of x if found.

    Raises:
        IOerror: if optimization fails

    """

    optimal = optimize.minimize(Qfunc, x0, method="Nelder-Mead", tol=tol)

    if not optimal.success == True:
        terms = {"message":optimal.message}
        raise FailedToOptimizeException("simplex", terms)

    return optimal.x

def solve_cg(Qfunc, x0, norm=1, gtol=10**-5, bounds=None, tol=None):
    """ use the scipy.optimize.minimize, method=CG """

    if bounds is None:
        optimal = optimize.minimize(Qfunc, x0, jac=True, method="CG", tol=tol, options={'norm':norm, 'gtol':gtol})
    else:
        optimal = optimize.minimize(Qfunc, x0, jac=True, method="CG", tol=tol, options={'norm':norm, 'gtol':gtol}, bounds=bounds)
    if not optimal.success == True:
        terms = {"message":optimal.message}
        raise FailedToOptimizeException("Conjugate-Gradient", terms)

    return optimal.x

def solve_bfgs(Qfunc, x0, bounds=None, gtol=None, ftol=None, tol=None, maxiter=10000):
    """ Use the scipy.optimize.minimize (bfgs) method """

    if ftol is None and gtol is None:
        # default termination criterion
        gtol = 10 ** -5
    options_dictionary = {'maxiter':maxiter}
    if ftol is not None:
        options_dictionary['ftol'] = ftol
    if gtol is not None:
        options_dictionary['gtol'] = gtol

    if bounds is None:
        optimal = optimize.minimize(Qfunc, x0, method="L-BFGS-B", jac=True, tol=tol, options=options_dictionary)
    else:
        optimal = optimize.minimize(Qfunc, x0, method="L-BFGS-B", jac=True, bounds=bounds, tol=tol, options=options_dictionary)

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

    Uses the solve_simplex() method by computing multiple random frontier points
    nearby the local minima closest to starting epsilons x0. Then optimizes each
    point and takes the most optimal result.

    Args:
        ntries (int): Number of random startung epsilons to generate in case
            solution is stuck in a local minima. Defaults to 0.

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
        print("Searching for a global minima. Trying %d times." % ntries)
        num_eps = np.shape(x0)[0]
        for i in range(ntries):
            #generate random +/- perturbation to local minima found with
            #uniform distribution from -1 to 1'
            rand = np.random.rand(num_eps)
            for j in range(num_eps):
                if np.random.rand(1)[0] < 0.5:
                    rand[j] *= -1.0

            print("new starting epsilons:")
            print(start_eps + rand)

            try:
                optime = solve_simplex(Qfunc, start_eps+rand)
                optimQ = Qfunc(optime)
            except FailedToOptimizeException:
                print("Epsilons failed to optimize. Try another.")

            #save if new found func Q value is better than the rest
            if out_Q > optimQ:
                out_eps = optime
                print("found better global Q")
                print("new finished epsilons:")
                print(optime)
                print("new Q")
                print(optimQ)
                out_Q = optimQ

    return out_eps

def solve_annealing_experimental(Qfunc, x0, ntries=1000, scale=0.2, Tbarrier=200):
    """ Experimental annealing methods for testing purposes only """
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
        print("Invalid bounds, setting to default of (-10,10)")

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
    """ Solve by taking a newton step """

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
        print("Current Q Value:")
        print(qval)
        #print "moving to:"
        #print xval

        #Find an optimal step size
        go_find_step = True
        go_find_step_count = 0
        while go_find_step:
            qval, qderiv = Qfunc(xval)
            if qval > qold:
                print("Scaling down the step")
                step *= 0.1
                xval = xold + step
                go_find_step_count += 1
                if go_find_step_count == 4:
                    print("Failed to find a minima within 1/1000 of the step")
                    print("Exiting the Optimizer")
                    go = False
                    go_find_step = False
                    qval = qold
            else:
                print("Step is OKay now")
                go_find_step = False

        go_along_line = True
        while go_along_line:
            print("Going along line")
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
            print("Stopping as dq is within qstop")
            go = False
        if count > maxiters:
            print("Reached the maximum number of iterations.")
            go = False

    return xval


def solve_one_step(Qfunc, x0, stepsize=1.0, bounds=None):
    """ take steps in steepest decent until reaches the smallest value """
    xval = x0
    xold = x0
    go = True
    count = 0
    qval, qderiv = Qfunc(xval)
    print("Starting Value for Q:")
    print(qval)
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
        print("Going along line")
        xval = enforce_bounds(xold+step, bound_terms)
        qval, qderiv = Qfunc(xval)
        print(qval)
        print(np.max(np.abs(xval-xold)))
        if qval > qold:
            print("Started going uphill. Terminating")
            go_along_line = False #started going up hill
        elif check_bounds(xval,bound_terms):
            print("Hit the bounded wall")
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
def solve_sgd_custom(Qfunc, x0,
                     stepsize=0.001,
                     maxiters=60,
                     batch_number=1,
                     gtol = 1.0e-5,
                     alpha = 0.0,
                     lr_decay = True,
                     num_of_step = 200,
                     multiplicator = 0.70):

    """ Solve using stochastic gradent descent

    Arguments:
    ----------
   Qfunc : function
       function for computing loss function and gradient
   x0    : list
       list of model parameters
   stepsize : float
       learning rate. Kept fixed  during the  optimization
  maxiters  : int
     maximum number of iterations
  batch_number : int
       number of batches to devide  parameter space
  alpha  : float
      regularization parameter
  lr_decay : bool, default True
      defines, whether learning rate decay is used
  num_of_step : int, default 200
      step, at which decay is introduced
  multiplicator : float default 0.70
      defines, how step decreases at iteration num_of_step

    """
    log_file = open("optimization_log.txt","wt")
    log_file.write("Starting stochastic gradient descent optimization \n")
    log_file.write("Parameters of optimization: \n")
    log_file.write("stepsize = {}, \n maxiters = {}, \n batch_number = {} \n".format(stepsize,maxiters, batch_number))
    log_file.write("gtol = {}, \n alpha = {}, \n lr_decay = {} \n".format(gtol,alpha, lr_decay))
    log_file.write("num_of_step = {}, \n multiplicator = {} \n".format(num_of_step,multiplicator))
    x_new = np.copy(x0)
    param_num = len(x_new) # number of parameters to optimize
    batch_size = param_num//batch_number # minimum number of elements in a batch
    for k in range(maxiters+1):
        if k%num_of_step==0:
            stepsize = multiplicator*stepsize
        changed_params = 0
        Q_value, gradient = Qfunc(x_new)
        gradient += (2*alpha)*(x_new-x0)  #Correct the gradient

        labels = np.random.permutation(param_num) # Generates list of elements
        for i in range(0,param_num,batch_size):
            batch_label = labels[i:i+batch_size]
            for j in batch_label:
                step = stepsize*gradient[j]
                x_new[j] = x_new[j] - step
                changed_params+=1
        Q_value,gradient=Qfunc(x_new)
        gradient += (2*alpha)*(x_new-x0)
        grad_norm = np.linalg.norm(gradient)
        log_file.write("New Q value after update: {} \n".format(Q_value))
        log_file.write("Norm of the gradient:   {} \n".format(grad_norm))
        log_file.write("New valuew of the loss function: {} \n".format(Q_value + alpha*sum(np.square(\
x_new-x0))))
        if grad_norm < gtol:
            log_file.write("Optimization done successfully in  {} steps \n".format(k))
            break

        print("Iteration {} done".format(k))
        log_file.write(("Iteration {} done \n".format(k))
        if k == maxiters:
            print("Number of interations exceeded. The last x is recorded")
            log_file.write("Number of interations exceeded. The last x is recorded \n")
            np.savetxt("epsilons_checkpoint.txt",x_new)
            raise FailedToOptimizeException("Number of iteration exceeded",{'iteration': maxiters})
    log_file.close()
    print(x_new)
    return(x_new)


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
function_dictionary["sgd"] = solve_sgd_custom
