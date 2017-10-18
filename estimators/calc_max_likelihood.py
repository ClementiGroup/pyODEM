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
import threading # only for the cross-validation function
import os

from estimators_class import EstimatorsObject
from optimizers import function_dictionary
from pyODEM.basic_functions import util

def ensure_derivative(derivative, solver):
    if derivative is None:
        if solver in ["cg", "newton", "bfgs", "one"]:
            derivative = True
        else:
            derivative = False

    return derivative

def get_solver(solver):
    if isinstance(solver, str):
        if solver not in function_dictionary:
            raise IOError("Invalid Solver. Please specify a valid solver")
        func_solver = function_dictionary[solver]
    else:
        func_solver = solver #assume a valid method was passed to it

    return func_solver

def max_likelihood_estimate(data, data_sets, observables, model, obs_data=None, solver="bfgs", logq=False, derivative=None, x0=None, kwargs={}, stationary_distributions=None, model_state=None):
    """ Optimizes model's paramters using a max likelihood method

    Args:
        See pyfexd.estimators.estimators_class.EstimatorsObject for:
            data (array), data_sets (list),
            observables (ExperimentalObservables), model (ModelLoader),
            obs_data(list) and stationary_distributions (list)

        solver (str): Optimization procedures. Defaults to Simplex.
            Available methods include: simplex, anneal, cg, custom.
        logq (bool): Use the logarithmic Q functions. Default: False.
        derivative (bool): True if Q function returns a derivative.
            False if it does not. Default is None, automatically
            selected based upon the requested solver.
        x0 (array): Specify starting epsilons for optimization methods.
            Defaults to current epsilons from the model.
        kwargs (dictionary): Key word arguments passed to the solver.

    Returns:
        eo (EstimatorsObject): Object that contains the data used for
            the computation and the results.

    """

    derivative = ensure_derivative(derivative, solver)

    eo = EstimatorsObject(data, data_sets, observables, model, obs_data=obs_data, stationary_distributions=stationary_distributions, model_state=model_state)

    Qfunction_epsilon = eo.get_function(derivative, logq)
    func_solver = get_solver(solver)

    if x0 is None:
        current_epsilons = eo.current_epsilons
    else:
        current_epsilons = x0

    print "Starting Optimization"
    t1 = time.time()
    #Then run the solver

    ##add keyword args thatn need to be passed
    #kwargs["logq"] = logq
    try:
        new_epsilons = func_solver(Qfunction_epsilon, current_epsilons, **kwargs)
    except:
        debug_dir = "debug_0"
        for count in range(100):
            debug_dir = "debug_%d" % count
            if not os.path.isdir(debug_dir):
                break
        os.mkdir(debug_dir)
        cwd = os.getcwd()
        os.chdir(debug_dir)
        eo.save_debug_files()
        os.chdir(cwd)
        raise

    t2 = time.time()
    total_time = (t2-t1) / 60.0

    print "Optimization Complete: %f minutes" % total_time

    #then return a new set of epsilons inside the EstimatorsObject
    eo.save_solutions(new_epsilons)
    return eo

def add_estimator_to_list(dtrajs, data, observables, model, obs_data, stationary_distributions, model_state, estimator_list, function_list, derivative, logq):

    indices = util.get_state_indices(dtrajs)
    estimators = EstimatorsObject(data, indices, observables, model, obs_data=obs_data, stationary_distributions=stationary_distributions, model_state=model_state)
    Qfunction_epsilon = estimators.get_function(derivative, logq) # use the logq function only
    estimator_list.append(estimators)
    function_list.append(Qfunction_epsilon)

class EstimateMulti(threading.Thread):
    def __init__(self, solver, current_epsilons, iter_container):
        self.solver = get_solver(solver)
        self.current_epsilons = current_epsilons
        self.iter_container = iter_container

        self.still_going = False # True when the loop is running

    def run(self):
        self.still_going = True
        kwargs, position, training_function, validation_function = self.iter_container.get_params()
        go = kwargs is not None
        while go:
            new_epsilons = self.solver(validation_function, self.current_epsilons)
            go = kwargs is not None
            this_score = validation_function(new_epsilons)
            self.iter_container.save(this_score, position)

            # get next params. If None, then the queue is empty
            kwargs, position, training_function, validation_function = self.iter_container.get_params()
            go = kwargs is not None
        self.still_going = False

class IterContainer(object):
    """ Contains the parameters and solution for cross validation """
    def __init__(self, training_functions, validation_functions, all_kwargs, all_kwargs_printable, order_list, order_sizes, lock=None):
        self.all_kwargs = all_kwargs
        self.all_kwargs_printable = all_kwargs_printable
        self.training_functions = training_functions
        self.validation_functions = validation_functions
        self.num_functions = len(self.training_functions)
        assert self.num_functions == len(self.validation_functions)
        self.num_params = len(all_kwargs)
        self.save_array = np.zeros((self.num_params, self.num_functions))

        self.send_indices = []
        for i in range(self.num_params):
            for j in range(self.num_functions):
                self.send_indices.append([i,j])

        total_send = len(self.send_indices)
        self.current_index = 0
        self.still_going = True
        self.lock = lock
        if self.lock is None:
            self.get_params = self._get_params_basic
        else:
            self.get_params = self._get_params_lock

    def _get_params_lock(self):
        self.lock.acquire()

        all_args = self._get_params_basic()

        self.lock.release()
        return all_args

    def _get_params_basic(self):
        if self.current_index < total_send:
            send_indices = self.send_indices[self.current_index]
            kwargs = self.all_kwargs[self.current_index]
            position = ()
            for i in send_indices:
                position += (i,)
            training_function = self.training_functions(send_indices[1])
            validation_function = self.validation_functions(send_indices[1])
        else:
            kwargs = None
            position = None
            training_function = None
            validation_function = None
            self.still_going = False

        return kwargs, position, training_function, validation_function

    def save(self, score, position):
        self.save_array[position] = score

    def get_best(self):
        total_scores = np.sum(self.save_array, axis=1)
        pos = 0
        best_score = None
        for idx,value in enumerate(total_scores):
            if best_score is None:
                best_score = value
                pos = idx
            elif value < best_score:
                best_score = value
                pos = idx

        print_str = "Best Score of %f, with the following selected parameters:\n" % (best_score)
        for thing in self.all_kwargs_printable[pos]:
            print_str += "%s:%s  " % (thing, self.all_kwargs_printable[pos][thing])
        print print_str

        return self.all_kwargs[pos]


def kfold_crossvalidation_max_likelihood(list_data, list_dtrajs, observables, model, list_obs_data=None, solver="bfgs", logq=False, derivative=None, x0=None, kwargs={}, list_kwargs={}, list_kwargs_printable={},  stationary_distributions=None, model_state=None, checkpoint_file=None, verbose=False, n_threads=1):

    derivative = ensure_derivative(derivative, solver)

    for thing in list_kwargs:
        if thing not in list_kwargs_printable:
            list_kwargs_printable[thing] = list_kwargs[thing]
        else:
            if not len(list_kwargs_printable[thing]) == len(list_kwrags[thing]):
                raise IOError("Option: %s. Size of list_kwrags_printable (%d) must be equal to list_kwargs (%d)" % (thing, len(list_kwargs_printable[thing]), len(list_kwargs[thing])))

    if n_threads == 1:
        use_multi = False
    else:
        if not n_threads > 0:
            raise IOError("n_threads must be between 1 - infinity")
        use_multi = True

    n_validations = len(list_data)
    cwd = os.getcwd()
    # Determine checkpoint file name and write/append to it
    if checkpoint_file is None:
        checkpoint_file = "%s/checkpoint_%d.txt" % (os.getcwd(), time.time()*1000)

    if os.path.isfile(checkpoint_file):
        f_check = open(checkpoint_file, "a")
        f_check.write("\n\n\n### Continuing Cross Validation ###")
    else:
        f_check = open(checkpoint_file, "w")
        f_check.write("### Starting Cross Validation ###")

    # determine the list of all keyword args
    all_entries = [entry for entry in list_kwargs]
    all_entry_sizes = []
    total_number_params = 1
    for entry in all_entries:
        entry_size = len(list_kwargs[entry])
        total_number_params *= entry_size
        all_entry_sizes.append(entry_size)

    all_kwargs = []
    all_kwargs_printable = []
    if len(all_entries) == 0:
        all_kwargs.append(kwargs)
    else:
        recursive_add_dic_entries(all_kwargs, kwargs, list_kwargs, all_entries, 0)
        recursive_add_dic_entries(all_kwargs_printable, {}, list_kwargs_printable, all_entries, 0)
    assert len(all_kwargs) == total_number_params
    assert len(all_kwargs_printable) == total_number_params

    if verbose:
        for idx,dic in enumerate(all_kwargs):
            print "Dictionary %d" % idx
            this_str = ""
            for entry in dic:
                this_str += "%s " % entry
                if isinstance(dic[entry],list):
                    this_str += str(dic[entry][0])
                else:
                    this_str += str(dic[entry])
                this_str += "   "
            print this_str

    # Make all the eo objects and qfunctions
    print "Initializing Training and Validation Functions"
    t1 = time.time()
    list_train_estimators = []
    list_validation_estimators = []

    list_train_qfunctions = []
    list_validation_qfunctions = []

    dtrajs_collected = []
    data_collected = []
    obs_data_collected = []
    estimators_collected = []
    functions_collected = []
    derivative_collected = []
    logq_collected = []

    func_solver = get_solver(solver)

    # determine which functions must be made
    for validation_index in range(n_validations):
        # For each validation index: make the q_functions and EO objects
        this_training = None
        this_dtrajs = None
        this_observables = None
        for idx in range(n_validations):
            if idx == validation_index:
                dtrajs_collected.append(list_dtrajs[idx])
                data_collected.append(list_data[idx])
                obs_data_collected.append(list_obs_data[idx])
                estimators_collected.append(list_validation_estimators)
                functions_collected.append(list_validation_qfunctions)
                derivative_collected.append(False)
                logq_collected.append(True)

            else:
                if this_training is None:
                    this_training = np.copy(list_data[idx])
                    this_dtrajs = np.copy(list_dtrajs[idx])
                    if list_obs_data is not None:
                        this_observables = []
                        for item in list_obs_data[idx]:
                            this_observables.append(np.copy(item))
                else:
                    this_training = np.append(this_training, list_data[idx], axis=0)
                    this_dtrajs = np.append(this_dtrajs, list_dtrajs[idx])
                    if list_obs_data is not None:
                        old_this_observables = this_observables
                        this_observables = []
                        for obs_count,item in enumerate(old_this_observables):
                            this_observables.append(np.append(item, list_obs_data[idx][obs_count], axis=0))

        this_n_frames = np.shape(this_dtrajs)[0]
        assert np.shape(this_training)[0] == this_n_frames
        for item in this_observables:
            assert np.shape(item)[0] == this_n_frames
        dtrajs_collected.append(this_dtrajs)
        data_collected.append(this_training)
        obs_data_collected.append(this_observables)
        estimators_collected.append(list_train_estimators)
        functions_collected.append(list_train_qfunctions)
        derivative_collected.append(derivative)
        logq_collected.append(logq)

    n_collected = len(dtrajs_collected)
    assert len(data_collected) == n_collected
    assert len(obs_data_collected) == n_collected
    assert len(estimators_collected) == n_collected
    assert len(functions_collected) == n_collected
    assert len(derivative_collected) == n_collected
    assert len(logq_collected) == n_collected

    for i in range(n_collected):
        add_estimator_to_list(dtrajs_collected[i], data_collected[i], observables, model, obs_data_collected[i], stationary_distributions, model_state, estimators_collected[i], functions_collected[i], derivative_collected[i], logq_collected[i])


    assert len(list_validation_estimators) == n_validations
    assert len(list_validation_qfunctions) == n_validations
    assert len(list_train_estimators) == n_validations
    assert len(list_train_qfunctions) == n_validations

    this_training = None
    this_dtrajs = None
    this_observables = None
    for idx in range(n_validations):
        if this_training is None:
            this_training = np.copy(list_data[idx])
            this_dtrajs = np.copy(list_dtrajs[idx])
            if list_obs_data is not None:
                this_observables = []
                for item in list_obs_data[idx]:
                    this_observables.append(np.copy(item))
        else:
            this_training = np.append(this_training, list_data[idx], axis=0)
            this_dtrajs = np.append(this_dtrajs, list_dtrajs[idx])
            if list_obs_data is not None:
                old_this_observables = this_observables
                this_observables = []
                for obs_count,item in enumerate(old_this_observables):
                    this_observables.append(np.append(item, list_obs_data[idx][obs_count], axis=0))

    all_the_indices = util.get_state_indices(this_dtrajs)
    complete_estimator = EstimatorsObject(this_training, all_the_indices, observables, model, obs_data=this_observables, stationary_distributions=stationary_distributions, model_state=model_state)

    t2 = time.time()
    print "Finished Initializing %d-Fold cross-validation in %f minutes" % (n_validations, (t2-t1)/60.)

    # go through and determine which hyper parameters need to be cycled
    # then perform a grid search for the ideal hyper parameters
    print "Beginning Grid Search of Hyper Parameters"

    if use_multi:
        qLock = threading.Lock()
    else:
        qLock = None

    iter_container = IterContainer(list_train_qfunctions, list_validation_qfunctions, all_kwargs, all_kwargs_printable, all_entries, all_entry_sizes, lock=qLock)

    if x0 is None:
        current_epsilons = complete_estimator.current_epsilons
    else:
        current_epsilons = x0

    if use_multi:
        all_threads = []
        for i in range(n_threads):
            all_threads.append(EstimateMulti(solver, current_epsilons, iter_container))

        all_check = [iter_container]
        for thrd in all_threads:
            thrd.start()
            all_check.append(thrd)

        while check_going(all_check):
            pass # wait until all objects register completion

    else:
        new_estimator = EstimateMulti(solver, current_epsilons, iter_container)
        new_estimator.run()

    best_hyper_params = iter_container.get_best()

    iteration_save_dir = "%s/best_params" % cwd
    if not os.path.isdir(iteration_save_dir):
        os.mkdir(iteration_save_dir)

    for entry in best_hyper_params:
        np.savetxt("%s/param_%s.dat" % (iteration_save_dir, entry), np.array(best_hyper_params[entry]))

    new_epsilons_cv = func_solver(complete_estimator, current_epsilons, **best_hyper_params)

    f_check.close()

    return new_epsilons_cv

def check_going(all_objects):
    still_going = False

    for thing in all_objects:
        still_going = still_going or thing.still_going

    return still_going

def recursive_add_dic_entries(all_dicts, current_dictionary, list_dictionary, ordered_entry_list, level):
    # list dictionary consists of entires and lists
    # will unwrap along the list axis
    this_entry = ordered_entry_list[level]
    for i in range(len(list_dictionary[this_entry])):
        new_dictionary = current_dictionary.copy()
        new_dictionary[this_entry] = list_dictionary[this_entry][i]

        if level == (len(ordered_entry_list) - 1):
            # terminate the recursion and add the current dictionary to the thing
            all_dicts.append(new_dictionary)
        else:
            #continue recursion
            recursive_add_dic_entries(all_dicts, new_dictionary, list_dictionary, ordered_entry_list, level+1)
