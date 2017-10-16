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
import os

from estimators_class import EstimatorsObject
from optimizers import function_dictionary
from pyODEM.basic_functions import util

def get_functions(eo, derivative, solver, logq):
    if derivative is None:
        if solver in ["cg", "newton", "bfgs", "one"]:
            derivative = True
            print "Derivative"
        else:
            derivative = False
            print "No Derivative"
    Qfunction_epsilon = eo.get_function(derivative, logq)

    return Qfunction_epsilon

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

        solver (str): Optimization procedures. Defaults to Simplex. Available
            methods include: simplex, anneal, cg, custom.
        logq (bool): Use the logarithmic Q functions. Default: False.
        derivative (bool): True if Q function returns a derivative. False if it
            does not. Default is None, automatically selected based upon the
            requested solver.
        x0 (array): Specify starting epsilons for optimization methods. Defaults
            to current epsilons from the model.
        kwargs (dictionary): Key word arguments passed to the solver.

    Returns:
        eo (EstimatorsObject): Object that contains the data used for the
            computation and the results.

    """

    eo = EstimatorsObject(data, data_sets, observables, model, obs_data=obs_data, stationary_distributions=stationary_distributions, model_state=model_state)

    Qfunction_epsilon = get_functions(eo, derivative, solver, logq)
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


def kfold_crossvalidation_max_likelihood(list_data, list_dtrajs, observables, model, list_obs_data=None, solver="bfgs", logq=False, derivative=None, x0=None, kwargs={}, list_kwargs={}, list_kwargs_names={},  stationary_distributions=None, model_state=None, checkpoint_file=None, verbose=False):

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

    total_number_params = 1
    for entry in all_entries:
        total_number_params *= len(list_kwargs[entry])

    all_kwargs = []
    if len(all_entries) == 0:
        all_kwargs.append(kwargs)
    else:
        recursive_add_dic_entries(all_kwargs, kwargs, list_kwargs, all_entries, 0)
    assert len(all_kwargs) == total_number_params

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


    func_solver = get_solver(solver)

    for validation_index in range(n_validations):
        # For each validation index: make the q_functions and EO objects
        this_training = None
        this_dtrajs = None
        this_observables = None
        for idx in range(n_validations):
            if idx == validation_index:
                validation_indices = util.get_state_indices(list_dtrajs[idx])
                validation_estimators = EstimatorsObject(list_data[idx], validation_indices, observables, model, obs_data=list_obs_data[idx], stationary_distributions=stationary_distributions, model_state=model_state)
                Qfunction_epsilon = validation_estimators.get_function(False, True) # use the logq function only
                list_validation_estimators.append(validation_estimators)
                list_validation_qfunctions.append(Qfunction_epsilon)
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

        training_indices = util.get_state_indices(this_dtrajs)
        this_n_frames = np.shape(this_dtrajs)[0]
        assert np.shape(this_training)[0] == this_n_frames
        for item in this_observables:
            assert np.shape(item)[0] == this_n_frames
        training_estimator = EstimatorsObject(this_training, training_indices, observables, model, obs_data=this_observables, stationary_distributions=stationary_distributions, model_state=model_state)
        Qfunction_epsilon = get_functions(training_estimator, derivative, solver, logq)
        list_train_estimators.append(training_estimator)
        list_train_qfunctions.append(Qfunction_epsilon)

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

    if x0 is None:
        current_epsilons = complete_estimator.current_epsilons
    else:
        current_epsilons = x0

    best_hyper_params = None
    best_hyper_params_score = None
    n_param_combos = len(all_kwargs)
    print_every = n_param_combos / 10
    count = 0
    for ka in all_kwargs:
        if (count % n_param_combos) == 0:
            print "Completed %d/%d possible combinations" % (count, n_param_combos)
        this_score = 0
        for idx in range(n_validations):
            new_epsilons = func_solver(list_train_qfunctions[idx], current_epsilons, **ka)
            this_score += list_validation_qfunctions[idx](new_epsilons)

        if best_hyper_params_score is None:
            best_hyper_params_score = this_score
            best_hyper_params = ka
        else:
            if this_score < best_hyper_params_score:
                best_hyper_params_score = this_score
                best_hyper_params = ka
        count += 1

    if verbose:
        print "Dictionary %d" % idx
        this_str = ""
        for entry in best_hyper_params:
            this_str += "%s " % entry
            if isinstance(best_hyper_params,list):
                this_str += str(best_hyper_params[entry][0])
            else:
                this_str += str(best_hyper_params[entry])
            this_str += "   "
        print this_str

    iteration_save_dir = "%s/best_params" % cwd
    if not os.path.isdir(iteration_save_dir):
        os.mkdir(iteration_save_dir)

    for entry in best_hyper_params:
        np.savetxt("%s/param_%s.dat" % (iteration_save_dir, entry), np.array(best_hyper_params[entry]))

    new_epsilons_cv = func_solver(complete_estimator, current_epsilons, **best_hyper_params)

    f_check.close()

    return new_epsilons_cv


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
