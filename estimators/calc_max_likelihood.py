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
import multiprocessing # only for the cross-validation function
import multiprocessing.managers as mpmanagers
import os
#import copy_reg
#import types

from estimators_class import EstimatorsObject
from optimizers import function_dictionary
from pyODEM.basic_functions import util

"""
# pickle is required for serializing the class methods
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)
"""

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

class GenerateEstimatorMulti(multiprocessing.Process):

    def __init__(self, observables, model, stationary_distributions, model_state, input_q):
        multiprocessing.Process.__init__(self)
        self.observables = observables
        self.model = model
        self.stationary_distributions = stationary_distributions
        self.model_state = model_state
        self.input_q = input_q
        self.solutions = []

    def run(self):
        while not self.input_q.empty():
            stuff = self.input_q.get()
            dtrajs = stuff[0]
            data = stuff[1]
            obs_data = stuff[2]
            derivative = stuff[3]
            logq = stuff[4]
            index = stuff[5]
            indices = util.get_state_indices(dtrajs)
            estimators = EstimatorsObject(data, indices, self.observables, self.model, obs_data=obs_data, stationary_distributions=self.stationary_distributions, model_state = self.model_state)
            Qfunction_epsilon = estimators.get_function(derivative, logq)
            self.solutions.append([estimators, Qfunction_epsilon, index])

class ListDataConstructors(object):
    def __init__(self, list_data, list_dtrajs, list_obs_data, n_validations, derivative, logq):
        if list_obs_data is None:
            # if list_obs_data, make a list of None
            list_obs_data = [None for i in range(n_validations)]

        self.list_train_estimators = []
        self.list_validation_estimators = []

        self.list_train_qfunctions = []
        self.list_validation_qfunctions = []

        self.inputs_collected = []
        self.estimators_collected = []
        self.functions_collected = []
        self.estimator_size = []

        self.n_validations = n_validations

        all_training = None
        all_dtrajs = None
        all_observables = None

        for validation_index in range(n_validations):
            # For each validation index: make the q_functions and EO objects
            this_training = None
            this_dtrajs = None
            this_observables = None
            all_training, all_dtrajs, all_observables = self._check_and_append_inputs(all_training, all_dtrajs, all_observables, list_data[validation_index], list_dtrajs[validation_index], list_obs_data[validation_index])
            for idx in range(n_validations):
                if idx == validation_index:
                    temp_params = self._convert_and_add_parameters_to_list(list_dtrajs[idx], list_data[idx], list_obs_data[idx], False, True, True)
                else:
                    this_training, this_dtrajs, this_observables = self._check_and_append_inputs(this_training, this_dtrajs, this_observables, list_data[idx], list_dtrajs[idx], list_obs_data[idx])

            this_n_frames = np.shape(this_dtrajs)[0]
            assert np.shape(this_training)[0] == this_n_frames
            for item in this_observables:
                assert np.shape(item)[0] == this_n_frames
            self._convert_and_add_parameters_to_list(this_dtrajs, this_training, this_observables, derivative, logq, False)

        self._convert_and_add_parameters_to_list(all_dtrajs, all_training, all_observables, derivative, logq, None)

    def get_queue(self):
        """ Get Queue sorted so larger estimators are selected first """

        generator_sync_manager = mpmanagers.SyncManager()
        generator_sync_manager.start()
        new_q = generator_sync_manager.Queue()

        sorted_args = np.argsort(np.array(self.estimator_size) * -1)
        print "Estimator Sizes: "
        print self.estimator_size
        for idx in sorted_args:
            new_q.put(self.inputs_collected[idx])

        return new_q, generator_sync_manager

    def add_estimators_to_list(self, results_q):
        num_found = len(results_q)
        print "Found %d/%d Estimators" % (num_found, (self.n_validations*2)+1)
        i_collected = []
        e_collected = []
        f_collected = []
        for stuff in results_q:
            i_collected.append(stuff[2])
            e_collected.append(stuff[0])
            f_collected.append(stuff[1])

        sorted_indices = np.argsort(i_collected)

        for idx in sorted_indices:
            index = i_collected[idx]
            this_estimator = e_collected[idx]
            this_function = f_collected[idx]

            if index < (self.n_validations*2)-1:
                self.estimators_collected[index].append(this_estimator)
                self.functions_collected[index].append(this_function)
            else:
                all_estimator = this_estimator
                all_function = this_function

        return self.list_train_estimators, self.list_validation_estimators, self.list_train_qfunctions, self.list_validation_qfunctions, all_estimator, all_function


    def _check_and_append_inputs(self, this_training, this_dtrajs, this_observables, data, dtraj, obs):
        if this_training is None:
            this_training = np.copy(data)
            this_dtrajs = np.copy(dtraj)
            if obs is not None:
                this_observables = []
                for item in obs:
                    this_observables.append(np.copy(item))
        else:
            this_training = np.append(this_training, data, axis=0)
            this_dtrajs = np.append(this_dtrajs, dtraj)
            if obs is not None:
                old_this_observables = this_observables
                this_observables = []
                for obs_count,item in enumerate(old_this_observables):
                    this_observables.append(np.append(item, obs[obs_count], axis=0))

        return this_training, this_dtrajs, this_observables

    def _convert_and_add_parameters_to_list(self, dtrajs, data, obs_data, derivative, logq, validation_thing):
        index = len(self.inputs_collected)
        temp_params =  [dtrajs, data, obs_data, derivative, logq, index]
        self.inputs_collected.append(temp_params)
        self.estimator_size.append(np.shape(dtrajs)[0])

        if validation_thing is None:
            self.estimators_collected.append(None)
            self.functions_collected.append(None)
        elif validation_thing:
            self.estimators_collected.append(self.list_validation_estimators)
            self.functions_collected.append(self.list_validation_qfunctions)
        else:
            self.estimators_collected.append(self.list_train_estimators)
            self.functions_collected.append(self.list_train_qfunctions)

        n_size = len(self.inputs_collected)
        assert len(self.estimators_collected) == n_size
        assert len(self.functions_collected) == n_size
        assert len(self.estimator_size) == n_size

class EstimateMulti(multiprocessing.Process):
    def __init__(self, solver, current_epsilons, iter_q, save_q, training_functions, validation_functions):
        multiprocessing.Process.__init__(self)
        self.solver = get_solver(solver)
        self.current_epsilons = current_epsilons
        self.iter_q = iter_q
        self.save_q = save_q
        self.training_functions = training_functions
        self.validation_functions = validation_functions

        self.still_going = False # True when the loop is running

    def run(self):
        self.still_going = True
        while not self.iter_q.empty():
            new_params = self.iter_q.get()
            kwargs = new_params[0]
            position = new_params[1]
            training_function = self.training_functions[position[1]]
            validation_function = self.validation_functions[position[1]]
            new_epsilons = self.solver(training_function, self.current_epsilons, **kwargs)
            this_score = validation_function(new_epsilons)
            print "Final Score: %f" % this_score
            self.save_q.put([this_score, position])
        self.still_going = False

        return

class IterContainer(object):
    """ Contains the parameters and solution for cross validation """
    def __init__(self, n_functions, all_kwargs, all_kwargs_printable, order_list, order_sizes):
        self.all_kwargs = all_kwargs
        self.all_kwargs_printable = all_kwargs_printable
        self.num_functions = n_functions
        self.num_params = len(all_kwargs)
        self.save_array = np.zeros((self.num_params, self.num_functions))
        self.save_complete = np.copy(self.save_array) - 1

        self.send_indices = []
        for i in range(self.num_params):
            for j in range(self.num_functions):
                self.send_indices.append([i,j])

        self.total_send = len(self.send_indices)
        self.current_index = 0
        self.still_going = True

        if self.total_send < 10:
            self.print_every = 1
        else:
            self.print_every = int(self.total_send / 10)
        print "Total of %d Optimizations Necessary" % self.total_send

    def get_queue(self):
        new_sync_manager = mpmanagers.SyncManager()
        new_sync_manager.start()
        new_q = new_sync_manager.Queue()
        for idx in range(self.total_send):
            send_indices = self.send_indices[idx]
            kwargs = self.all_kwargs[send_indices[0]]
            position = ()
            for jdx in send_indices:
                position += (jdx,)

            new_list = [kwargs, position]
            new_q.put(new_list)

        return new_q, new_sync_manager

    def save_queue(self, queue):
        count = 0
        while not queue.empty():
            stuff = queue.get()
            score = stuff[0]
            position = stuff[1]
            self.save(score, position)
            print self.save_array
            count += 1

        if count == self.total_send:
            print "Success!"
        else:
            print "Failure!"
        print "Completed %d/%d computations" % (count, self.total_send)

    def save(self, score, position):
        self.save_array[position] = score
        self.save_complete[position] = 1

    def reset_q(self):
        self.current_index = 0

    def get_best(self):
        if np.any(self.save_complete < 0):
            print "Warning: Many parameters were not saved. See the save_complete attribute for which ones."

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
    print_str = "Initializing Training and Validation Functions"
    if use_multi:
        print_str += " with %d threads" % n_threads
    print print_str
    t1 = time.time()
    data_constructor = ListDataConstructors(list_data, list_dtrajs, list_obs_data, n_validations, derivative, logq)
    input_q, server_manager = data_constructor.get_queue()

    if use_multi:
        generate_estimator_threads = []
        for i in range(n_threads):
            generate_estimator_threads.append(GenerateEstimatorMulti(observables, model, stationary_distributions, model_state, input_q))

        for thrd in generate_estimator_threads:
            thrd.start()

        for thrd in generate_estimator_threads:
            thrd.join()

    else:
        new_generator = GenerateEstimatorMulti(observables, model, stationary_distributions, model_state, input_q)
        generate_estimator_threads = [new_generator]
        new_generator.run()

    collected_solutions = []
    for thrd in generate_estimator_threads:
        for thing in thrd.solutions:
            collected_solutions.append(thing)

    server_manager.shutdown()
    list_train_estimators, list_validation_estimators, list_train_qfunctions, list_validation_qfunctions, complete_estimator, complete_qfunction = data_constructor.add_estimators_to_list(collected_solutions)

    t2 = time.time()
    print "Finished Initializing %d-Fold cross-validation in %f minutes" % (n_validations, (t2-t1)/60.)

    # go through and determine which hyper parameters need to be cycled
    # then perform a grid search for the ideal hyper parameters
    print "Beginning Grid Search of Hyper Parameters"

    iter_container = IterContainer(n_validations, all_kwargs, all_kwargs_printable, all_entries, all_entry_sizes)

    inputs_q, server_manager = iter_container.get_queue()
    results_q = multiprocessing.Queue()

    if x0 is None:
        current_epsilons = complete_estimator.current_epsilons
    else:
        current_epsilons = x0

    if use_multi:
        all_threads = []
        for i in range(n_threads):
            all_threads.append(EstimateMulti(solver, current_epsilons, inputs_q, results_q, list_train_qfunctions, list_validation_qfunctions))

        all_check = [iter_container]
        for thrd in all_threads:
            thrd.start()
            all_check.append(thrd)

        for thrd in all_threads:
            thrd.join()

        #while check_going(all_check):
        #    pass # wait until all objects register completion

    else:
        new_estimator = EstimateMulti(solver, current_epsilons, inputs_q, results_q, list_train_qfunctions, list_validation_qfunctions)
        new_estimator.run()

    server_manager.shutdown()
    iter_container.save_queue(results_q)
    best_hyper_params = iter_container.get_best()

    iteration_save_dir = "%s/best_params" % cwd
    if not os.path.isdir(iteration_save_dir):
        os.mkdir(iteration_save_dir)

    for entry in best_hyper_params:
        np.savetxt("%s/param_%s.dat" % (iteration_save_dir, entry), np.array(best_hyper_params[entry]))

    new_epsilons_cv = func_solver(complete_qfunction, current_epsilons, **best_hyper_params)

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
