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
from mpi4py import MPI
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

def max_likelihood_estimate(formatted_data, observables, model, solver="bfgs", logq=False, derivative=None, x0=None, kwargs={}, stationary_distributions=None):
    """ Optimizes model's paramters using a max likelihood method

    Args:
        formatted_data (list of dict): Each entry corresponds to a metastable
            state. The dictionary contains the "index" of the state, as well as
            the "data" for potential energy and "obs_result".
        observables (ExperimentalObservables): See object in
            pyODEM.observables.exp_observables.ExperimentalObservables
        model (ModelLoader/list): See object in the module
             pyODEM.model_loaders.X for the particular model.
        solver (str): Optimization procedures. Defaults to Simplex. Available
            methods include: simplex, anneal, cg, custom.
        logq (bool): Use the logarithmic Q functions. Default: False.
        derivative (bool): True if Q function returns a derivative. False if it
            does not. Default is None, automatically selected based upon the
            requested solver.
        x0 (array): Specify starting epsilons for optimization methods. Defaults
            to current epsilons from the model.
        kwargs (dictionary): Key word arguments passed to the solver.
        stationary_distributions (array): The probability of each state. The
            stationary_distributions[idx] corresponds to the idx, "index" in
            the formatted_data.

    Returns:
        eo (EstimatorsObject): Object that contains the data used for the
            computation and the results.

    """
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()

    all_indices = []
    all_data = []
    all_obs_data = []

    if stationary_distributions is None:
        this_stationary_distribution = None
    else:
        this_stationary_distribution = []

    for stuff in formatted_data:
        all_indices.append(stuff["index"])
        all_data.append(stuff["data"])
        all_obs_data.append(stuff["obs_result"])
        if this_stationary_distribution is not None:
            this_stationary_distribution.append(stationary_distributions[stuff["index"]])
            #np.append(this_stationary_distribution,stationary_distributions[stuff["index"]])
    if this_stationary_distribution is not None:
        this_stationary_distribution = np.array(this_stationary_distribution)

    derivative = ensure_derivative(derivative, solver)
    print "number of inputted data sets: %d" % len(all_data)
    eo = EstimatorsObject(all_indices, all_data, all_obs_data, observables, model, stationary_distributions=this_stationary_distribution)

    Qfunction_epsilon = eo.get_function(derivative, logq)
    comm.Barrier()

    eo.set_good_pill()
    if x0 is None:
        current_epsilons = eo.current_epsilons
    else:
        current_epsilons = x0

    # now parallelize
    if rank == 0:
        func_solver = get_solver(solver)

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
        eo.set_poison_pill() #activate the poison pill
        final = Qfunction_epsilon(new_epsilons)
    else:
        while(eo.get_pill()):
            # for rank != 0, return a boolean instead.
            Qfunction_epsilon(current_epsilons)
        new_epsilons = None

    comm.Barrier()

    new_epsilons = comm.bcast(new_epsilons, root=0)

    eo.set_poison_pill()
    #then return a new set of epsilons inside the EstimatorsObject
    eo.save_solutions(new_epsilons)
    return eo


def max_likelihood_estimate_serial(data, dtrajs, observables, model, obs_data=None, solver="bfgs", logq=False, derivative=None, x0=None, kwargs={}, stationary_distributions=None, model_state=None):
    """ Optimizes model's paramters using a max likelihood method

    Args:
        See pyfexd.estimators.estimators_class.EstimatorsObject for:
            data (array), observables (ExperimentalObservables),
            model (ModelLoader), obs_data(list) and
            stationary_distributions (list)
        dtrajs (array of int): Discrete trajectory index for each frame of data.
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
    if type(data) is list:
        # need to vstack data, dtrajs and obs_data
        # do checks to verify everything is the right shape
        nsets = len(data)
        if not len(dtrajs) == nsets:
            raise IOError("data and dtrajs must have same number of sets")
        if obs_data is None:
            if not len(obs_data) == nsets:
                raise IOError("data and obs_data must have same number of sets")
        sizes = []
        for thing in data:
            sizes.append(np.shape(thing)[0])
        if obs_data is not None:
            for i,first in enumerate(obs_data):
                for second in first:
                    if not np.shape(second)[0] == sizes[i]:
                        raise IOError("Shape of obs_data does not match shape of data")
        for i,thing in enumerate(dtrajs):
            if not np.shape(thing)[0] == sizes[i]:
                raise IOError("Shape of dtrajs does not match shape of data")

        # at this point, all checks passed and data is formatted as expected
        all_data = None
        all_obs_data = None
        all_dtrajs = None
        for thing in data:
            if all_data is None:
                all_data = np.copy(thing)
            else:
                all_data = np.append(all_data, thing, axis=0)
        nframes_total = np.shape(all_data)[0]
        try:
            assert np.sum(sizes) == nframes_total #sanity check its correct shape
        except:
            print sizes
            print nframes_total
            raise

        if obs_data is not None:
            for thing in obs_data:
                if all_obs_data is None:
                    all_obs_data = [obs_value for obs_value in thing]
                else:
                    all_obs_data = [np.append(all_obs_value, obs_value, axis=0) for all_obs_value,obs_value in zip(all_obs_data,thing)]

            for thing in all_obs_data:
                assert np.shape(thing)[0] == nframes_total # sanity check

        for thing in dtrajs:
            if all_dtrajs is None:
                all_dtrajs = np.copy(thing)
            else:
                all_dtrajs = np.append(all_dtrajs, thing)
        assert np.shape(all_dtrajs)[0] == nframes_total

    else:
        all_data = data
        all_obs_data = obs_data
        all_dtrajs = dtrajs

    derivative = ensure_derivative(derivative, solver)
    data_sets = util.get_state_indices(all_dtrajs)
    print "number of inputted data sets: %d" % len(data_sets)
    eo = EstimatorsObject(all_data, data_sets, observables, model, obs_data=all_obs_data, stationary_distributions=stationary_distributions, model_state=model_state)

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

            if index < (self.n_validations*2):
                self.estimators_collected[index].append(this_estimator)
                self.functions_collected[index].append(this_function)
            else:
                all_estimator = this_estimator
                all_function = this_function

        self._check_list_sizes()

        return self.list_train_estimators, self.list_validation_estimators, self.list_train_qfunctions, self.list_validation_qfunctions, all_estimator, all_function

    def _check_list_sizes(self):
        try:
            assert len(self.list_train_estimators) == self.n_validations
            assert len(self.list_validation_estimators) == self.n_validations
            assert len(self.list_train_qfunctions) == self.n_validations
            assert len(self.list_validation_qfunctions) == self.n_validations
        except:
            print len(self.list_train_estimators)
            print len(self.list_validation_estimators)
            print len(self.list_train_qfunctions)
            print len(self.list_validation_qfunctions)
            raise

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
    def __init__(self, solver, current_epsilons, iter_q, save_q, training_functions, validation_functions, total_computations):
        multiprocessing.Process.__init__(self)
        self.solver = get_solver(solver)
        self.current_epsilons = current_epsilons
        self.iter_q = iter_q
        self.save_q = save_q
        self.training_functions = training_functions
        self.validation_functions = validation_functions
        self.total_computations = total_computations
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
            #print "Final Score: %f" % this_score
            self.save_q.put([this_score, position])
            num_completed = self.save_q.qsize()
            print "Completed %d/%d computations" % (num_completed, self.total_computations)

        self.still_going = False

        return

class IterContainer(object):
    """ Contains the parameters and solution for cross validation """
    def __init__(self, n_functions, all_kwargs, all_kwargs_printable, order_list, order_sizes, all_coordinates):
        self.all_kwargs = all_kwargs
        self.all_kwargs_printable = all_kwargs_printable
        self.num_functions = n_functions
        self.num_params = len(all_kwargs)
        self.save_array = np.zeros((self.num_params, self.num_functions))
        self.save_complete = np.copy(self.save_array) - 1
        self.all_coordinates = all_coordinates
        array_shape = tuple(i for i in order_sizes)
        self.comparison_array = np.zeros(array_shape) - 1

        self.send_indices = []
        for i in range(self.num_params):
            for j in range(self.num_functions):
                self.send_indices.append([i,j])

        self.total_send = len(self.send_indices)

        all_scores = [] # bigger score means do first
        for idx in range(self.total_send):
            score = 0
            all_params = self.all_kwargs_printable[self.send_indices[idx][0]]
            if "gtol" in all_params:
                # smaller gtol means convergence takes longer
                score += (1. / all_params["gtol"])

            if "bounds" in all_params:
                score += all_params["bounds"]

            all_scores.append(score)

        stuff = -1. * np.array(all_scores)
        self.sorted_indices = np.argsort(stuff)

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
        for idx in self.sorted_indices:
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

    def _compute_score(self):
        if np.any(self.save_complete < 0):
            print "Warning: Many parameters were not saved. See the save_complete attribute for which ones."

        total_scores = np.sum(self.save_array, axis=1)
        return total_scores

    def get_plottable_array(self):
        total_scores = self._compute_score()
        for coord,score in zip(self.all_coordinates, total_scores):
            self.comparison_array[coord] = score

        if np.any(self.comparison_array < 0):
            print "Error: Some scores not properly set"

        return self.comparison_array

    def get_best(self):
        total_scores = self._compute_score()
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

        return self.all_kwargs[pos], self.all_kwargs_printable[pos]

def prepare_kwargs_dictionary(kwargs, epsilon_possible, bounds_function, gtol_possible, ftol_possible, current_epsilons, epsilon_function_type, verbose=False):
    list_kwargs = {}
    list_kwargs_printable = {}
    if epsilon_possible is not None:
        list_kwargs_printable["bounds"] = epsilon_possible
        temporary_bounds_list = []
        for eps_pos in epsilon_possible:
            new_bounds = bounds_function(eps_pos, current_epsilons, epsilon_info=epsilon_function_type)
            temporary_bounds_list.append(new_bounds)
        list_kwargs["bounds"] = temporary_bounds_list

    if gtol_possible is not None:
        list_kwargs_printable["gtol"] = gtol_possible
        list_kwargs["gtol"] = gtol_possible

    if ftol_possible is not None:
        list_kwargs_printable["ftol"] = ftol_possible
        list_kwargs["ftol"] = ftol_possible

    for thing in list_kwargs:
        if thing not in list_kwargs_printable:
            list_kwargs_printable[thing] = list_kwargs[thing]
        else:
            if not len(list_kwargs_printable[thing]) == len(list_kwargs[thing]):
                raise IOError("Option: %s. Size of list_kwargs_printable (%d) must be equal to list_kwargs (%d)" % (thing, len(list_kwargs_printable[thing]), len(list_kwargs[thing])))

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
    all_coordinates = []
    if len(all_entries) == 0:
        all_kwargs.append(kwargs)
    else:
        recursive_add_dic_entries(all_kwargs, kwargs, list_kwargs, all_entries, 0, all_coordinates=all_coordinates, current_coordinate=tuple() )
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

    return list_kwargs, list_kwargs_printable, all_kwargs, all_kwargs_printable, all_entries, all_entry_sizes, all_coordinates

def kfold_crossvalidation_max_likelihood(list_data, list_dtrajs, observables, model, list_obs_data=None, solver="bfgs", logq=False, derivative=None, x0=None, kwargs={}, epsilon_possible=None, bounds_function=util.bounds_simple, gtol_possible=None, ftol_possible=None,  stationary_distributions=None, model_state=None, checkpoint_dir=None, verbose=False, n_threads=1):

    derivative = ensure_derivative(derivative, solver)

    if n_threads == 1:
        use_multi = False
    else:
        if not n_threads > 0:
            raise IOError("n_threads must be between 1 - infinity")
        use_multi = True

    n_validations = len(list_data)
    cwd = os.getcwd()
    # Determine checkpoint file name and write/append to it
    if checkpoint_dir is None:
        checkpoint_dir = "%s/checkpoint_%d" % (cwd, time.time()*1000)

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    checkpoint_file = "%s/checkpoint.txt" % checkpoint_dir

    if os.path.isfile(checkpoint_file):
        f_check = open(checkpoint_file, "a")
        f_check.write("\n\n\n### Continuing Cross Validation ###")
    else:
        f_check = open(checkpoint_file, "w")
        f_check.write("### Starting Cross Validation ###")


    # Make all the eo objects and qfunctions
    print_str = "Initializing Training and Validation Functions"
    if use_multi:
        print_str += " with %d threads" % n_threads
    print print_str
    t1 = time.time()
    data_constructor = ListDataConstructors(list_data, list_dtrajs, list_obs_data, n_validations, derivative, logq)
    input_q, server_manager = data_constructor.get_queue()
    '''
    if use_multi:
        generate_estimator_threads = []
        for i in range(n_threads):
            generate_estimator_threads.append(GenerateEstimatorMulti(observables, model, stationary_distributions, model_state, input_q))

        for thrd in generate_estimator_threads:
            thrd.start()

        for thrd in generate_estimator_threads:
            thrd.join()

    else:
    '''
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

    # set the epsilons
    if x0 is None:
        current_epsilons = complete_estimator.current_epsilons
    else:
        current_epsilons = x0
        assert np.shape(x0)[0] == np.shape(complete_estimator.current_epsilons)[0]

    epsilon_function_type = complete_estimator.current_epsilon_function_types

    # go through and determine which hyper parameters need to be cycled
    # then perform a grid search for the ideal hyper parameters
    print "Beginning Grid Search of Hyper Parameters"
    list_kwargs, list_kwargs_printable, all_kwargs, all_kwargs_printable, all_entries, all_entry_sizes, all_coordinates = prepare_kwargs_dictionary(kwargs, epsilon_possible, bounds_function, gtol_possible, ftol_possible, current_epsilons, epsilon_function_type, verbose=verbose)

    iter_container = IterContainer(n_validations, all_kwargs, all_kwargs_printable, all_entries, all_entry_sizes, all_coordinates)

    f_temp = open("%s/entry_order" % checkpoint_dir, "w")
    for thing in all_entries:
        f_temp.write("%s\n" % thing)
    f_temp.close()
    for thing in list_kwargs_printable:
        np.savetxt("%s/kwarg_%s" % (checkpoint_dir, thing), list_kwargs_printable[thing] )

    inputs_q, server_manager = iter_container.get_queue()
    results_q = multiprocessing.Queue()

    if use_multi:
        all_threads = []
        for i in range(n_threads):
            all_threads.append(EstimateMulti(solver, current_epsilons, inputs_q, results_q, list_train_qfunctions, list_validation_qfunctions, iter_container.total_send))

        all_check = [iter_container]
        for thrd in all_threads:
            thrd.start()
            all_check.append(thrd)

        for thrd in all_threads:
            thrd.join()

        #while check_going(all_check):
        #    pass # wait until all objects register completion

    else:
        new_estimator = EstimateMulti(solver, current_epsilons, inputs_q, results_q, list_train_qfunctions, list_validation_qfunctions, iter_container.total_send)
        new_estimator.run()

    server_manager.shutdown()
    iter_container.save_queue(results_q)
    best_hyper_params, best_hyper_params_printable = iter_container.get_best()

    cross_validation_array = iter_container.get_plottable_array()
    try:
        np.savetxt("%s/cross_validation_array.dat" % checkpoint_dir, cross_validation_array)
    except:
        np.save("%s/cross_validation_array" % checkpoint_dir, cross_validation_array)

    try:
        f_save_printable = open("%s/param_%s.dat" % (checkpoint_dir, entry), "w")
        for entry in best_hyper_params_printable:
            f_save_printable.write("%s\n" % entry)
            f_save_printable.write("%s\n\n" % str(best_hyper_params_printable[entry]))
        f_save_printable.close()
    except:
        print "Failed to save best hyper params. The printable form is not easy to save"
        print best_hyper_params_printable

    func_solver = get_solver(solver)
    new_epsilons_cv = func_solver(complete_qfunction, current_epsilons, **best_hyper_params)

    f_check.close()
    complete_estimator.save_solutions(new_epsilons_cv)
    return complete_estimator, iter_container

def check_going(all_objects):
    still_going = False

    for thing in all_objects:
        still_going = still_going or thing.still_going

    return still_going

def recursive_add_dic_entries(all_dicts, current_dictionary, list_dictionary, ordered_entry_list, level, all_coordinates=None, current_coordinate=None):
    # list dictionary consists of entires and lists
    # will unwrap along the list axis
    this_entry = ordered_entry_list[level]
    for i in range(len(list_dictionary[this_entry])):
        new_dictionary = current_dictionary.copy()
        new_dictionary[this_entry] = list_dictionary[this_entry][i]

        if all_coordinates is not None:
            new_coordinate = current_coordinate + tuple()
            new_coordinate += (i,)
        else:
            new_coordinate = None

        if level == (len(ordered_entry_list) - 1):
            # terminate the recursion and add the current dictionary to the thing
            all_dicts.append(new_dictionary)
            if new_coordinate is not None:
                all_coordinates.append(new_coordinate)
        else:
            #continue recursion
            recursive_add_dic_entries(all_dicts, new_dictionary, list_dictionary, ordered_entry_list, level+1, all_coordinates=all_coordinates, current_coordinate=new_coordinate)
