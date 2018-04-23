""" Contains EstimatorsObject """

import numpy as np
import time
import mdtraj as md
import os
from mpi4py import MPI

np.seterr(over="raise")

class EstimatorsObject(object):
    """ Contains inputted formatted data and results from analysis

    This object contains the necessary data formatted appropriately, as well as
    the resultant Q functions and the results of any optimization routine.

    Attributes:
        new_epsilons (array of float): Optimized values of epsilons.
        old_epsilons (array of float): Starting values of epsilons.
        oldQ (float): Starting Q value.
        newQ (float): Optimized Q value.

    """
    def __init__(self, data_indices, data, expectation_observables, observables, model, stationary_distributions=None):
        """ Initialize object and process all the inputted data

        Args:
            data (array): First index is the frame, the other indices are the
                data for the frame. Should be the data loaded from
                model.load_data().
            data_sets (list of array): Each entry is an array with the frames
                corresponding to that equilibrium state.
            observables (ExperimentalObservables): See object in
                pyfexd.observables.exp_observables.ExperimentalObservables
            model (ModelLoader/list): See object in the module
                 pyfexd.model_loaders.X for the particular model.
            obs_data (list): Use if data set for computing observables is
                different from data for computing the energy. List contains
                arrays where each array-entry corresponds to the observable in
                the ExperimentalObservables object. Arrays are specified with
                first index corresponding to the frame and second index to the
                data. Default: Use the array specified in data for all
                observables.
            stationary_distributions (list of float): List of values for pi for
                each stationary distribution. Must be same size as data_sets.
                Default will compute the distribution based upon the weighting
                of each data_sets.
            model_state (list): List which model object to use when model is a
                list. Default None.
        """
        print "Initializing EstimatorsObject"
        t1 = time.time()
        # set MPI stuff:
        self.comm = MPI.COMM_WORLD

        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        #observables get useful stuff like value of beta
        self.number_equilibrium_states = len(data_indices)
        # check everything is the right size:
        if not len(data) == self.number_equilibrium_states:
            raise IOError("data and data_indices must have the same length")
        if not len(expectation_observables) == self.number_equilibrium_states:
            raise IOError("expecatation_observables and data_indices must have same length")
        self.expectation_observables = expectation_observables
        self.observables = observables
        self.observables.prep()

        self.Q_function, self.dQ_function = observables.get_q_functions()
        self.log_Q_function, self.dlog_Q_function = observables.get_log_q_functions()

        #calculate average value of all observables and associated functions
        self.epsilons_functions = []
        self.derivatives_functions = []


        self.h0 = []

        self.pi = np.zeros(self.number_equilibrium_states).astype(float)
        self.ni = np.zeros(self.number_equilibrium_states).astype(int)

        ####Format Inputs####
        self.model = model
        self.current_epsilons = model.get_epsilons() #assumes the first model is the one you want
        self.number_params = np.shape(self.current_epsilons)[0]

        #load data for each set, and compute energies and observations
        count = -1
        self.non_zero_states = []
        self.state_ham_functions = []#debugging
        for state_count in range(self.number_equilibrium_states):
            # first determine size of states, get the hepsilon functions
            count += 1
            use_data = data[state_count]
            num_in_set = np.shape(use_data)[0]
            if not num_in_set == 0:
                self.non_zero_states.append(state_count)
            epsilons_function, derivatives_function = self.model.get_potentials_epsilon(use_data)
            size_array = np.shape(epsilons_function(self.current_epsilons))[0]
            for test in derivatives_function(self.current_epsilons):
                assert np.shape(test)[0] == size_array

            # All thigns saved for later should go below here
            self.epsilons_functions.append(epsilons_function)
            self.derivatives_functions.append(derivatives_function)
            self.h0.append(epsilons_function(self.current_epsilons))

            self.ni[state_count] = num_in_set

        ##check the assertion. make sure everything same size
        for i in self.non_zero_states:
            #in future, should add debug flags to print out extra stuff
            #print "For state %d" % i
            #print np.shape(self.epsilons_functions[i](self.current_epsilons))[0]
            #print np.shape(self.h0[i])[0]
            assert np.shape(self.epsilons_functions[i](self.current_epsilons))[0] == np.shape(self.h0[i])[0]
            size = np.shape(self.epsilons_functions[i](self.current_epsilons))[0]
            for arrr in self.derivatives_functions[i](self.current_epsilons):
                assert np.shape(arrr)[0] == size

        ##number of observables
        self.num_observable = np.shape(self.expectation_observables[0])[0]
        self.pi =  np.array(self.pi).astype(float)
        self.pi /= np.sum(self.pi)

        if stationary_distributions is None:
            print "Determining Stationary Distribution Based on State Counts"
            # need to send eachother the total number of states
            if self.rank == 0:
                # receive from each thread:
                total = np.sum(self.ni)
                for i in range(1, self.size):
                    this_sum = self.comm.recv(source=i, tag=3)
                    total += this_sum

                # now send back the total
                for i in range(1, self.size):
                    self.comm.send(total, dest=i, tag=5)
            else:
                # send this threads total
                self.comm.send(np.sum(self.ni), dest=0, tag=3)
                # now get back the true total
                total = self.comm.recv(source=0, tag=5)

            # now compute the pi for each state
            self.pi = self.ni.astype(float) / float(total)
        else:
            print "Using Inputted Stationary Distribution"
            if np.shape(stationary_distributions)[0] == len(self.ni):
                print "Percent Difference of Selected Stationary Distribution from expected"
                total_approximate = self.ni.astype(float) / np.sum(self.ni.astype(float))
                diff = stationary_distributions - total_approximate
                print np.abs(diff/stationary_distributions)
                self.pi = stationary_distributions
            else:
                print "Input Stationry Distribution Dimensions = %d" % np.shape(stationary_distributions)[0]
                print "Number of Equilibrium States = %d" % len(self.ni)
                raise IOError("Inputted stationary distributions does match not number of equilibrium states.")

        print "THREAD %d PI: %s" % (self.rank, str(self.pi))
        ##Compute factors that don't depend on the re-weighting
        self.state_prefactors = []
        for i in range(self.number_equilibrium_states):
            state_prefactor = self.pi[i] * self.expectation_observables[i]
            self.state_prefactors.append(state_prefactor)

        t2 = time.time()
        total_time = (t2-t1) / 60.0
        print "Initializaiton Completed: %f Minutes" % total_time

        ##debugging options below
        self.count_Qcalls = 0
        self.count_hepsilon = 0
        self.count_dhepsilon = 0
        self.trace_Q_values= []
        self.trace_log_Q_values = []

        self.set_poison_pill() # default should be to not continue indefinitely

    def set_good_pill(self):
        self.pill = True

    def set_poison_pill(self):
        self.pill = False

    def set_pill(self, value):
        self.pill = value

    def get_pill(self):
        return self.pill

    def get_reweighted_observable_function(self):
        return self.calculate_observables_reweighted

    def calculate_observables_reweighted(self, epsilons):
        """ Calculates the observables using a set of epsilons

        Takes as input a new set of epsilons (model parameters). Calculates a
        new set of observables using self.observables and outputs the
        observables as an array.

        Args:
            epsilons (array of float): Model parameters

        Returns:
            next_observed (array of float): Values for all the observables.

        """

        #initiate value for observables:
        next_observed = np.zeros(self.num_observable)


        #add up all re-weighted terms for normalizaiton
        total_weight = 0.0
        #calculate re-weighting for all terms
        for i in self.non_zero_states:
            next_weight = np.sum(np.exp(self.epsilons_functions[i](epsilons) - self.h0[i])) / self.ni[i]
            next_observed += next_weight * self.state_prefactors[i]
            total_weight += next_weight * self.pi[i]

        #normalize so total re-weighted probability is = 1.0
        next_observed /= total_weight

        return next_observed

    def save_solutions(self, new_epsilons):
        """ Save the solutions and store the new and old Q """

        self.new_epsilons = new_epsilons
        self.oldQ = self.Qfunction_epsilon(self.current_epsilons)
        self.newQ = self.Qfunction_epsilon(self.new_epsilons)
        self.old_epsilons = self.current_epsilons

    def get_function(self, derivatives, logq):
        """ Returns the function of the specific type

        Args:
            derivatives (bool): If true, return a function that also computes
                the derivative.
            logq (bool): If true, return a function that computes using the
                logarithmic version of the statistical functions in observables.

        return:
            method: Computes the Q value.

        """

        if derivatives:
            if logq:
                func = self.derivatives_log_Qfunction_epsilon
            else:
                func = self.derivatives_Qfunction_epsilon
        else:
            if logq:
                func = self.log_Qfunction_epsilon
            else:
                func = self.Qfunction_epsilon

        return func

    def Qfunction_epsilon(self, epsilons):
        """ Compute the Q value.

        Args:
            epsilons (array of float): Model parameters

        return:
            float: Q value.

        """
        # send the values of the epsilons from rank=0 to all other processes
        epsilons = self.comm.bcast(epsilons, root=0)

        #initiate value for observables:
        next_observed, total_weight, boltzman_weights = self.get_reweights_norescale(epsilons)

        if self.rank == 0:
            total_observed = next_observed
            total_all_weights = total_weight
            for i in range(1, self.size):
                that_observed = self.comm.recv(source=i, tag=7)
                that_weight = self.comm.recv(source=i, tag=11)
                total_observed += that_observed
                total_all_weights += total_weight
            total_observed /= total_all_weights
            Q = -1.0 * self.Q_function(total_observed)
        else:
            self.comm.send(next_observed, dest=0, tag=7)
            self.comm.send(total_weight, dest=0, tag=11)
            Q = None
        #Minimization, so make maximal value a minimal value with a negative sign.
        Q = self.comm.bcast(Q, root=0)

        ##debug
        self.count_Qcalls += 1
        self.trace_Q_values.append(Q)

        return Q

    def log_Qfunction_epsilon(self, epsilons):
        """ Compute the -log(Q) value.

        Args:
            epsilons (array of float): Model parameters

        return:
            float: -log(Q) value.

        """
        epsilons = self.comm.bcast(epsilons, root=0)

        next_observed, total_weight, boltzman_weights = self.get_reweights_norescale(epsilons)

        if self.rank == 0:
            total_observed = next_observed
            total_all_weights = total_weight
            for i in range(1, self.size):
                that_observed = self.comm.recv(source=i, tag=7)
                that_weight = self.comm.recv(source=i, tag=11)
                total_observed += that_observed
                total_all_weights += total_weight
            total_observed /= total_all_weights
            Q = self.log_Q_function(next_observed)
        else:
            self.comm.send(next_observed, dest=0, tag=7)
            self.comm.send(total_weight, dest=0, tag=11)
            Q = None
        #Minimization, so make maximal value a minimal value with a negative sign.
        Q = self.comm.bcast(Q, root=0)

        #Minimization, so make maximal value a minimal value with a negative sign.
        #print epsilons

        ##debug
        self.count_Qcalls += 1
        self.trace_log_Q_values.append(Q)

        return Q

    def derivatives_Qfunction_epsilon(self, epsilons):
        """ Compute the Q value and its derivative.

        Args:
            epsilons (array of float): Model parameters

        return:
            float: Q value.
            array of float: dQ/dEpsilon

        """
        epsilons = self.comm.bcast(epsilons, root=0)

        next_observed, total_weight, boltzman_weights = self.get_reweights_norescale(epsilons)
        if self.rank == 0:
            total_observed = next_observed
            total_all_weights = total_weight
            for i in range(1, self.size):
                that_observed = self.comm.recv(source=i, tag=7)
                that_weight = self.comm.recv(source=i, tag=11)
                total_observed += that_observed
                total_all_weights += total_weight
            total_observed /= total_all_weights
            Q = self.Q_function(total_observed)
        else:
            self.comm.send(next_observed, dest=0, tag=7)
            self.comm.send(total_weight, dest=0, tag=11)
            Q = None
            total_all_weights = None

        Q = self.comm.bcast(Q, root=0)
        total_all_weights = self.comm.bcast(total_all_weights, root=0)

        derivative_observed_first, derivative_observed_second = self.get_derivative_pieces(epsilons, boltzman_weights)

        if self.rank == 0:
            for i in range(1, self.size):
                that_first = self.comm.recv(source=i, tag=13)
                that_second = self.comm.recv(source=i, tag=17)
                derivative_observed_first += that_first
                derivative_observed_second += that_second
            dQ_vector = []
            for j in range(self.number_params):
                derivative_observed = (derivative_observed_first[j]  - (total_observed * derivative_observed_second[j])) / total_all_weights
                dQ = self.dQ_function(next_observed, derivative_observed) * Q
                dQ_vector.append(dQ)

            dQ_vector = np.array(dQ_vector)

        else:
            self.comm.send(derivative_observed_first, dest=0, tag=13)
            self.comm.send(derivative_observed_second, dest=0, tag=17)

            dQ_vector = None

        dQ_vector = self.comm.bcast(dQ_vector, root=0)

        dQ_vector = -1. * np.array(dQ_vector)
        Q *= -1.

        self.trace_Q_values.append(Q)
        self.count_Qcalls += 1

        return Q, dQ_vector

    def derivatives_log_Qfunction_epsilon(self, epsilons):
        """ Compute the -log(Q) value and its derivative.

        Args:
            epsilons (array of float): Model parameters

        return:
            float: Q value.
            array of float: dQ/dEpsilon

        """
        epsilons = self.comm.bcast(epsilons, root=0)

        next_observed, total_weight, boltzman_weights = self.get_reweights_norescale(epsilons)

        if self.rank == 0:
            total_observed = next_observed
            total_all_weights = total_weight
            for i in range(1, self.size):
                that_observed = self.comm.recv(source=i, tag=7)
                that_weight = self.comm.recv(source=i, tag=11)
                total_observed += that_observed
                total_all_weights += total_weight
            total_observed /= total_all_weights
            Q = self.log_Q_function(total_observed)
        else:
            self.comm.send(next_observed, dest=0, tag=7)
            self.comm.send(total_weight, dest=0, tag=11)
            Q = None
            total_all_weights = None
        #Minimization, so make maximal value a minimal value with a negative sign.

        # broadcast the Q-value and total_all_weights to all threads
        Q = self.comm.bcast(Q, root=0)
        total_all_weights = self.comm.bcast(total_all_weights, root=0)

        # compute each individual piece
        derivative_observed_first, derivative_observed_second = self.get_derivative_pieces(epsilons, boltzman_weights)
        # then sum up the derivative pieces form each thread
        if self.rank == 0:
            for i in range(1, self.size):
                that_first = self.comm.recv(source=i, tag=13)
                that_second = self.comm.recv(source=i, tag=17)
                derivative_observed_first += that_first
                derivative_observed_second += that_second
            dQ_vector = []
            for j in range(self.number_params):
                derivative_observed = (derivative_observed_first[j]  - (total_observed * derivative_observed_second[j])) / total_all_weights
                dQ = self.dlog_Q_function(total_observed, derivative_observed)
                dQ_vector.append(dQ)

            dQ_vector = np.array(dQ_vector)
        else:
            self.comm.send(derivative_observed_first, dest=0, tag=13)
            self.comm.send(derivative_observed_second, dest=0, tag=17)

            dQ_vector = None

        dQ_vector = self.comm.bcast(dQ_vector, root=0)

        self.trace_log_Q_values.append(Q)
        self.count_Qcalls += 1

        # broadcast the pill:
        this_pill = self.comm.bcast(self.get_pill(), root=0)
        self.set_pill(this_pill)
        #print "%f   %f" % (Q, np.abs(np.max(dQ_vector)))

        return Q, dQ_vector

    def get_derivative_pieces(self, epsilons, boltzman_weights):
        derivative_observed_first = np.zeros((self.number_params, self.num_observable))
        derivative_observed_second = np.zeros((self.number_params, self.num_observable))
        for i in self.non_zero_states:
            deriv_function = self.derivatives_functions[i](epsilons)
            self.count_dhepsilon += 1
            next_weight_derivatives = np.sum(boltzman_weights[i] * deriv_function, axis=1) / self.ni[i]
            #next_weight_derivatives = np.expand_dims(next_weight_derivatives, axis=1)
            next_weight_derivatives = next_weight_derivatives[:,np.newaxis]
            derivative_observed_first += next_weight_derivatives * self.state_prefactors[i]
            derivative_observed_second += next_weight_derivatives * self.pi[i]

        return derivative_observed_first, derivative_observed_second

    def get_reweights_norescale(self, epsilons):
        #initialize final matrices
        next_observed = np.zeros(self.num_observable)

        #add up all re-weighted terms for normalizaiton
        total_weight = 0.0

        #Calculate the reweighting for all terms
        boltzman_weights = self.get_boltzman_weights(epsilons)

        for i in self.non_zero_states:
            next_weight = np.sum(boltzman_weights[i]) / self.ni[i]
            next_observed += next_weight * self.state_prefactors[i]
            total_weight += next_weight * self.pi[i]
        return next_observed, total_weight, boltzman_weights

    def get_reweights(self, epsilons):
        next_observed, total_weight, boltzman_weights = self.get_reweights_norescale(epsilons)

        next_observed /= total_weight

        return next_observed, total_weight, boltzman_weights

    def get_boltzman_weights(self, epsilons):
        #calculate the boltzman weights.
        #If OverflowExcpetion, recompute with shift to all values
        #shift is simply -max_val+K_Shift
        K_shift = 600
        try:
            boltzman_weights = [1 for i in range(self.number_equilibrium_states)]
            for i in self.non_zero_states:
                boltzman_wt = np.exp(self.epsilons_functions[i](epsilons) - self.h0[i])
                self.count_hepsilon += 1
                boltzman_weights[i] = boltzman_wt
        except:
            print "Exponential Function Failed"
            exponents = []
            boltzman_weights = []
            max_val = 0
            exponents = [0 for i in range(self.number_equilibrium_states)]
            for i in self.non_zero_states:
                exponent = self.epsilons_functions[i](epsilons) - self.h0[i]
                self.count_hepsilon += 1
                max_exponent = np.max(exponent)
                if max_exponent > max_val:
                    max_val = max_exponent
                exponents[i] = exponent
            boltzman_weights = [1 for i in range(self.number_equilibrium_states)]
            for i in range(self.number_equilibrium_states):
                boltzman_wt = np.exp(exponents[i] - max_val + K_shift)
                boltzman_weights[i] = boltzman_wt

        assert len(boltzman_weights) == self.number_equilibrium_states
        for idx in self.non_zero_states: # only check non-zero, rest okay
            state = boltzman_weights[idx]
            assert np.shape(state)[0] == self.ni[idx]
        return boltzman_weights

    def save_debug_files(self):
        """ save debug files

        file1: counts for functions calls.
        file2: trace of Q values during optmization

        """
        cwd = os.getcwd()
        print "Saving Debug files to %s" % cwd
        f = open("function_calls.dat", "w")
        f.write("Number of times Q was computed: %d\n" % self.count_Qcalls)
        f.write("Number of times the Hamiltonian was computed: %d\n" % self.count_hepsilon)
        f.write("Number of times the derivative of the Hamiltonian was computed: %d\n" % self.count_dhepsilon)
        f.write("Number of equilibrium states used: %d\n" % self.number_equilibrium_states)

        np.savetxt("trace_Q_values.dat", self.trace_Q_values)
        np.savetxt("trace_log_Q_values.dat", self.trace_log_Q_values)

        f.close()


class HamiltonianCalculator(object):
    def __init__(self, hamiltonian_list, derivative_list, number_params, size, state):
        self.hamiltonian_list = hamiltonian_list
        self.derivative_list = derivative_list
        assert len(self.hamiltonian_list) == len(self.derivative_list)
        self.num_functions = len(self.hamiltonian_list)
        self.number_params = number_params
        self.size = size
        self.state = state

        if self.num_functions == 0:
            self.epsilon_function = self._epsilon_zero
            self.derivatives_function = self._derivatives_zero
            self.derivatives_zero_list = [np.zeros(1) for i in range(self.number_params)]
        else:
            self.epsilon_function = self._epsilon_function
            self.derivatives_function = self._derivatives_function

    def _epsilon_zero(self, epsilons):
        return np.zeros(self.number_params).reshape((1,self.number_params))

    def _derivatives_zero(self, epsilons):
        return self.derivatives_zero_list

    def _epsilon_function(self,epsilons):
        except_count = 0
        total_array = np.copy(self.hamiltonian_list[0](epsilons))
        for idx in range(1, self.num_functions):
            this_array = self.hamiltonian_list[idx](epsilons)
            total_array = np.append(total_array, this_array, axis=0)

        return total_array

    def _derivatives_function(self,epsilons):
        #count = 0
        #except_count = 0
            #must duplicate list
            #if not duplicated, errors will result
            #derivative_list functions might return a stored list
            #List pass by reference, so python points to that list
            #when it appends, it appends to the internal list as well
            #this causes total_list to grow every function call
            #this is not a problem, with arrays
            #Arrays are pass by reference as well
            #But np.append duplicates the array automatically

        total_list = list(self.derivative_list[0](epsilons))
        for idx in range(1, self.num_functions):
            this_list = self.derivative_list[idx](epsilons)
            #count += np.shape(this_list[0])[0]
            for j in range(self.number_params):
                    total_list[j] = np.append(total_list[j], this_list[j], axis=0)


        return total_list


def save_error_files(diff, epsilons, h0, kshift):
    np.savetxt("ERROR_EXPONENTIAL_FUNCTION", diff)
    np.savetxt("ERROR_EPSILONS", epsilons)
    np.savetxt("ERROR_H0", h0)
    f = open("ERROR_LOG_EXPONENTIAL_FUNCTION", "w")
    f.write("\n\nCurrent Shift: %f\n" % kshift)
    f.close()
