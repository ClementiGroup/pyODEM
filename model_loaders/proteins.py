""" Loading data for Molecular Dynamics Simulations

Requires package: https://github.com/ajkluber/model_builder

For non-bonded calculations, require:
    PySPH @ https://pysph.readthedocs.io/en/latest/#

"""
import numpy as np
import mdtraj as md

# improt non-bonded methods for protein calculation
#from calc_nb_gromacs import check_if_dist_longer_cutoff, check_arr_sizes_are_equal, calc_nb_ene

from .calc_nb_gromacs import parse_and_return_relevant_parameters, parse_traj_neighbors, prep_compute_energy_fast, compute_energy_fast, compute_derivative_fast, order_epsilons_atm_types, compute_mixed_table, compute_mixed_derivatives_table, get_c6c12_matrix_noeps, convert_sigma_eps_to_c6c12

from .data_loaders import DataObjectList

from pyODEM.model_loaders import ModelLoader
try:
    import model_builder as mdb
    #import model_builder_mod as mdb
except:
    pass

class ProtoProtein(ModelLoader):
    """ subclass of ModelLoaders, super class of all Protein models

    The __init__ for most protein models have shared methods. This way, they can
    all call the same methods with modificaitons of their own later.

    """

    def __init__(self, ini_file_name):
        self.GAS_CONSTANT_KJ_MOL = 0.0083144621 #kJ/mol*k

        ##remove .ini suffix
        self.model, self.fittingopts = mdb.inputs.load_model(ini_file_name)

        if "fret_pairs" in self.fittingopts and not self.fittingopts["fret_pairs"] is None:
            self.fret_pairs = self.fittingopts["fret_pairs"]
        else:
            self.fret_pairs = [None]

        self.beta = 1.0 #set temperature

    def load_data(self,fname):
        """ Load a data file and format for later use

        For Proteins, it uses the self.pairs to load the pair-pair distances for
        every frame. This is the data format that would be used for computing
        the energy later.

        Args:
            fname (string): Name of a file to load.

        Return:
            Array (floats): First index is frame, second index is every
                pair in the order of pairs.

        """

        traj = md.load(fname, top=self.model.mapping.topology)
        data = md.compute_distances(traj, self.use_pairs, periodic=False)

        return data

    def load_data_from_traj(self, traj):
        """ Load a data file and format for later use

        For Proteins, it uses the self.pairs to load the pair-pair distances for
        every frame. This is the data format that would be used for computing
        the energy later.

        Args:
            traj (mdtraj.Trajectory): Traj file to read and parse.

        Return:
            Array (floats): First index is frame, second index is every
                pair in the order of pairs.

        """

        data = md.compute_distances(traj, self.use_pairs, periodic=False)

        return data

class Protein(ProtoProtein):
    """ Subclass for making a ModelLoader for a Protein Model

    Methods:
        See ModelLoader in pyODEM/super_model/ModelLoader

    """

    def __init__(self, ini_file_name):
        """ Initialize the Protein model, override superclass

        Args:
            ini_file_name (str): Name of a .ini file to load containing the
                model information.

        Attributes:
            See superclass for generic attributes.
            epsilons (array of float): Chosen from a specific list of tunable
                parameters from the .ini file.

        """

        ProtoProtein.__init__(self, ini_file_name)

        # get indices corresponding to epsilons to use
        # Assumes only parameters to change are pair interactions
        self.use_params = np.arange(len(self.model.Hamiltonian._pairs)) #assumes you use all pairs
        self.pairs = self.model.mapping._contact_pairs
        self.use_pairs = []
        for i in self.use_params: #only load relevant parameters
            self.use_pairs.append([self.pairs[i][0].index, self.pairs[i][1].index])

        self.epsilons = np.array(self.model.fitted_epsilons)
        function_type_names = self.model.fitted_function_types
        self.function_types = [] # 0 = always positive, 1 = can be negative
        for thing in function_type_names:
            if thing == "LJ12GAUSSIANTANH":
                self.function_types.append(1)
            else:
                self.function_types.append(0)
        assert len(self.function_types) == len(function_type_names)

    def get_potentials_epsilon(self, data):
        """ Return PotentialEnergy(epsilons)

        See superclass for full description of purpose. Override superclass.
        Potential Energy is easily calculated since for this model, all epsilons
        are linearly related to the potential energy.

        """

        #check to see if data is the expected shape for this analysis:
        if not np.shape(data)[1] == np.shape(self.use_params)[0]:
            err_str = "dimensions of data incompatible with number of parameters\n"
            err_str += "Second index must equal number of parameters \n"
            err_str += "data is: %s, number of parameters is: %d" %(str(np.shape(data)), len(self.use_params))
            raise IOError(err_str)

        #list of constant pre factors to each model epsilons
        constants_list = []
        constants_list_derivatives = []
        for i in self.use_params:
            constants_list.append(self.model.Hamiltonian._pairs[i].dVdeps(data[:,i]) )
            constants_list_derivatives.append(self.model.Hamiltonian._pairs[i].dVdeps(data[:,i])* -1. * self.beta  )

        constants_array = np.array(constants_list)
        constants_array_derivatives = np.array(constants_list_derivatives)
        #compute the function for the potential energy
        def hepsilon(epsilons):
            value = epsilons[:,np.newaxis] * constants_array
            total = np.sum(value, axis=0) * -1. * self.beta

            return total

        #compute the function for the derivative of the potential energy
        def dhepsilon(epsilons):
            #first index is corresponding epsilon, second index is frame

            return constants_list_derivatives

        return hepsilon, dhepsilon

    def save_model_parameters(self, parameters):

        for idx,param in enumerate(self.use_params):
            self.model.Hamiltonian._pairs[param].set_epsilon(parameters[idx])

class ProteinMultiTemperature(Protein):
    def load_data(self, fname, temperature):
        """ Load a data file and format for later use

        For Proteins, it uses the self.pairs to load the pair-pair distances for
        every frame. This is the data format that would be used for computing
        the energy later.

        Args:
            fname (string): Name of a file to load.
            temperature (float/np.ndarray): Temperature of this data set.

        Return:
            Array (floats): First index is frame, second index is the
                temperature followed by every pair in the order of pairs.

        *** BUG NOTE ***
        The results from hepsilon and dhepsilon function from
        get_potentials_epsilon will differ from the Protein method's results
        EVEN IF you use the same traj file. This is because mdtraj default
        output is numpy.float32, while the result of appending the temperature
        to the pair-distance output results in a numpy.float64. Comparison of
        the resultant data matrix would be exact suggesting there is no error.
        But when computing the potential energy using the
        model._pairs[i].dVdeps function will result in a different result, due
        to their differing precision on input.

        """

        traj = md.load(fname, top=self.model.mapping.topology)
        data = md.compute_distances(traj, self.use_pairs, periodic=False)
        temperature = np.ones((np.shape(data)[0],1)) * temperature

        all_data = np.append(temperature, data, axis=1)

        return all_data

    def get_potentials_epsilon(self, all_data):
        """ Return PotentialEnergy(epsilons)

        See superclass for full description of purpose. Override superclass.
        Potential Energy is easily calculated since for this model, all epsilons
        are linearly related to the potential energy.

        """
        these_temperatures = all_data[:,0]
        data = all_data[:,1:]

        all_betas = self._convert_temperature_to_beta(these_temperatures)

        #check to see if data is the expected shape for this analysis:
        if not np.shape(data)[1] == np.shape(self.use_params)[0]:
            err_str = "dimensions of data incompatible with number of parameters\n"
            err_str += "Second index must equal number of parameters \n"
            err_str += "data is: %s, number of parameters is: %d" %(str(np.shape(data)), len(self.use_params))
            raise IOError(err_str)

        #list of constant pre factors to each model epsilons
        constants_list = []
        constants_list_derivatives = []
        for i in self.use_params:
            constants_list.append(self.model.Hamiltonian._pairs[i].dVdeps(data[:,i]) )
            constants_list_derivatives.append(self.model.Hamiltonian._pairs[i].dVdeps(data[:,i])* -1. * all_betas)

        constants_array = np.array(constants_list)
        constants_array_derivatives = np.array(constants_list_derivatives)
        #compute the function for the potential energy
        def hepsilon(epsilons):
            value = epsilons[:,np.newaxis] * constants_array
            total = np.sum(value, axis=0) * -1. * all_betas

            return total

        #compute the function for the derivative of the potential energy
        def dhepsilon(epsilons):
            #first index is corresponding epsilon, second index is frame

            return constants_list_derivatives

        return hepsilon, dhepsilon


class ProteinNonLinear(Protein):
    """ Same as protein, except handles nonlinear H(epsilons)

    Just like the class Protein, except it handles a non-linear Hamiltonian
    function of epsilons. Thusly, only the get_potentials_epsilon is overridden.

    """

    def get_potentials_epsilon(self, data):
        num_frames = np.shape(data)[0]

        functions_list = []
        dfunctions_list = []
        for i in self.use_params:
            functions_list.append(self.model.Hamiltonian._pairs[i].get_V_epsilons(data[:,i]))
            dfunctions_list.append(self.model.Hamiltonian._pairs[i].get_dV_depsilons(data[:,i]))
        #compute the function for the potential energy
        def hepsilon(epsilons):
            total = np.zeros(np.shape(data)[0])
            for i in range(np.shape(epsilons)[0]):
                total += functions_list[i](epsilons[i]) * -1. * self.beta
            #total *= -1. * self.beta

            return total

        #compute the function for the derivative of the potential energy
        def dhepsilon(epsilons):
            #first index is corresponding epsilon, second index is frame
            scaled_beta = -1. * self.beta
            derivatives_list = [func(epsilons[idx])*scaled_beta for idx,func in enumerate(dfunctions_list)]

            return derivatives_list

        return hepsilon, dhepsilon

'''
        def hepsilon(epsilons):
            total_energy = np.zeros(np.shape(data)[0])
            for idx, param in enumerate(self.use_params):
                self.model.Hamiltonian._pairs[param].set_epsilon(epsilons[idx])
                energy = self.model.Hamiltonian._pairs[param].V(data[:,idx])
                total_energy += energy

            return total_energy * self.beta * -1.


        def dhepsilon(epsilons):
            total_energy = np.zeros(np.shape(data))
            for idx, param in enumerate(self.use_params):
                self.model.Hamiltonian._pairs[param].set_epsilon(epsilons[idx])

                energy = self.model.Hamiltonian._pairs[param].dVdeps(data[:,idx])
                total_energy[:,idx] += energy

            total_energy *= self.beta * -1.

            gradient = [total_energy[:,idx] for idx in range(np.shape(self.use_params)[0])]


            #assert len(gradient) == num_frames
            #for term in gradient:
            #    assert np.shape(term)[0] == np.shape(self.use_params)[0]
            return gradient


        return hepsilon, dhepsilon

'''

class ProteinAwsem(ProtoProtein):
    def __init__(self, ini_file_name):
        ProtoProtein.__init__(self, ini_file_name) #not implemented Need to do so in future, but will require modelbuilder implementation as well
        self.GAS_CONSTANT_KJ_MOL /= 4.184 #convert to kCal/mol*K
        self.use_frag = False
        self.use_gammas = False
        self.param_codes = [] #fragment memory potential
        self.epsilons_codes = []
        self.code_to_function = {"direct":self._get_gammas_potentials, "frag":self._get_frag_potentials, "water_mediated":self._get_water_mediated_potentials}
        self._num_frag_parameters = 0
        self._num_gamma_parameters = 0

    def make_epsilons_array(self, epsilons, code):
        if hasattr(self, "epsilons"):
            self.epsilons = np.append(self.epsilons, epsilons, axis=0)
        else:
            self.epsilons = np.array(epsilons)
        for i in range(len(epsilons)):
            self.epsilons_codes.append(code)
        assert hasattr(self, "epsilons")
        assert self.epsilons.ndim == 1

    def add_fragment_memory_params(self, param_path, mem_file, max_frag_length=9, cycle=True, fragment_memory_scale=0.1):
        """ Add fragment memory interactions for fitting """

        self.param_codes.append("frag")
        self.model.add_fragment_memory(param_path, mem_file, max_frag_length=max_frag_length, cycle=cycle, fragment_memory_scale=fragment_memory_scale)


        self.frag_gammas = []
        for potential in self.model.Hamiltonian.fragment_potentials:
            self.frag_gammas.append(potential.weight)
            self._num_frag_parameters += 1
        self.make_epsilons_array(self.frag_gammas, "frag")

        self.frag_scale = self.model.Hamiltonian.fragment_memory_scale


    def add_contact_params(self, water_mediated=False):
        """ Add direct contact interactions for fitting

        Only uses the gammas that are present in the model. It has to assign
        each interaction to a corresponding gamma, as well as collect epsilon
        values(gammas) that exist. Code is weird, as its more memory intensive
        than it needs to be so that sorting things can be completed in 2N+210
        time. Where N is the number of direct potentials in the model, and 210
        is the number of possible unique gammas it has to sort through. N~R**2,
        where R is number of residues so minimizing the number of times it goes
        through each list.

        Args:
            water_mediated (bool): True will also add the water mediated gammas
            to the end of the epsilons list. Default False.

        Attributes:
            use_pairs (list of int): List contains atom indices.
            gamma (array of float): 20x20 gamma-matrix. gm(i,j) = gm(j,i).
            gamma_indices (array of int): Nx2 array of gamma-matrix indices for
                each parameter.
            use_indices (list of int): Has len of X. Where X is number of unique
                gammas. Values are gamma indices used.
            use_params (list of float): Has len of X. Values are gamma values
                from the gamma-matrix corresponding to indices in
                use_indices.
            param_assignment (list of int): Has len of X, values are the index
                for each interaction's corresponding gamma in the use_params
                list.
            param_assigned_indices (list of int): Has len X,
                param_assigned_indices[i] = [indices], where indices are the
                index of the N parameters with the gamma indices =
                use_indices[i]

        """
        self.param_codes.append("direct")
        self.water_mediated_used = water_mediated
        if water_mediated:
            self.param_codes.append("water_mediated")

        # get indices corresponding to epsilons to use
        # Assumes only parameters to change are pair interactions
        self.use_pairs = []
        array_pairs = self.model.Hamiltonian._contact_pairs
        for i in range(np.shape(array_pairs)[0]):
             self.use_pairs.append(array_pairs[i,:])

        self.gamma_indices = self.model.Hamiltonian._contact_gamma_idxs
        self.direct_matrix = self.model.Hamiltonian.gamma_direct
        self.water_matrix = self.model.Hamiltonian.gamma_water
        self.protein_matrix = self.model.Hamiltonian.gamma_protein
        #go through gamma, append the parameters to use_params
        self.use_indices = []
        self.direct_gammas = []
        if water_mediated:
            self.water_gammas = []
            self.protein_gammas = []
        check_index_array = np.zeros((20,20))
        check_index_assignment = []
        #go through list, mark all positions where a potential is found
        for i in range(np.shape(self.gamma_indices)[0]):
            idx = self.gamma_indices[i,0]
            jdx = self.gamma_indices[i,1]
            check_index_array[idx,jdx] = 1 #set to 1 if found
            check_index_assignment.append((idx*20) + jdx) #convert 2-d to 1-d value

        #save index if check(i,j) = 1 or check(j, i) = 1
        count = 0
        check_index_conversion_array = np.zeros(400).astype(int)
        check_index_conversion_array -= 1
        for i in range(20): #this potential is always 20x20
            for j in np.arange(i,20):
                if check_index_array[i,j] == 1 or check_index_array[j,i] == 1:
                    self.use_indices.append([i,j])
                    self.direct_gammas.append(self.direct_matrix[i,j])
                    if water_mediated:
                        self.water_gammas.append(self.water_matrix[i,j])
                        self.protein_gammas.append(self.protein_matrix[i,j])
                    coord1 = (i*20) + j
                    coord2 = (j*20) + i
                    check_index_conversion_array[coord1] = count
                    check_index_conversion_array[coord2] = count
                    count += 1

        #Now we have which sets of gamma are indeed used.
        #Next, assign each interaction to each parameter, for later
        self.param_assignment = [] #each potential has value of parameter index
        self.param_assigned_indices = [[] for i in range(len(self.use_indices))]
        for i in range(len(check_index_assignment)):
            param_idx = check_index_conversion_array[check_index_assignment[i]]
            self.param_assignment.append(param_idx)
            self.param_assigned_indices[param_idx].append(i)

        self.make_epsilons_array(self.direct_gammas, "direct") #self.epsilons expected in places
        if water_mediated:
            self.make_epsilons_array(self.water_gammas, "water_mediated-water")
            self.make_epsilons_array(self.protein_gammas, "water_mediated-protein")
        ##### assertion checks #####
        # Check consistency of param_assignment and param_assigned_indices
        for idx, lst in enumerate(self.param_assigned_indices):
            for index in lst:
                assert self.param_assignment[index] == idx

        #Check consistent number of parameters
        assert len(self.param_assigned_indices) == len(self.direct_gammas)
        self._num_gamma_parameters = len(self.direct_gammas)

    def load_data(self,fname):
        """ Load a data file and format for later use

        For Proteins, it uses the self.pairs to load the pair-pair distances for
        every frame. This is the data format that would be used for computing
        the energy later.

        Args:
            fname (string): Name of a file to load.

        Return:
            traj (mdtraj.trajectory): Trajectory object in 3-bead
                representation. Will be converted to all-atom later.

        """

        #remarkably, traj object is indexable like an array
        #it is however only a 1-D array
        traj = md.load(fname, top=self.model.mapping.topology)

        return traj

    def save_gamma_parameters(self, new_eps, directory, write=False):
        assert np.shape(new_eps)[0] == np.shape(self.epsilons)[0]
        count_direct = 0
        count_water = 0
        count_protein = 0
        gamma_matrix = self.model.Hamiltonian.gamma_direct
        water_matrix = self.model.Hamiltonian.gamma_water
        protein_matrix = self.model.Hamiltonian.gamma_protein
        for i in range(np.shape(new_eps)[0]):
            if self.epsilons_codes[i] == "direct":
                idx = self.use_indices[count_direct]
                gamma_matrix[idx[0], idx[1]] = new_eps[i]
                gamma_matrix[idx[1], idx[0]] = new_eps[i]
                count_direct += 1
            elif self.epsilons_codes[i] == "water_mediated-water":
                idx = self.use_indices[count_water]
                water_matrix[idx[0], idx[1]] = new_eps[i]
                water_matrix[idx[1], idx[0]] = new_eps[i]
                count_water += 1
            elif self.epsilons_codes[i] == "water_mediated-protein":
                idx = self.use_indices[count_protein]
                protein_matrix[idx[0], idx[1]] = new_eps[i]
                protein_matrix[idx[1], idx[0]] = new_eps[i]
                count_protein += 1

        assert count_direct == self._num_gamma_parameters
        if self.water_mediated_used:
            assert count_water == self._num_gamma_parameters
            assert count_protein == self._num_gamma_parameters

        for i in range(20):
            for j in range(i+1, 20):
                assert gamma_matrix[i,j] == gamma_matrix[j,i]
                assert water_matrix[i,j] == water_matrix[j,i]
                assert protein_matrix[i,j] == protein_matrix[j,i]
        if write:
            self.model.write_new_gammas(directory)

    def save_fragment_memory_parameters(self, new_eps, name, directory, write=False):
        assert np.shape(new_eps)[0] == np.shape(self.epsilons)[0]
        count = 0
        for i in range(np.shape(new_eps)[0]):
            if self.epsilons_codes[i] == "frag":
                self.model.Hamiltonian.fragment_potentials[i].weight = new_eps[i]
                count += 1
        assert count == self._num_frag_parameters

        if write:
            self.model.write_new_fragment_memory(directory, name)

    def save_debug_files(self, old_eps, new_eps):
        f = open("debug_used_parameters.txt", "w")
        for index in self.use_indices:
            f.write(self.model.Hamiltonian.gamma_residues[index[0]])
            f.write(" - ")
            f.write(self.model.Hamiltonian.gamma_residues[index[1]])
            f.write("\n")
        f.close()
        np.savetxt("debug_old_eps.dat", old_eps)
        np.savetxt("debug_new_eps.dat", new_eps)

        diff = new_eps-old_eps
        np.savetxt("debug_diff_eps.dat", diff)
        sort_idxs = np.argsort(np.abs(diff))
        wrt_str = ""
        for idx in sort_idxs:
            res1 = self.model.Hamiltonian.gamma_residues[self.use_indices[idx][0]]
            res2 = self.model.Hamiltonian.gamma_residues[self.use_indices[idx][1]]
            this_str =  "%s - %s   %f\n" %(res1, res2, diff[idx])
            wrt_str = this_str + wrt_str

        f = open("debug_sorted_parameters.txt", "w")
        f.write(wrt_str)
        f.close()

    def _get_gammas_potentials(self, data):
        #for potentials, 1st index is frame, 2nd index is potential function
        constants_list = []
        constants_list_derivatives = []
        potentials = self.model.Hamiltonian.calculate_direct_energy(data, total=False, dgamma=True)
        assert np.shape(potentials)[1] == len(self.param_assignment)

        for indices,param in zip(self.param_assigned_indices, self.direct_gammas):
            try:
                assert param != 0 # Temporary check as we do  divide by zero later
            except:
                print(param)
                print(indices)
                raise
            constant_value = np.sum(potentials[:,indices], axis=1)
            constants_list.append(constant_value)
            constants_list_derivatives.append(constant_value * -1. * self.beta)
            self._check_code_potential_assignment.append("direct")
        return constants_list, constants_list_derivatives

    def _get_water_mediated_potentials(self, data):
        #for potentails, 1st index is frame, 2nd index is potential function
        constants_list = []
        constants_list_derivatives = []
        water_mediated, protein_mediated = self.model.Hamiltonian.calculate_water_energy(data, total=False, split=True, dgamma=True)
        assert np.shape(water_mediated)[1] == len(self.param_assignment)
        assert np.shape(protein_mediated)[1] == len(self.param_assignment)

        for indices,param in zip(self.param_assigned_indices, self.water_gammas):
            try:
                assert param != 0 # Temporary check as we do  divide by zero later
            except:
                print(param)
                print(indices)
                raise
            constant_value = np.sum(water_mediated[:,indices], axis=1)
            constants_list.append(constant_value)
            constants_list_derivatives.append(constant_value * -1. * self.beta)
            self._check_code_potential_assignment.append("water_mediated-water")

        for indices,param in zip(self.param_assigned_indices, self.protein_gammas):
            try:
                assert param != 0 # Temporary check as we do  divide by zero later
            except:
                print(param)
                print(indices)
                raise
            constant_value = np.sum(protein_mediated[:,indices], axis=1)
            constants_list.append(constant_value)
            constants_list_derivatives.append(constant_value * -1. * self.beta)
            self._check_code_potential_assignment.append("water_mediated-protein")

        return constants_list, constants_list_derivatives

    def _get_frag_potentials(self, data):
        #for potentials, 1st index is frame, 2nd index is potential function
        constants_list = []
        constants_list_derivatives = []
        potentials = self.model.Hamiltonian.calculate_fragment_memory_potential(data, total=False, dgamma=True)
        assert np.shape(self.frag_gammas)[0] == np.shape(potentials)[0]
        for idx in range(np.shape(potentials)[0]):
            rescale_constants = potentials[idx,:]
            constants_list.append(rescale_constants)
            constants_list_derivatives.append(rescale_constants * -1. * self.beta)
            self._check_code_potential_assignment.append("frag")
        return constants_list, constants_list_derivatives

    def get_potentials_epsilon(self, data):
        """ Return PotentialEnergy(epsilons)

        See superclass for full description of purpose.
        Override superclass. Potential Energy is easily calculated since for
        this model, all epsilons are linearly related to the potential energy.

        """

        self._check_code_potential_assignment = []

        constants_list = []
        constants_list_derivatives = []

        if len(self.param_codes) == 0:
            raise IOError("No Parameters set for fitting. See Documentation for optional parameters to fit")

        #Depending on which flag is on, it will compose constants_list and constants_list_derivatives
        for code in self.param_codes:
            consts, dconsts = self.code_to_function[code](data)
            for v in consts:
                constants_list.append(v)
            for dv in dconsts:
                constants_list_derivatives.append(dv)

        assert len(self._check_code_potential_assignment) == len(self.epsilons_codes)
        for i in range(len(self.epsilons_codes)):
            try:
                assert self._check_code_potential_assignment[i] == self.epsilons_codes[i]
            except:
                print(self._check_code_potential_assignment[i])
                print(self.epsilons_codes[i])
                raise

        #compute the function for the potential energy
        def hepsilon(epsilons):
            total = np.zeros(data.n_frames)
            for i in range(np.shape(epsilons)[0]):
                value = epsilons[i]*constants_list[i]
                total += value * -1. * self.beta

            return total

        #compute the function for the derivative of the potential energy
        def dhepsilon(epsilons):
            #first index is corresponding epsilon, second index is frame

            return constants_list_derivatives

        return hepsilon, dhepsilon


class ProteinNonBonded(ModelLoader):
    """ subclass of ProtoProtein, includes non-bonded option

    Use this loader for computing the non-bonded potential energy as well as the
    native potential energy.

    Makes heavy use of the calc_nb_gromacs.py package.

    """

    def __init__(self, topf):
        """ Initialize the ProteinNonBonded using a GROMACS .top file

        Uses the parse_and_return_relevant_parameters() from calc_nb_gromacs.py in order to parse a GROMACS .top file. This method assumes the .top file parameterizes a Calpha model.

        The Calpha types are read in the order of the .top file and assumes a
        LJ10-12 format. The mixing rule is assumed to be of type 1, meaning the
        C6, C12 are in the .top file, resulting in the C6 and C12 values being
        converted to sigma and epsilons on initilization of this class. The
        epsilon are what is being optimized while the sigmas are treated as
        constants.

        It is also assumed that the gaussian pairwise interactions are used. In
        doing so only the first parameter is treated as the epsilon and that is
        the parameter being optimzied while all other parameters are held
        constant for pairwise interactions. These are treated as and often
        referred to as "native contacts".

        self.dict_atm_types_extended: example 'CAY': [18, 0.0052734375, 0.001098632812]
        self.dict_atm_types: example 'CAY': [18, 1.0, 0.0, 'A', 0.0052734375, 0.001098632812]
        Args:
            topf (str): Path to the a GROMACS .top file.

        Attributes:
            epsilons (array): A len(E) array where the first X entries refer to
                the X unique Calpah types, and the remaining entries refer to
                the G epsions from the gaussian native contacts.

        """
        print("Initializing a Protein non-bonded interactions model")

        self.GAS_CONSTANT_KJ_MOL = 0.0083144621 #kJ/mol*k
        self.dict_atm_types_extended, self.dict_atm_types, self.numeric_atmtyp, self.pairsidx_ps, self.all_ps_pairs, self.pot_type1_, self.pot_type2_, self.parms_mt, self.parms2_, self.nrexcl = parse_and_return_relevant_parameters(topf)

        # get nonbonded epsilons
        self.n_atom_types = len(self.dict_atm_types)
        self.epsilons_atm_types, self.sigmas_atm_types = order_epsilons_atm_types(self.dict_atm_types, len(self.dict_atm_types))
        self.sigma_params_matrix = get_c6c12_matrix_noeps(self.sigmas_atm_types, [1])
        # get pairwise epsilons
        self.epsilons_pairs = []
        for thing in self.parms2_:
            self.epsilons_pairs.append(thing[0])

        self._epsilons = np.append(self.epsilons_atm_types, self.epsilons_pairs)
        self.n_original_epsilons = np.shape(self._epsilons)[0]

        self.group_indices = None
        self.parameter_indices = None

    @property
    def epsilons(self):
        return self.get_epsilons()

    def get_epsilons(self):
        if self.group_indices is None:
            return self._epsilons
        else:
            # there are group_indices
            epsilons = np.zeros(self.n_groups)
            for idx in range(self.n_groups):
                epsilons[idx] = self._epsilons[self.parameter_indices[idx][0]]
            return epsilons

    def reconstruct_epsilons(self, epsilons):
        eps = np.zeros(self.n_original_epsilons)
        assert np.shape(epsilons)[0] == self.n_groups
        for grp_idx, group in enumerate(self.parameter_indices):
            for param_idx in group:
                eps[param_idx] = epsilons[grp_idx]

        return eps

    def group_epsilons(self, group_indices):
        """ Assign each epsilon to a group that varies the values together

        Args:
            group_indices (list or np.ndarray): Length E for E self._epsilon
                values. Integers denote group index and must vary from 0 to N-1
                for N epsilon groups

        """

        if np.shape(group_indices)[0] != np.shape(self._epsilons)[0]:
            raise IOError("group_indexes must be same length as self._epsilons, got %d and %d respectively" % (np.shape(group_indices)[0], np.shape(self._epsilons)[0]))

        n_groups = int(np.max(group_indices) + 1)

        self.group_indices = np.array(group_indices).astype(int)
        self.n_groups = n_groups
        self.parameter_indices = [np.zeros(0).astype(int) for i in range(self.n_groups)]

        for idx,group in enumerate(self.group_indices):
            self.parameter_indices[group] = np.append(self.parameter_indices[group], [idx])

        for idx,group in enumerate(self.parameter_indices):
            if np.shape(group)[0] == 0:
                raise IOError("Group index %d is missing" % idx)

        for idx,group in enumerate(self.parameter_indices):
            starting_value = self._epsilons[group[0]]
            for eps_idx in group:
                if self._epsilons[eps_idx] != starting_value:
                    raise IOError("For group %d, Epsilon index %d differs from the group value of %f" % (idx, eps_idx, starting_value))

        # getting to this point means everything is in order

    def load_data(self, traj):
        """ Parse a traj object and return a DataObjectList

        The trajectory of N frames is parsed, and four lists of length N are
        produced. First, a list  of the nearest neighbors within a cutoff at
        each step of the trajectory is produced. The nearest neighbor list is
        length M, which can vary significantly from frame to frame. Second, a
        U/eps list of length M is generated for each step of the trajectory,
        where the values are the corresponding potential energy / epsilon for
        each pair in the nearest neighbors for each trajectory step. This
        process is done seperately for the pairwise interactions and nonbonded
        atom-type interactions (two lists each, total of four).

        In doing so, the size of the data passed to the get_potentials_epsilon()
        function scales as O(R), where R is the number of residues in the
        protein. Furthermore, the computation time is reduced significantly as
        the distances and potentials do not need to be recomputed fully, and the
        problem becomes a simple array multiplicatin problem.

        Args:
            traj (mdtraj.Trajectory): A mdtraj trajectory object.

        Returns:
            parsed_data (DataObjectList): An object that can be indexed like an
                array and contains the neighbor list and U/eps list.
        """
        all_nl_ps, all_nl_atmtyp_w_excl, parms1, pot_type1, parms2, pot_type2, rcut2 = parse_traj_neighbors(traj, self.numeric_atmtyp, self.pairsidx_ps, self.all_ps_pairs, self.pot_type1_, self.pot_type2_, self.parms_mt, self.parms2_, self.nrexcl, nstlist=1)

        all_nonbonded_eps_idxs, all_nonbonded_factors, all_pairwise_eps_idx, all_pairwise_factors = prep_compute_energy_fast(traj, all_nl_ps, all_nl_atmtyp_w_excl, self.numeric_atmtyp, self.pairsidx_ps, self.sigma_params_matrix, self.parms2_, self.pot_type1_,self.pot_type2_, rcut2)

        parsed_data = DataObjectList([all_nonbonded_eps_idxs, all_nonbonded_factors, all_pairwise_eps_idx, all_pairwise_factors])

        return parsed_data

    def _read_out_lists(self, data):
        """ Internal method for reading results from load_data() """
        list_of_lists = data.list_of_lists
        all_nonbonded_eps_idxs = list_of_lists[0]
        all_nonbonded_factors = list_of_lists[1]
        all_pairwise_eps_idx = list_of_lists[2]
        all_pairwise_factors = list_of_lists[3]

        return all_nonbonded_eps_idxs, all_nonbonded_factors, all_pairwise_eps_idx, all_pairwise_factors

    def get_potentials_epsilon(self, data):
        """ Generate the hepsilon and dhepsilon functions.

        The hepsilon(epsilons) and dhepsilon(epsilons) functions have to first
        sort the epsilons and generate the appropriate matrix of nonbonded
        epsilon combinations and list of native gaussian epsilons.

        For the hepsilon() function, this entails generating the separate
        epsilon lists and passing it to the compute_energy_fast() method. For
        the dhepsilon() function, this entails pre-computing the native gaussian
        derivatives (which do not change for varying epsilons) and using the
        compute_derivative_fast() method for the Calpha nonbonded atom-type
        functions.

        Args:
            data (DataObjectList):

        Returns:
            hepsilon (function): A length N array where each entry is the
                potential energy for the n'th frame.
            dhepsilon (function): A length E list of length N arrays. Compute
                the derivative with respect to the e'th epsilons parameter for
                the n'th frame.

        """
        all_nonbonded_eps_idxs, all_nonbonded_factors, all_pairwise_eps_idx, all_pairwise_factors = self._read_out_lists(data)

        # pre-compute derivatives (constants) for the pairwise interactions
        # Update with the nonbonded derivatives with each function call

        n_frames = len(all_nonbonded_eps_idxs)
        pairwise_epsilons = self._epsilons[self.n_atom_types:]
        n_pairwise_eps = np.shape(pairwise_epsilons)[0]
        all_pairwise_derivatives = [np.zeros(n_frames) for j in range(n_pairwise_eps)]

        for i_frame in range(n_frames):
            pairwise_idxs = all_pairwise_eps_idx[i_frame]
            pairwise_factors = all_pairwise_factors[i_frame]
            n_pairwise = np.shape(pairwise_factors)[0]
            for i_pw in range(n_pairwise):
                this_idx = pairwise_idxs[i_pw]
                all_pairwise_derivatives[this_idx][i_frame] += pairwise_factors[i_pw]

        if self.group_indices is None:
            def hepsilon(epsilons):
                nonbonded_epsilons = epsilons[:self.n_atom_types]
                pairwise_epsilons = epsilons[self.n_atom_types:]
                nonbonded_matrix_epsilons = compute_mixed_table(nonbonded_epsilons, [1])
                U_new = compute_energy_fast(nonbonded_matrix_epsilons, pairwise_epsilons, all_nonbonded_eps_idxs, all_nonbonded_factors, all_pairwise_eps_idx, all_pairwise_factors)
                return U_new

            def dhepsilon(epsilons):
                nonbonded_epsilons = epsilons[:self.n_atom_types]
                nonbonded_matrix_d_epsilons = compute_mixed_derivatives_table(nonbonded_epsilons)
                dU_nonbonded = compute_derivative_fast(nonbonded_matrix_d_epsilons, all_nonbonded_eps_idxs, all_nonbonded_factors)

                return dU_nonbonded + all_pairwise_derivatives
        else:
            def hepsilon(epsilons):
                true_epsilons = self.reconstruct_epsilons(epsilons)
                nonbonded_epsilons = true_epsilons[:self.n_atom_types]
                pairwise_epsilons = true_epsilons[self.n_atom_types:]
                nonbonded_matrix_epsilons = compute_mixed_table(nonbonded_epsilons, [1])
                U_new = compute_energy_fast(nonbonded_matrix_epsilons, pairwise_epsilons, all_nonbonded_eps_idxs, all_nonbonded_factors, all_pairwise_eps_idx, all_pairwise_factors)
                return U_new

            # pre-allocate the true derivatives numpy array in memory
            true_derivatives = [np.zeros(n_frames) for j in range(self.n_groups)]

            # define everything here as arrays for easy of reconstructing the
            # final derivatives list
            all_derivatives_array = np.zeros((self.n_atom_types+n_pairwise_eps, n_frames))
            for idx,thing in enumerate(all_pairwise_derivatives):
                all_derivatives_array[idx+self.n_atom_types,:] = thing

            def dhepsilon(epsilons):
                true_epsilons = self.reconstruct_epsilons(epsilons)
                nonbonded_epsilons = true_epsilons[:self.n_atom_types]
                nonbonded_matrix_d_epsilons = compute_mixed_derivatives_table(nonbonded_epsilons)
                dU_nonbonded = compute_derivative_fast(nonbonded_matrix_d_epsilons, all_nonbonded_eps_idxs, all_nonbonded_factors)

                all_derivatives_array[:self.n_atom_types,:] = dU_nonbonded

                for grp_idx, group in enumerate(self.parameter_indices):
                    true_derivatives[grp_idx][:] = np.sum(all_derivatives_array[group], axis=0)

                return true_derivatives

        return hepsilon, dhepsilon

    def save_model_parameters(self, new_epsilons):
        """

        """

        if self.group_indices is None:
            nonbonded_epsilons = new_epsilons[:self.n_atom_types]
            pairwise_epsilons = new_epsilons[self.n_atom_types:]
        else:
            true_new_epsilons = self.reconstruct_epsilons(new_epsilons)
            nonbonded_epsilons = true_new_epsilons[:self.n_atom_types]
            pairwise_epsilons = true_new_epsilons[self.n_atom_types:]

        new_atm_types = self.dict_atm_types_extended.copy()
        #new_atm_types = self.dict_atm_types.copy()
        #self.all_ps_pairs

        self.epsilons_atm_types, self.sigmas_atm_types
        for key,values in new_atm_types.items():
            idx = values[0]
            new_c6, new_c12 = convert_sigma_eps_to_c6c12(self.sigmas_atm_types[idx], nonbonded_epsilons[idx])
            values[4] = new_c6
            values[5] = new_c12

        new_pairwise_parameters = [thing for thing in self.parms2_]

        for i_count,thing in enumerate(new_pairwise_parameters):
            thing[0] = pairwise_epsilons[i_count]

        return new_atm_types, self.pairsidx_ps, new_pairwise_parameters
