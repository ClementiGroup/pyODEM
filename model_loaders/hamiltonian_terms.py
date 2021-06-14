"""
Classes for calculationg different energy terms
"""
import numpy as np
import mdtraj as md


class Hamiltonian():
    """
    Parent class for different types of interactions.
    Assumptions and rules used in interaction construction:
    1) Interaction contibution to the Hamiltonian can be described as
       H = sum_i (gamma(a_i)*Q_i)
    2) Each term in the sum linearly depends on a single parameter
    3) Number of terms is less or equal than number of different parameters

    Should be able to do the following:
    1) Calculate  an array, that contains Q_i for all i
    2) Parse parameter file and retrieve corresponding parameter for each Q_i.
    3) For each of the parameter, calculate a Hamiltonian derivative with respect
       to this parameter.


    """
    def __init__(self):
         """
         Initialize the interaction and pass all the constant parameters
         needed for calculation
         """
         return 0

    @staticmethod
    def get_tanh_well(r, eta, r_min, r_max):
        """
        Calculate Tanh well.
        """
        return 0.25*(1+np.tanh(eta*(r-r_min)))*(1+np.tanh(eta*(r_max-r)))


    def load_paramters(self, parameter_file):
        """
        Load paramters from parameter_file.
        """
        return 0


    def _calculate_Q(self, **kwargs):
        """
        Calculate Q for each term of the Hamiltonian
        """
        return 0

    def calculate_derivatives(self):
        """

        """

    def get_unique_parameters(self):
        """
        Get parameters. Order should be the same as in return of
        calculate_derivatives
        """
        return 0

    def get_all_parameters(self):
        """
        Returns parameters and their types, in the same order as Q
        """

    def get_H(self):
        """
        Calculate H.
        """


class AWSEMDirectInteraction(Hamiltonian):
    """
    Is responsible for direct interaction potential.
    """
    def __init__(self,
                 n_residues,
                 lambda_direct=4.184,
                 r_min_I=0.45,
                 r_max_I=0.65,
                 eta=50,
                 separation=9):
        self.type = 'awsem_direct'
        self.lambda_direct = lambda_direct
        self.r_min_I = r_min_I
        self.r_max_I = r_max_I
        self.eta = eta
        # Here, determine a mask wich defines,
        # which indexes are neded for calculation
        counter = 0
        mask_indexes = []
        residue_pairs = []
        for i in range(n_residues-1):
            for j in range(i+1, n_residues):
                if j-i > separation:
                    mask_indexes.append(counter)
                    residue_pairs.append([i,j])
                counter += 1
        self.mask = np.array(mask_indexes, dtype=int)
        self.residue_pairs = residue_pairs

    def _calculate_Q(self,
                    input,
                    input_type ='distances'
                    ):
        """
        Calculate Q values for the direct potential.

        Arguments:

        input : numpy array
        Input data (either distances of tanh well)
        It is assumed, that these data are given for
        """
        masked_input = input[:, self.mask]
        if input_type == 'distances':
            q_direct = -1*self.lambda_direct*self.get_tanh_well(masked_input,
                                                              self.eta,
                                                              self. r_min_I,
                                                              self.r_max_I)
        elif input_type == 'tanh_well':
            q_direct = -self.lambda_direct*masked_input

        self.q = q_direct
        return q_direct


    def load_paramters(self, parameter_file):
        """
        Load parameters and determine their  corresponding types

        """

        gamma_se_map_1_letter = {   'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
                                    'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
                                    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                                    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

        ndx_2_letter = {value : key for key, value in gamma_se_map_1_letter.items() }
        types = []
        data = np.loadtxt(parameter_file)
        gamma_direct = data[:210,0]
        self.gamma = gamma_direct


        for i in range(20):
            for j in range(i, 20):
                type=frozenset([ndx_2_letter[i],ndx_2_letter[j]])
                types.append(type)
        self.types = types
        return 0


    def map_types_to_pairs(self, sequence):
        """
        Create a mapping between types of parameters and residue pairs, that
        contribute to the H. As an outcome, creates a dictionary. Keys of the
        dictionary - frozen sets representing all the pair types. Values - list of integers -
        indexes of pairs in self.pairs, that corresponds to the key type
        """

        type_to_pair = { type: [] for type in self.types} # just 210 types
        for ndx, pair in enumerate(self.residue_pairs):

            pair_type = frozenset([sequence[pair[0]], sequence[pair[1]]])
            type_to_pair[pair_type].append(ndx)
        return type_to_pair


    def precompute_data(self, input, input_type):
        """
        Calculate values, that are used repeatedly for different calculations
        """
        self.q = self._calculate_Q(input, input_type=input_type)


    def calculate_derivatives(self, sequence, input=None, input_type='distances'):
        """
        Calculate derivatives with respect of parameters
        of each type
        """
        if not hasattr(self, 'q'):
            self.q = self._calculate_Q(input, input_type=input_type)

        #Getting mapping
        types_to_pair = self.map_types_to_pairs(sequence)
        derivatives = []  # At the end, derivatives should be a matrix
        for pair_type in self.types:
            fragment = np.sum(self.q[:, types_to_pair[pair_type]], axis=1)
            derivatives.append(fragment)
        derivatives = np.array(derivatives).T
        return(derivatives)

    def get_parameters(self):
        return self.gamma


    def get_n_params(self):
        return len(self.gamma)



class AWSEMBurialInteraction(Hamiltonian):
    """
    Is responsible for direct interaction potential.
    """
    def __init__(self,
                 n_residues,
                 lambda_burial=4.184, # will yeild energy in kJ/mol
                 eta_burial=4.0,
                 rho_I_limits = [0.0, 3.0],
                 rho_II_limits = [3.0, 6.0],
                 rho_III_limits = [6.0, 9.0]
                 ):
        self.type = 'awsem_burial'
        self.lambda_burial = lambda_burial
        self.eta_burial = eta_burial
        self.rho_I_limits = rho_I_limits
        self.rho_II_limits = rho_II_limits
        self.rho_III_limits = rho_III_limits



    def _burial_q(self, densities, rho_limits):
        """
        Calculate part, that does not depend on parameter
        for a particular range of q_values
        """
        rho_min, rho_max = rho_limits
        term =  np.tanh(self.eta_burial*(densities-rho_min))
        term += np.tanh(self.eta_burial*(rho_max - densities))

        return -0.5*self.lambda_burial*term



    def _calculate_Q(self,
                    densities
                    ):
        """
        Calculate Q values for burial potential.

        Arguments:

        distances, densities : numpy array
        Input distances
        """
        self.q_I = self._burial_q(densities, self.rho_I_limits)
        self.q_II = self._burial_q(densities, self.rho_II_limits)
        self.q_III = self._burial_q(densities, self.rho_III_limits)

        return self.q_I, self.q_II, self.q_III


    def load_paramters(self, parameter_file):
        """
        Load parameters and determine their  corresponding types

        """
        data = np.loadtxt(parameter_file)
        gamma_se_map_1_letter = {   'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
                                    'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
                                    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                                    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

        ndx_2_letter = {value : key for key, value in gamma_se_map_1_letter.items() }
        self.gamma  = np.loadtxt(parameter_file)
        self.types = [ndx_2_letter[i] for i in range(20)]
        return 0


    def map_types_to_residues(self, sequence):
        """
        Create a mapping between types of parameters and aminoacid residue, that
        contribute to the H. As an outcome, creates a dictionary. Keys of the
        dictionary - 1-letter  aminoacid type.  Values - list of integers -
        indexes of residues that corresponds to the key type
        """

        type_to_res = { type: [] for type in self.types} # just 210 types
        for ndx, residue in enumerate(sequence):
            type_to_res[residue].append(ndx)
        return type_to_res


    def precompute_data(self, densities):
        """
        Calculate values, that are used repeatedly for different calculations
        """
        self._calculate_Q(densities)


    def calculate_derivatives(self, sequence, densities=None):
        """
        Calculate derivatives with respect of parameters
        of each type
        """
        if not (hasattr(self, 'q_I') and hasattr(self, 'q_II')  and hasattr(self, 'q_III')):
            self._calculate_Q(densities)

        #Getting mapping
        types_to_res = self.map_types_to_residues(sequence)
        n_params_per_type = len(self.gamma)
        n_params = 3*n_params_per_type
        n_frames = len(self.q_I)
        derivatives = np.zeros((n_frames, n_params))
        # Dirivatives will contain 3 blocks: for 1, 2, 3 density conditions
        # than a block for protein-mediated contacts
        for ndx, res_type in enumerate(self.types):
            fragment_I = np.sum(self.q_I[:, types_to_res[res_type]], axis=1)
            fragment_II = np.sum(self.q_II[:, types_to_res[res_type]], axis=1)
            fragment_III = np.sum(self.q_III[:, types_to_res[res_type]], axis=1)
            derivatives[:,ndx] = fragment_I
            derivatives[:,ndx+n_params_per_type] = fragment_II
            derivatives[:,ndx+2*n_params_per_type] = fragment_III
        return(derivatives)

    def get_parameters(self):
        return self.gamma.flatten('F')


    def get_n_params(self):
        return  3*self.gamma.shape[0]


class AWSEMMediatedInteraction(Hamiltonian):
    """
    Is responsible for direct interaction potential.
    """
    def __init__(self,
                 n_residues,
                 lambda_mediated=4.184,
                 rho_0 = 2.6,
                 r_min_II=0.65,
                 r_max_II=0.95,
                 eta_sigma =7.0,
                 eta=50,
                 separation=9,
                 density_separation=2):

        self.lambda_mediated = lambda_mediated
        self.r_min_II = r_min_II
        self.r_max_II = r_max_II
        self.eta_sigma = eta_sigma
        self.eta = eta
        self.rho_0 = rho_0
        self.type = 'awsem_mediated'
        # Here, determine a mask wich defines,
        # which indexes are neded for calculation
        counter = 0
        mask_indexes = []
        residue_pairs = []
        for i in range(n_residues-1):
            for j in range(i+1, n_residues):
                if j-i > separation:
                    mask_indexes.append(counter)
                    residue_pairs.append([i,j])
                counter += 1
        self.mask = np.array(mask_indexes, dtype=int)
        self.residue_pairs = residue_pairs


    def _calculate_sigma(self, densities):
        """
        Function computes sigma water and sigma protein
        (Equation 12 from AWSEM_MD support info)

        rho : 2D numpy array of floats
        size NxM, N - number of frames, M - number of particles.
        Contains local density for each Calpha bead in each
        frame
        """
        n_frames = densities.shape[0]
        n_pairs = len(self.residue_pairs)
        sigma_water = np.zeros((n_frames, n_pairs))
        multiplier  = 1 - np.tanh(self.eta_sigma*(densities-self.rho_0))
        for ndx, pair in enumerate(self.residue_pairs):
            sigma_water_fragment = 0.25*np.multiply(multiplier[:,pair[0]],multiplier[:,pair[1]])
            sigma_water[:,ndx] = sigma_water_fragment
        sigma_prot = 1 - sigma_water
        return sigma_water, sigma_prot


    def _calculate_Q(self,
                    distances,
                    densities
                    ):
        """
        Calculate Q values for the mediated potential.

        Arguments:

        distances, densities : numpy array
        Input distances
        """
        masked_distances = distances[:, self.mask]

        # 1) Calculate tanh well II
        tanh_II = self.get_tanh_well(masked_distances,
                                self.eta,
                                self.r_min_II,
                                self.r_max_II)
        # 2) Calculate sigma ij (water). (eq 12 in the SI)
        # Put 0.25 and 0.75 for debugging purpose
        sigma_water, sigma_prot = self._calculate_sigma(densities)
        q_water = -1.0*self.lambda_mediated*tanh_II*sigma_water
        q_prot = -1.0*self.lambda_mediated*tanh_II*sigma_prot
        self.q_water = q_water
        self.q_prot = q_prot
        return self.q_water, self.q_prot


    def load_paramters(self, parameter_file):
        """
        Load parameters and determine their  corresponding types

        """

        gamma_se_map_1_letter = {   'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
                                    'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
                                    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                                    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

        ndx_2_letter = {value : key for key, value in gamma_se_map_1_letter.items() }
        types = []
        data = np.loadtxt(parameter_file)
        self.gamma_mediated_water  = data[210:,1]
        self.gamma_mediated_prot = data[210:,0]

        for i in range(20):
            for j in range(i, 20):
                type=frozenset([ndx_2_letter[i],ndx_2_letter[j]])
                types.append(type)
        self.types = types
        return 0


    def map_types_to_pairs(self, sequence):
        """
        Create a mapping between types of parameters and residue pairs, that
        contribute to the H. As an outcome, creates a dictionary. Keys of the
        dictionary - frozen sets representing all the pair types. Values - list of integers -
        indexes of pairs in self.pairs, that corresponds to the key type
        """

        type_to_pair = { type: [] for type in self.types} # just 210 types
        for ndx, pair in enumerate(self.residue_pairs):

            pair_type = frozenset([sequence[pair[0]], sequence[pair[1]]])
            type_to_pair[pair_type].append(ndx)
        return type_to_pair


    def precompute_data(self, distances, densities):
        """
        Calculate values, that are used repeatedly for different calculations
        """
        self._calculate_Q(distances, densities)


    def calculate_derivatives(self, sequence, distances=None, densities=None):
        """
        Calculate derivatives with respect of parameters
        of each type
        """
        if not (hasattr(self, 'q_water') and hasattr(self, 'q_prot')):
            self._calculate_Q(distances, densities)

        #Getting mapping
        types_to_pair = self.map_types_to_pairs(sequence)
        n_params_water = len(self.gamma_mediated_water)
        n_params_prot = len(self.gamma_mediated_prot)
        n_params = n_params_water + n_params_prot
        n_frames = len(self.q_water)
        derivatives = np.zeros((n_frames, n_params))
        # Dirivatives will contain first block for water-mediated contacts
        # than a block for protein-mediated contacts
        for ndx, pair_type in enumerate(self.types):
            fragment_water = np.sum(self.q_water[:, types_to_pair[pair_type]], axis=1)
            derivatives[:,ndx] = fragment_water
            fragment_prot = np.sum(self.q_prot[:, types_to_pair[pair_type]], axis=1)
            derivatives[:,ndx+n_params_water] = fragment_prot
        return(derivatives)

    def get_parameters(self):
        return np.concatenate([self.gamma_mediated_water, self.gamma_mediated_prot])


    def get_n_params(self):
        return   len(self.gamma_mediated_water) + len(self.gamma_mediated_prot)




class SBMNonbondedInteraction(Hamiltonian):
    """
    Is responsible for direct interaction potential.
    """
    def __init__(self, n_residues, params_description_file):
        self.load_parameter_description(params_description_file)
        self.n_residues = n_residues
        # Now, need to create a mask for distances. Here
        # it is assumed that during model loading stage a single distance
        # is calculated for each pairs of residues.
        counter = 0
        mask_indexes = []
        self.type = 'sbm_nonbonded'
        for i in range(n_residues-1):
            for j in range(i+1, n_residues):
                if frozenset((i,j)) in self.pairs:
                    mask_indexes.append(counter)
                counter += 1
        self.mask = np.array(mask_indexes, dtype=int)



    def load_parameter_description(self, file):
        """
        Load parameters descriptions and types.
        Should create list of pairs, lists of
        params.
        """
        description = np.genfromtxt(file,
                                    dtype=None,
                                    unpack=True,
                                    encoding=None,
                                    names=['atom_i', 'atom_j', 'ndx', 'type', 'sigma', 'r0', 'sigma_tg'])
        self.pairs = [frozenset((i[0]-1, i[1]-1)) for i in zip(description['atom_i'], description['atom_j'])]
        self.pair_types = description['type']
        self.sigma = description['sigma']
        self.r0 = description['r0']
        self.sigma_tg = description['sigma_tg']
        return

    def calculate_lj12gaussian(self, distance, r0, sigma_g):
         return -1.0*np.exp(-(distance - r0)**2/(2*sigma_g**2))

    def calculate_lj12gaussiantanh(self, distance, r0, sigma_t):
        return 0.5*(np.tanh((r0-distance + sigma_t)/sigma_t) + 1)


    def _calculate_Q(self,
                    distances,
                    ):
        """
        Calculate Q values for the mediated potential.
        Get distances. Should take descriptions and
        calculate q.

        Arguments:

        distances, densities : numpy array
        Input
        """
        type_dict = {'LJ12GAUSSIAN' : self.calculate_lj12gaussian,
                     'LJ12GAUSSIANTANH' : self.calculate_lj12gaussiantanh}
        masked_distances = distances[:, self.mask]
        q = np.zeros(masked_distances.shape)

        for ndx, type in enumerate(self.pair_types):
            distance = masked_distances[:, ndx]
            r0 = self.r0[ndx]
            sigma_tg = self.sigma_tg[ndx]
            q[:, ndx] = type_dict[type](distance, r0, sigma_tg)

        self.q = q

        return self.q


    def load_paramters(self, parameter_file):
        """
        Load parameters and determine their  corresponding types.
        In future, may need to add types to make parameter-specific
        optimization

        """
        self.params = np.loadtxt(parameter_file)
        return 0


    def precompute_data(self, distances):
        """
        Calculate values, that are used repeatedly for different calculations
        """
        self._calculate_Q(distances)


    def calculate_derivatives(self, distances=None, fraction=None):
        """
        Calculate derivatives with respect of parameters
        of each type.
        """
        if not (hasattr(self, 'q')):
            self._calculate_Q(distances)
        if fraction is None:
            derivatives = self.q
        else:
            derivatives = np.multiply(self.q, fraction)
            print("Multiplication done")
        return derivatives


    def get_parameters(self):
        return self.params


    def get_n_params(self):
        return   len(self.params)


# Create another class, that inherits from  SBMNonbondedInteraction, as it
# uses the same form for function. The difference is in handling parameters
# and functional types.
class SBMNonbondedInteractionsByResidue(SBMNonbondedInteraction):
    """
     Class holds energy calculations for SBM nonbonded ineractions,
     where parameters for each pair of residues are selected based on aminoacid
     identity.
    """


    def __init__(self, func_type ='from_file', topology_file=None):
        """
        Initialize SBMNonbondedInteractionsByResidue object.

        Parameters:
        -----------

        func_type : str, {'from_file', 'auto'}, default 'from_file'
                 Parameter defines, how functional type will be chosen for each
                 of the parameters
                 'from_file' : functional types are specified in a description file
                 'auto' : functional types are specified based on parameters values

        topology_file : str, default None
                 mdtraj topology file. Should be coarse-grained, with indexes matching
                 parameter file

        """
        self.func_type = func_type
        self.type = 'SBM nonbonded, residue-specific'
        if topology_file is not None:
            self.load_topology(topology_file)
        return


    def load_topology(self, topology_file):
        """
        Load topology so that number of residues can be identified.

        Parameters:

        topology_file : str, default None
                 mdtraj topology file. Should be coarse-grained, with indexes matching
                 parameter file

        """
        self.top = md.load(topology_file).top
        return


    def map_types_to_pairs(self):
        """
        Create a mapping between types of parameters and residue pairs, that
        contribute to the H. As an outcome, creates a dictionary. Keys of the
        dictionary - frozen sets representing all the pair types. Values - list of integers -
        indexes of pairs in self.pairs, that corresponds to the key type
        # NOTE: only works with one chain proteins.
        """

        sequence = self.top.to_fasta()[0]
        print("Types are assigned based on the following sequence:")
        print(sequence)
        type_to_pair = { type: [] for type in self.types} # just 210 types
        # Self.pairs contain atom numbers. Need to convert them to residue number
        for ndx, pair in enumerate(self.pairs):
            residue_1 = self.top.atom(pair[0]).residue.index
            residue_2 = self.top.atom(pair[1]).residue.index
            pair_type = frozenset([sequence[residue_1], sequence[residue_2]])
            type_to_pair[pair_type].append(ndx)
        return type_to_pair


    def load_parameter_description_full(self, file):
        """
        Load parameters descriptions and types.
        Should create list of pairs, lists of
        params.
        """
        description = np.genfromtxt(file,
                                    dtype=None,
                                    unpack=True,
                                    encoding=None,
                                    names=['atom_i', 'atom_j', 'ndx', 'type', 'sigma', 'r0', 'sigma_tg'])
        self.pairs = [[i[0]-1, i[1]-1] for i in zip(description['atom_i'], description['atom_j'])]
        self.pair_types = description['type']
        self.sigma = description['sigma']
        self.r0 = description['r0']
        self.sigma_tg = description['sigma_tg']
        return


    def load_parameter_description(self, file=None, mode='full'):
        """
        Here an appropriate form of parameter description is performed
        """
        method_dict = {'full': self.load_parameter_description_full}
        method_dict[mode](file)
        return


    def _calculate_Q(self,
                    distances,
                    ):
        """
        Calculate Q values for the mediated potential.
        Get distances. Should take descriptions and
        calculate q. Assume, distances are already prepared

        Arguments:

        distances : numpy array, n framees x m pairs
        Input
        """
        type_dict = {'LJ12GAUSSIAN' : self.calculate_lj12gaussian,
                     'LJ12GAUSSIANTANH' : self.calculate_lj12gaussiantanh}
        q = np.zeros(distances.shape)

        for ndx, type in enumerate(self.pair_types):
            distance = distances[:, ndx]
            r0 = self.r0[ndx]
            sigma_tg = self.sigma_tg[ndx]
            q[:, ndx] = type_dict[type](distance, r0, sigma_tg)

        self.q = q

        return self.q


    def load_parameters(self, parameter_file):
        """
        Load parameters and determine their  corresponding types

        """

        gamma_se_map_1_letter = {   'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
                                    'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
                                    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                                    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}

        ndx_2_letter = {value : key for key, value in gamma_se_map_1_letter.items() }
        types = []
        epsilons = np.loadtxt(parameter_file)
        for i in range(20):
            for j in range(i, 20):
                type=frozenset([ndx_2_letter[i],ndx_2_letter[j]])
                types.append(type)
        self.types = types
        return 0



    def precompute_data(self, distances):
        """
        Calculate values, that are used repeatedly for different calculations
        """
        self._calculate_Q(distances)


    def calculate_derivatives(self, input=None):
        """
        Calculate derivatives with respect of parameters
        of each type. In this case, as input we use distances formatted

        """
        sequence = self.top.to_fasta()[0]
        if not hasattr(self, 'q'):
            self.q = self._calculate_Q(distances)

        #Getting mapping
        types_to_pair = self.map_types_to_pairs(sequence)
        derivatives = []  # At the end, derivatives should be a matrix
        for pair_type in self.types:
            fragment = np.sum(self.q[:, types_to_pair[pair_type]], axis=1)
            derivatives.append(fragment)
        derivatives = np.array(derivatives).T
        return(derivatives)

    def calculated_energy(self, derivatives=None):
        if derivatives is None:
            derivatives = calculate_derivatives


    def get_parameters(self):
        return self.params


    def get_n_params(self):
        return   len(self.params)
