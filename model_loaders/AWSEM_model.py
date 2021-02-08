import numpy as np
import mdtraj as md
import os
import sys
import importlib.util
from pyODEM.model_loaders import ModelLoader

#### Set of imports required for opwnawsemProtein modeling
import simtk.unit as u
OPENAWSEM_LOCATION = os.environ["OPENAWSEM_LOCATION"]
sys.path.append(OPENAWSEM_LOCATION)

from openmmawsem  import *
from helperFunctions.myFunctions import *


class OpenAWSEMProtein():
    """
    Class calculates Hamiltonian for each frame of a given
    trajectory. Class relies on openawsem package and does not
    calculate derivatives.
    """

    def __init__(self):
        self.h_type="awsem"

    def prepare_system(self,
                       input_pdb_filename,
                       parameters_location,
                       force_setup,
                       sequence,
                       chains="A",
                       k_awsem=1.0,
                       xml_filename=None,
                       submode=-1
                       ):
        """
        Prepare system. Here, force_setup can be either a name of
        a file containing set_up_forces function definition, or a list
        of forces functions directly
        """
        if xml_filename is None:
            xml_filename=f'{OPENAWSEM_LOCATION}awsem.xml'

        oa = OpenMMAWSEMSystem(input_pdb_filename,
                                   k_awsem=k_awsem,
                                   chains=chains,
                                   xml_filename=xml_filename,
                                   seqFromPdb=sequence)

        if isinstance(force_setup, str):
            spec = importlib.util.spec_from_file_location("forces",
                                                          force_setup_file
                                                          )
            forces = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(forces)
            myForces = forces.set_up_forces(oa,
                                            submode=submode,
                                            parameterLocation=parameters_location)
        else:
            myForces = [force(oa,parametersLocation=parameters_location) for force in force_setup]
        oa.addForcesWithDefaultForceGroup(myForces)
        self.oa = oa

    def calculate_H_for_trajectory(self,traj, platform_name='CUDA'):
        """
        The prepared system is used to calculate hamiltonian value
        for all the frames of the supplied trajectory
        """
        energies = []
        integrator = CustomIntegrator(0.001)
        platform = Platform.getPlatformByName(platform_name)
        simulation = Simulation(self.oa.pdb.topology, self.oa.system, integrator, platform)
        for positions in traj.xyz:
            simulation.context.setPositions(positions)
            H = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
            energies.append(H)
        return np.array(energies)


class AWSEMProtein(ModelLoader):
    """
    Class calculates nonbonded interactions for AWSEM model.
    """
    def __init__(self, topology=None, parameter_location='.'):
        """ Initialize the Customized Protein model, override superclass

        Arguments
        ---------

        topology : mdtraj topology object.
        Topology of the system. Should use file generated with

        """
        self.GAS_CONSTANT_KJ_MOL = 0.0083144621 #kJ/mol*k
        self.model = type('temp', (object,), {})()
        self.epsilons = []
        self.beta = 1.0
        self.temperature = 1.0 / (self.beta*self.GAS_CONSTANT_KJ_MOL)
        self.R_MIN_I = 0.45 # nm
        self.R_MAX_I = 0.65 # nm
        self.ETA = 50 # nm^-1
        self.topology = topology # Md traj topology object
        self.parameter_location = parameter_location # parameter location

        if topology is not None:
            self.topology = topology

    def set_topology(self,topology):
        self.topology = topology
        return

    @staticmethod
    def get_tanh_well(r, eta, r_min, r_max):
        """
        Calculate Tanh well.
        """
        return 0.25*(1+np.tanh(eta*(r-r_min)))*(1+np.tanh(eta*(r_max-r)))


    def _get_C_beta(self):
        """
        Get indexes of C_beta atoms.
        If there is no C_beta atom, return index of
        C_alpha atom (in case of Gly).
        Gly has a special name in AWSEM pdb files (IGL)
        """
        atoms = self.topology.select("name CB or (name CA and (resname IGL or resname GLY))")
        assert len(atoms) == self.topology.n_residues
        return atoms

    def _get_atom_pairs(self, atom_list):
        """
        Generates atom_pairs out of atom_list.
        """
        atom_pairs = []
        n_atoms = len(atom_list)
        for atom_ndx_i in range(n_atoms-1):
            for atom_ndx_j in range(atom_ndx_i + 1, n_atoms): #Assume no diagonal elements
                atom_pairs.append([atom_list[atom_ndx_i], atom_list[atom_ndx_j]])
        self.atom_pairs = atom_pairs
        return atom_pairs

    def _compute_pairwise_distances(self, traj):
        """
        Function calculates pairwise distances for all the pairs of atoms in the
        atom_list.
        """
        # Compute distances between atoms specified in atom_pairs, end direct well potential
        self.distances = md.compute_distances(traj,
                                         self.atom_pairs,
                                         periodic=False)
        return

    def get_local_density(self,
                          traj
                          ):
        """
        Function calulates local density for trajectory.
        Pay attention, that all the distances in mdtraj trajectory are
        stored in nm, so all the parameters should be passed in appropriate
        units

        Params:
        -------

        traj : mdtraj Trajectory object
               AWSEM trajectory

        """
        assert self.R_MIN_I > 0.1, "R min should not be too small, danger of nonzero potential of interaction atom with itself"
        n_frames = traj.n_frames
        n_res = traj.top.n_residues

        #Get pairs of atoms, for which distance, and, subsequently, density will be computed
        atom_list = self._get_C_beta()
        n_atoms = len(atom_list)
        assert n_res == n_atoms, "Number of atom does not match number of residues."
        atom_pairs = self._get_atom_pairs(atom_list)
        self._compute_pairwise_distances(traj)
        self.tanh_well_I  = self.get_tanh_well(self.distances,
                                          self.ETA,
                                          self.R_MIN_I,
                                          self.R_MAX_I
                                          )
        # For each frame, expand 1D representation into 2D array and do summation to get
        # local density
        local_density_all = []
        for frame_well in self.tanh_well_I:
            tanh_well_2d = np.zeros((n_res,n_res))
            tanh_well_2d[np.triu_indices(n_res, k = 1)] = frame_well
            assert np.all(np.diag(tanh_well_2d)) == 0
            tanh_well_2d = tanh_well_2d + tanh_well_2d.T
            local_density = np.sum(tanh_well_2d, axis=0),
            local_density_all.append(local_density)
        self.local_density = np.array(local_density_all)
        return self.local_density


    def load_data(self, traj):
        """ Load a data file and format for later use

        Args:
            traj: str
             MD traj trajectory object

        Return:

        All the terms in this model depend on pairwise distances between
        Cbeta atoms and local densities. I.e., only these parameters are stored
        in data object.
        """
        # load traj
        self.get_local_density(traj)
        self.data = (self.distances, self.local_density)
        # select Calpha atoms if residue name is GLY and Cbeta atoms otherwise.
        return self.data

    def add_direct_interactions(self):
        """
        Add direct interactions to the Hamiltonian. Method is supposed to be
        used inside get_potentials_epsilon.
        """
        direct_interaction = ml.DirectInteraction(self.sequence)
        direct_interaction.load_paramters(f'{self.parameter_location}/gamma.dat')
        derivs, unique_params = direct_interaction.calculate_derivatives(self.tanh_well_I)
        unique_params_types = direct_interaction.unique_pair_types

    def add_mediated_interactions(self):
        """
        Add mediated interactions to the Hamiltonian. Is supposed to be
        used inside get_potentials_epsilon.
        """
        pass


    def add_burrial_interactions(self):
        """
        Add mediated interactions to the Hamiltonian. Is supposed to be
        used inside get_potentials_epsilon.
        """
        pass


    def get_potentials_epsilon(self,
                               data,
                               terms=['direct', 'mediated', 'burrial']):
      """
      Generate two functions, that can calculate Hamiltonian and Hamiltonian
      derivatives.

      Parameters
      ----------
      data :

      Returns
      -------
      hepsilon : function
                 function takes model parameters and calculates -beta*H for each frame.
                 See details in the function description

      dhepsilon : function
                 function takes model parameters and calculates  derivative of
                 See details in the function description



      """
      function_dict = {'direct' : add_direct_interactions,
                       'mediated' : add_mediated_interactions,
                       'burrial' : add_burrial_interactions }
      for type, function in function_dict.items():
          params, derivatives = self.function()


      # Do conversion to -beta*H

      def dhepsilon(params):
          return(derivatives)




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


class DirectInteraction(Hamiltonian):
    """
    Is responsible for direct interaction potential.
    """
    def __init__(self,
                 sequence,
                 lambda_direct=4.184,
                 r_min_I=0.45,
                 r_max_I=0.65,
                 eta=50,
                 separation=9):
        self.lambda_direct = lambda_direct
        self.r_min_I = r_min_I
        self.r_max_I = r_max_I
        self.eta = eta
        self.sequence = sequence

        # Here, determine a mask wich defines,
        # which indexes are neded for calculation
        n_res = len(sequence)
        counter = 0
        mask_indexes = []
        residue_pairs = []
        for i in range(n_res-1):
            for j in range(i+1, n_res):
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
                if input_type == 'distances':
                    q_direct = -self.lambda_direct*masked_input
        return q_direct


    def load_paramters(self, parameter_file):
        """
        Load parameters and store them in a form that is
        convinient for parameter retrival

        """
        data = np.loadtxt(parameter_file)
        gamma_direct = data[:210]
        gamma_direct_2d = np.zeros((20,20))
        count = 0
        for i in range(20):
            for j in range(i, 20):
                gamma_direct_2d[i][j] = gamma_direct[count][0]
                gamma_direct_2d[j][i] = gamma_direct[count][0]
                count += 1

        self.gamma = gamma_direct_2d
        return 0

    def get_all_parameters(self, sequence):
        """
        Go over the pairs and do the following:
        1) Create an array of parameters for each Qi parameters.
        2) Create an array, that contains an index of corresponding parameter
           for eqch of the Q values
        """
        # Correspondence between one-letter code and index in datafiles
        gamma_se_map_1_letter = {   'A': 0,  'R': 1,  'N': 2,  'D': 3,  'C': 4,
                                    'Q': 5,  'E': 6,  'G': 7,  'H': 8,  'I': 9,
                                    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
                                    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
        pair_types = []
        params_list = []
        for pair in self.residue_pairs:
            res_1 = sequence[pair[0]]
            res_2 = sequence[pair[1]]
            res_1_ndx = gamma_se_map_1_letter[res_1]
            res_2_ndx = gamma_se_map_1_letter[res_2]
            params_list.append(self.gamma[res_1_ndx, res_2_ndx])
            pair_type = [res_1, res_2]
            pair_type.sort()
            pair_type_str = f'{pair_type[0]}{pair_type[1]}'
            pair_types.append(pair_type_str)
        self.params_all = np.array(params_list)
        self.pair_types = np.array(pair_types)
        return np.array(params_list), pair_types

    def calculate_derivatives(self, input, input_type='distances'):
        """
        Calculate derivatives with respect of parameters
        of each type
        """
        # First, calculate q_values and sort them by pair types
        try:
            ndx = np.argsort(self.pair_types)
        except AttributeError:
            print("Should use get_all_parameters first")

        q = self._calculate_Q(input, input_type=input_type)
        ndx = np.argsort(self.pair_types)
        q =  q[:, ndx]
        self.pair_types = self.pair_types[ndx]
        self.params_all = self.params_all[ndx]
        self.sorted = True

        #Getting unique values
        unique_ndx = np.unique(self.pair_types, return_index=True)[1]
        unique_params = self.params_all[unique_ndx]
        unique_pair_types = self.pair_types[unique_ndx]
        self.unique_pair_types = unique_pair_types

        splitted_q = np.split(q, unique_ndx[1:], axis=1)
        derivs = np.array([np.sum(i, axis=1) for i in splitted_q]).T

        return derivs, unique_params
