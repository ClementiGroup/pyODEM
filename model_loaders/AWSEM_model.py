import numpy as np
import mdtraj as md
import os
import sys
import importlib.util
import pyODEM
from pyODEM.model_loaders import ModelLoader
from .hamiltonian_terms import AWSEMDirectInteraction as DirectInteraction
from .hamiltonian_terms import AWSEMMediatedInteraction as MediatedInteraction
from .hamiltonian_terms import AWSEMBurialInteraction as BurialInteraction
#### Set of imports required for opwnawsemProtein modeling
import simtk.unit as u

try:
    OPENAWSEM_LOCATION = os.environ["OPENAWSEM_LOCATION"]
    sys.path.append(OPENAWSEM_LOCATION)
except:
    print("OPENAWSEM LOCATION is not specified. class OpenAWSEMProtein will be anavailable")

try:
    from openmmawsem  import *
    from helperFunctions.myFunctions import *
except:
    print ("Packege openmmawsem was not found. Class OpenAWSEMProtein will be is anavailable")



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
    Class calculates nonbonded interactions for AWSEM model. Is constructed
    such that it can deal with mutations.
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
        self.temperature = 1.0/(self.beta*self.GAS_CONSTANT_KJ_MOL)
        self.R_MIN_I = 0.45 # nm
        self.R_MAX_I = 0.65 # nm
        self.ETA = 50 # nm^-1
        self.parameter_location = parameter_location # parameter location
        self.terms = []

        if topology is not None:
            self.set_topology(topology)

    def set_topology(self,topology):
        self.topology = topology
        self.n_residues = topology.n_residues
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
        n_residues = traj.top.n_residues
        self.n_frames = n_frames
        #Get pairs of atoms, for which distance, and, subsequently, density will be computed
        atom_list = self._get_C_beta()
        n_atoms = len(atom_list)
        assert n_residues == n_atoms, "Number of atom does not match number of residues."
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
            tanh_well_2d = np.zeros((n_residues,n_residues))
            tanh_well_2d[np.triu_indices(n_residues, k = 1)] = frame_well
            for res_ndx in range(self.n_residues-1):
                tanh_well_2d[res_ndx, res_ndx + 1]  = 0.0
            assert np.all(np.diag(tanh_well_2d) == 0)
            tanh_well_2d = tanh_well_2d + tanh_well_2d.T
            local_density = np.sum(tanh_well_2d, axis=0)
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
        direct_interaction = DirectInteraction(self.n_residues)
        direct_interaction.load_paramters(f'{self.parameter_location}/gamma.dat')
        direct_interaction.precompute_data(input=self.tanh_well_I,
                                           input_type='tanh_well')

        self.terms.append(direct_interaction)
        self.n_params += direct_interaction.get_n_params()
        return


    def add_mediated_interactions(self):
        """
        Add mediated interactions to the Hamiltonian. Is supposed to be
        used inside get_potentials_epsilon.
        """
        mediated_interaction = MediatedInteraction(self.n_residues)
        mediated_interaction.load_paramters(f'{self.parameter_location}/gamma.dat')
        mediated_interaction.precompute_data(distances=self.distances,
                                             densities=self.local_density)

        self.terms.append(mediated_interaction)
        self.n_params += mediated_interaction.get_n_params()
        return


    def add_burrial_interactions(self):
        """
        Add burial interactions to the Hamiltonian. Is supposed to be
        used inside get_potentials_epsilon.
        """
        burial_interaction = BurialInteraction(self.n_residues)
        burial_interaction.load_paramters(f'{self.parameter_location}/burial_gamma.dat')
        burial_interaction.precompute_data(densities=self.local_density)

        self.terms.append(burial_interaction)
        self.n_params += burial_interaction.get_n_params()
        return


    def setup_Hamiltonian(self, terms=['direct']):
        """
        Add all the required terms, as specified in the term
        list
        """
        self.terms = []
        self.n_params = 0
        method_dict = {'direct' : self.add_direct_interactions,
                     'mediated' : self.add_mediated_interactions,
                     'burial' : self.add_burrial_interactions }
        for type in terms:
            method_dict[type]()


    def get_potentials_epsilon(self,
                               sequence):
        """
        Generate two functions, that can calculate Hamiltonian and Hamiltonian
        derivatives.
        Grand assumption: DERIVATIVES DO NOT DEPEND ON PARAMETERS
                          H = sum_i (epsilon* dH/d(epsilon))


        Parameters
        ----------

        Returns
        -------
        hepsilon : function
                 function takes model parameters and calculates -beta*H for each frame.
                 See details in the function description

        dhepsilon : function
                 function takes model parameters and calculates  derivative of
                 See details in the function description



        """
        derivatives = np.zeros((self.n_frames, self.n_params))


        # Creation of a list and subsequent concatenation
        # of small arrays into a bigger one is not memory efficient.
        # In this case, at some point both the list and concatenated
        # array will exist in the memory, doubling requirements
        pointer = 0
        for term in self.terms:
          derivatives_term = term.calculate_derivatives(sequence)
          derivatives_term_shape = derivatives.shape
          assert derivatives_term_shape[0] == self.n_frames
          n_params = term.get_n_params()
          assert derivatives_term_shape[1] == self.n_params
          derivatives[:,pointer:pointer+n_params] = derivatives_term
          pointer += n_params

        # Do conversion to -beta*H
        derivatives = -1.0*self.beta*derivatives


        def dhepsilon(params):
          return(derivatives)

        def hepsilon(params):
          H = np.sum(np.multiply(derivatives, params), axis=1)
          return H

        return hepsilon, dhepsilon


    def get_H_func(self, sequence):
        hepsilon, dhepsilon = self.get_potentials_epsilon(sequence)

        def H_func(params, return_derivatives=False):
            if return_derivatives:
                return hepsilon(params), dhepsilon(params)
            else:
                return hepsilon(params)
        return H_func


    def get_epsilons(self):
        parameter_list = []
        for term in self.terms:
            params = term.get_parameters()
            print(type(params))
            print(params)
            parameter_list.append(params)
        param_array = np.concatenate(parameter_list)
        return param_array
