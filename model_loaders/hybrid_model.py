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
from .hamiltonian_terms import SBMNonbondedInteraction
from .hamiltonian_terms import SBMNonbondedInteractionsByResidue
from .hamiltonian_terms import TwoBodyBSpline


class HybridProtein(ModelLoader):
    """
    Class handles arbitrary protein models and prepares data for
    ODEM optimization.

    """
    def __init__(
        self,
        topology=None,
        parameter_location='.',
        traj_type=None,
        param_dict=None):
        """ Initialize the Customized Protein model, override superclass

        Arguments
        ---------

        topology : mdtraj topology object.
        Topology of the system. Should use file generated with
        traj_type : {'awsem', 'sbm_ca'}
        param_dict: dict
        Dictionary of additional parameters needed to setup the Hamiltonian 
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
        self.traj_type = traj_type

        if topology is not None:
            self.set_topology(topology)


        if param_dict is not None:
            self.set_additional_params(param_dict)

    def set_topology(self,topology):
        self.topology = topology
        self.n_residues = topology.n_residues
        return

    def set_additional_params(self, param_dict):
        """
        Set parameters required for specific hamiltonian terms
        """
        for key, value in param_dict.items():
            setattr(self, key, value)
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

    def _get_density_atoms(self):
        """
        Get atoms, that will be used to calculate density.
        """
        if self.traj_type == None:
            self.traj_type = 'awsem'
        if self.traj_type =='awsem':
            atoms = self._get_C_beta()
        if self.traj_type == 'sbm_ca':
            atoms = self.topology.select("name CA")
            assert len(atoms == self.topology.n_residues)
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
        atom_list = self._get_density_atoms()
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

    def clear_data(self, clear_distance=True, clear_local_density=True):
        """
        Removes data 
        """
        if clear_distance:
            self.distances = None
        if clear_density:
            self.density = None
        return

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

    def add_sbm_nonbonded_interactions(self):
        """
        Add structure-based nonbonded interactions to the Hamiltonian.
        Is supposed to be  used inside get_potentials_epsilon.
        """
        sbm_interaction = SBMNonbondedInteraction(self.n_residues, f'{self.parameter_location}/pairwise_params')
        sbm_interaction.load_paramters(f'{self.parameter_location}/model_params')
        sbm_interaction.precompute_data(distances=self.distances)

        self.terms.append(sbm_interaction)
        self.n_params += sbm_interaction.get_n_params()
        return


    def add_sbm_nonbonded_residue_specific_interactions(self):
        """
        Add structure-based nonbonded interactions to the Hamiltonian,
        where each pair of aminoacid types has a unique model parameter
        """
        sbm_residue_specific_interaction = SBMNonbondedInteractionsByResidue(func_type ='auto')
        print(sbm_residue_specific_interaction.type)
        sbm_residue_specific_interaction.load_topology(f'{self.parameter_location}/ref.pdb')
        sbm_residue_specific_interaction.load_parameter_description(f'{self.parameter_location}/pairwise_params', mode='full')
        sbm_residue_specific_interaction.load_parameters(f'{self.parameter_location}/model_params')
        # The distances that exist in HybridProtein (self.distances) are not the same that need to be used
        # in calculations for this nonbonded interactions. Here, I will do masking externally to SBMNonbondedInteractionsByResidue.
        # SBMNonbondedInteractionsByResidues is a generic class. It does not need to know how distances in HybridModels are organized.
        counter=0
        mask_indexes = []
        n_residues = sbm_residue_specific_interaction.top.n_residues
        for i in range(n_residues-1):
            for j in range(i+1, n_residues):
                if [i,j] in sbm_residue_specific_interaction.pairs:
                    mask_indexes.append(counter)
                counter += 1
                mask = np.array(mask_indexes, dtype=int)
        print("MASK")
        print(sbm_residue_specific_interaction.pairs)
        print(mask)
        # Need to be sure that each  pair has correspondent mask in the distance
        assert len(mask) == len(sbm_residue_specific_interaction.pairs), "Not all the pairs have corresponding distance in the mask. Check atom order in pairs"
        sbm_residue_specific_interaction.precompute_data(distances=self.distances[:, mask])
        self.terms.append(sbm_residue_specific_interaction)
        self.n_params += sbm_residue_specific_interaction.get_n_params()
        return

    def add_two_body_b_spline_nonbonded_interactions(self):
        """
        Add structure-based nonbonded interactions to the Hamiltonian.
        Is supposed to be  used inside get_potentials_epsilon.
        """
        counter=0
        mask_indexes = []
        spline_interaction = TwoBodyBSpline(
            self.n_bf, 
            self.spline_range,
            params_description_file=f'{self.parameter_location}/pairwise_params_spline'
            )
        spline_interaction.load_paramters(f'{self.parameter_location}/model_params_spline')
        # Create a mask to select only the distances that are needed:
        for i in range(self.n_residues-1):
            for j in range(i+1, self.n_residues):
                if frozenset([i,j]) in spline_interaction.pairs:
                    mask_indexes.append(counter)
                    counter += 1        
        mask = np.array(mask_indexes, dtype=int) 
        spline_interaction.precompute_data(distances=self.distances[:, mask])
        print("Q array")
        print(spline_interaction.q)
        print("#"*10)
        self.terms.append(spline_interaction)
        self.n_params += spline_interaction.get_n_params()
        return


    def setup_Hamiltonian(self, terms=['direct']):
        """
        Add all the required terms, as specified in the term
        list
        """
        print("Setting up Hamiltonian")
        self.terms = []
        self.n_params = 0
        method_dict = {'direct' : self.add_direct_interactions,
                     'mediated' : self.add_mediated_interactions,
                     'burial' : self.add_burrial_interactions,
                     'sbm_nonbonded' : self.add_sbm_nonbonded_interactions,
                     'sbm_nonbonded_residue_specific' : self.add_sbm_nonbonded_residue_specific_interactions,
                     '2body_spline' : self.add_two_body_b_spline_nonbonded_interactions}
        for type in terms:
            method_dict[type]()




    def get_potentials_epsilon(self,
                               **kwargs):
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
            print(term.type)
            if term.type in ['awsem_burial', 'awsem_mediated', 'awsem_direct' ]:
                derivatives_term = term.calculate_derivatives(kwargs['sequence'])
            elif term.type in ['sbm_nonbonded', '2body_spline']:
                derivatives_term = term.calculate_derivatives(fraction=kwargs['fraction'])
            
            elif term.type in ['SBM nonbonded, residue-specific']:
                print("Hello!")
                derivatives_term = term.calculate_derivatives(sequence=kwargs['sequence'])
            else:
                raise ValueError

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

    def get_H_func(self, **kwargs):
        hepsilon, dhepsilon = self.get_potentials_epsilon(**kwargs)

        def H_func(params, return_derivatives=False, return_derivatives_only=False):
            if return_derivatives:
                return hepsilon(params), dhepsilon(params)
            elif return_derivatives_only:
                return dhepsilon(params)
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
