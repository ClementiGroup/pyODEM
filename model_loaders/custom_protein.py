""" Loading Customized Protein Model set of functions will

WangFei Yang

"""
import numpy as np
import mdtraj as md
from pyODEM.model_loaders import ModelLoader


class Custom_Protein(ModelLoader):
    """ Subclass for making a ModelLoader for a Customized Protein Model

    Methods:
        See ModelLoader in pyODEM/super_model/ModelLoader

    """

    def __init__(self):
        """ Initialize the Customized Protein model, override superclass

        """
        self.GAS_CONSTANT_KJ_MOL = 0.0083144621 #kJ/mol*k
        self.model = type('temp', (object,), {})()
        self.epsilons = []
        self.beta = 1.0
        self.temperature = 1.0 / (self.beta*self.GAS_CONSTANT_KJ_MOL)
        self.n_beads = 0
        self.n_pairs = 0
        self.native_distances = []
        self.exclusive_distances = []

    def load_data(self, traj):
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

        self.n_beads = traj.n_atoms
        pairs = []
        for i in range(self.n_beads):
            for j in range(i+1, self.n_beads):
                pairs.append([i,j])

        self.n_pairs = len(pairs) # For later checking if the length of epsilons is consistent

        data = md.compute_distances(traj, pairs, periodic=False)

        return data

    def load_epsilons(self, epsilons):
        """ Load the initial values of epsilons

        Args:
            epsilons (array of floats): directly loaded from external source.

        Return:
            Array of floats: Values to be interpreted by the
                get_potentials_epsilon() method.
        """

        self.epsilons = epsilons

        if epsilons.shape[0] != self.n_pairs:
            print('The length of epsilons is not consistent with the model')

        return 

    def get_epsilons(self):
        return self.epsilons   

    def load_native_distances(self, native_distances):
        """ Load the fixed parameter of native pairwise distances

        Args:
            native_distances (array of floats): directly loaded from external source.
        """

        self.native_distances = native_distances

        return

    def load_exclusive_distances(self, exclusive_distances):
        """ Load the fixed parameter of exclusive pairwise distances

        Args:
            exclusive_distances (array of floats): directly loaded from external source.
        """

        self.exclusive_distances = exclusive_distances

        return

    def get_potentials_epsilon(self, data):
        """ Return PotentialEnergy(epsilons)

        Potential Energy is calculated since as a factors * epsilons, as the
        Hamiltonian only depends linearly on each epsilon.

        The native distances and exclusive distances are the fixed parameters, 
        the sigmas will be fixed.

        """

        # Set up for parameters
        sigmas = 0.05
        native_distances = self.native_distances
        exclusive_distances = self.exclusive_distances

        # Check the consistence of fixed parameters
        if native_distances.shape != exclusive_distances.shape:
            print('Two sets of parameters are not consistent!')

        if native_distances.shape[0] != data.shape[1]:
            print('The fixed parameters are not consistent to the data!')

        """
        # Define functions for different forms of energy function
        # Attractive energy
        def h_attractive(r, epsilon, native_distance, exclusive_distance):
            V_rep = 1 + (1 / epsilon) * (exclusive_distance / r) ** 12
            V_gauss = epsilon * (1 - np.exp(- (((r - native_distance) ** 2)/(2 * (sigmas ** 2)))))

            return V_rep * V_gauss - epsilon

        # Attractive derivative
        def dh_attractive(r, native_distance):
            dh = - np.exp(- (((r - native_distance) ** 2)/(2 * (sigmas ** 2))))

            return dh

        # Pure exclusive energy
        def h_exclusive(r, exclusive_distance):
            return (exclusive_distance / r) ** 12

        # Repulsive energy
        def h_repulsive(r, epsilon, native_distance, exclusive_distance):
            V_tanh = - (epsilon * (np.tanh((native_distance - r + sigmas) / sigmas) + 1)) / 2

            return h_exclusive(r, exclusive_distance) + V_tanh

        # Repulsive derivative
        def dh_repulsive(r, native_distance):
            dh = - (np.tanh((native_distance - r + sigmas) / sigmas) + 1) / 2

            return dh
        """

        # Compute the function for the potential energy
        def hepsilon(epsilons):
            total = np.zeros(np.shape(data)[0])
            # Precalculate all attractive energies
            V_rep = 1 + (1 / epsilons) * (exclusive_distances / data) ** 12
            V_gauss = epsilons * (1 - np.exp(- (((data - native_distances) ** 2)/(2 * (sigmas ** 2)))))
            h_attractive = V_rep * V_gauss - epsilons

            # Precalculate all repulsive energies
            V_tanh = - (epsilons * (np.tanh((native_distances - data + sigmas) / sigmas) + 1)) / 2
            h_repulsive = (exclusive_distances / data) ** 12 + V_tanh

            # Precalculate all exclusive energies
            h_exclusive = (exclusive_distances / data) ** 12
       
            for i in range(np.shape(epsilons)[0]):
                if epsilons[i] > 0:
                    total += h_attractive[:,i]
                elif epsilons[i] == 0:
                    total += h_exclusive[:,i]
                else:
                    total += h_repulsive[:,i]
            
            return total * -1. * self.beta

        # Compute the function for the derivative of the potential energy
        def dhepsilon(epsilons):
            # It will return a (n * k) array, where n is the length of the trajectory,
            # and k is the number of epsilons.
            derivative = np.zeros((np.shape(epsilons)[0], np.shape(data)[0]))

            # Precalculate all attractive derivatives
            dh_attractive = - np.exp(- (((data - native_distances) ** 2)/(2 * (sigmas ** 2))))

            # Precalculate all repulsive derivatives
            dh_repulsive = - (np.tanh((native_distances - data + sigmas) / sigmas) + 1) / 2

            for i in range(np.shape(epsilons)[0]):
                if epsilons[i] < 0:
                    derivative[i,:] += dh_repulsive[:,i]
                else:
                    derivative[i,:] += dh_attractive[:,i]

            return derivative * -1. * self.beta

        return hepsilon, dhepsilon
