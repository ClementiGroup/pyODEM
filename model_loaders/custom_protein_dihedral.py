"""
Loader for dihedral parameters of protein.

Iryna Zaporozhets
"""
from pyODEM.model_loaders import ModelLoader
import numpy as np
import mdtraj as md

class CustomProteinDihedral(ModelLoader):
    """
    A child class of ModelLoader. Loads all the data from trajectory
    that are required for ODEM optimization of dihedral interactions.
    At this point, only PeriodicTorsionForce are implemented.

    Dependencies
    """
    def __init__(self):
        """ Initialize the dihedral protein class
        """
        self.GAS_CONSTANT_KJ_MOL = 0.0083144621 #kJ/mol*k
        self.model = type('temp', (object,), {})()
        self.epsilons = []
        self.particles = [] # list of lists. Each inner list containts 4 integers, that correspond
                            # to atom indexes that form corresponding dihedral.
        self.n_dihedrals = 0
        self.n = [] # n-values for each of the parameters
        self.phi0 = [] # Phi0 parameters for each of the parameters
        self.temperature = None

    def __str__(self):
        """
        Return message about the state of the CustomProteinDihedral object
        """
        str  = "Protein dihedral angle model. {} parameters. Temperature: {}".format(
        self.n_dihedrals,
        self.temperature
        )
        return str

    def set_temperature(self,temperature):
        """
        Set model temperature in SBM model units and
        set beta in kJ/mol
        Parameters
        ----------
        temperature : float
        Model temperatures
        """
        self.temperature = temperature
        self.beta = 1.0/(self.temperature*self.GAS_CONSTANT_KJ_MOL)
        return

    def add_paramter(self,atoms,n,phi0):
        """
        Add a single dihedral angle paramter
        atoms : list of int
              `atoms` is a list of 4 integers, indexed from 0,
               that represent indexes of 4 atoms forming a dihedral
               angle.
        n : int
            parameter to use for energy calculation
        phi0 : equilibrium angle in radians
        """
        assert len(atoms) == 4, "Wrong number of atoms. Should be 4"
        self.n.append(n)
        self.phi0.append(phi0)
        self.n_dihedrals += 1

    def read_parameters_from_file(self,file_name):
        """
        Method reads parameters of dihedral interactions from file.
        Only constant parameters are read.
        The file is organized in the format:
        atom1  atom2  atom3  atom4  param_ndx  Force_Type  n  phi0
        """
        counter = 0
        input_data = open(file_name,'rt')
        for line in input_data.readlines():
            parameter_description  =  line.split()
            if parameter_description[0][0] != '#':
                assert int(parameter_description[4]) == counter, "Incorrect order of parameters"
                assert parameter_description[5] == 'PeriodicTorsionForce', "Incorrect ForceType"
                atm0 = int(parameter_description[0]) -1 # convert to 0-indexing
                atm1 = int(parameter_description[1]) -1
                atm2 = int(parameter_description[2]) -1
                atm3 = int(parameter_description[3]) -1
                self.particles.append([atm0,atm1,atm2,atm3])
                self.n.append(int(parameter_description[6]))
                self.phi0.append(float(parameter_description[7]))
                counter += 1
        self.n_dihedrals = counter
        input_data.close()

    def load_data(self, traj):
        """
        Calculate dihedral angles

        Parameters
        ----------
        traj : mdtraj trajectory object
               A trajectory that should be featurized

        Return
        ------
        data : np.ndarray, shape=(n_frames, n_dihedrals), dtype=float
              A two-dimensional numpy array, that contains values for dihedral angles
              in radians, one value for each list of  four atoms in self.particles and
              each frame in the trajectory. Definition of dihedral angles is consistent
              with openMM definition and is shifted with respect to mdtraj definition by
              pi rad.
        """

        data = md.compute_dihedrals(traj, self.particles)
        return data

    def get_epsilons(self):
        """
        Return variable parameters of the model

        Returns
        -------
        nd.array, shape=(n_dihedrals)
        Current model parameters
        """
        return self.epsilons

    def set_epsilons(self,epsilons):
        """
        Set variable parameters of the model

        Parameters
        ----------
        epsilons : list or nd.array, dtype=float
        Set of epsilons. Number of input parameters
        should be equal to `self.n_dihedrals` (number of existing dihedral angles)

        """
        length = len(epsilons)
        assert length == self.n_dihedrals, 'Number of parameter passed: {}. \
        Expected: {}'.format(length,self.n_dihedrals)

        self.epsilons = epsilons
        return

    def set_epsilons_from_file(self,filename):
        """
        Load variable parameters from file

        Parameters
        ----------

        filename : str
                  Path to the textfile with array of epsilons
        """
        epsilons = np.loadtxt(filename)
        self.set_epsilons(epsilons)
        return

    def get_potentials_epsilon(self, data):
        """
        Generate two functions, that can calculate Hamiltonian and Hamiltonian
        derivatives.

        Parameters
        ----------
        data : np.ndarray, shape=(n_frames, n_dihedrals), dtype=float
              A two-dimensional numpy array, that contains values for dihedral angles
              in radians, one value for each list of  four atoms in self.particles and
              each frame in the trajectory

        Returns
        -------
        hepsilon : function
                   function takes model parameters and calculates -beta*H for each frame.
                   See details in the function description

        dhepsilon : function
                   function takes model parameters and calculates  derivative of
                   See details in the function description

        """
        # First, need to precompute the part of the hamiltonian, that does not
        # depend on model parameters.
        #derivatives = -1*self.beta*(np.cos(data*self.n-self.phi0) + 1)
        derivatives = (np.cos(data*self.n-self.phi0) + 1)

        def hepsilon(epsilons):
            """
            Computes Hamiltonian for each frame of the trajectory.
            Parameters
            -----------
            epsilons : np.ndarray, shape=(n_dihedrals), dtype=float
                       Adjustable model parameters

            Returns
            -------
            hepsilon : np.ndarray, shape=(n_frames), dtype = float
            """
            return np.sum(derivatives*epsilons,axis=1)

        def dhepsilon(epsilons):
            """
            Computes negative partial derivative of H for each frame and each
            paramter in `epsilons`.
            Parameters
            -----------
            epsilons : np.ndarray, shape=(n_dihedrals), dtype=float
                       Adjustable model parameters

            Returns
            -------

            derivatives:  np.ndarray, shape=(n_frames,n_dihedrals), dtype=Float
                        Element (i,j) corresponds to the derivative of Hamiltonian
                        for frame i with respect to variable parameter of dihedral
                        angle j, multiplied by -beta
            """
            return derivatives

        return hepsilon, dhepsilon
