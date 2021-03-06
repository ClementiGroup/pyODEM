""" Loading 1-Dimensional Langevin Dynamics Model set of functions will

Requires package: https://github.com/TensorDuck/langevin_model

"""
import numpy as np
from pyODEM.model_loaders import ModelLoader

try:
    from langevin_model.model import langevin_model as lmodel
except:
    pass

def compute_gaussian(eps, r, r0, sigma):
    return - eps * np.exp(-((r-r0)**2) / (2 * (sigma**2)))

class LangevinCustom(ModelLoader):
    def __init__(self):
        self.GAS_CONSTANT_KJ_MOL = 0.0083144621 #kJ/mol*k

        self.set_temperature(1.0)

        self.epsilons_list = []
        self.epsilons = np.array(self.epsilons_list)
        self.parameters = []
        self.n_params = 0



    def add_gaussian(self, parameters):
        self.epsilons_list.append(parameters["epsilons"])
        self.epsilons = np.array(self.epsilons_list)
        self.parameters.append([parameters["r0"], parameters["sigma"]])
        self.n_params = np.shape(self.epsilons)[0]

    def get_potentials_epsilon(self, data):

        def hepsilon(epsilons):
            total = np.zeros(np.shape(data)[0])

            for eps, param in zip(epsilons, self.parameters):
                value = compute_gaussian(eps, data, param[0], param[1])
                assert np.shape(value)[0] == np.shape(total)[0]
                total += value
            total *= - self.beta

            return total

        def dhepsilon(epsilons):
            total = np.zeros((np.shape(self.epsilons)[0], np.shape(data)[0]))

            for i in range(self.n_params):
                total[i,:] = compute_gaussian(1, data, self.parameters[i][0], self.parameters[i][1])
            total *= - self.beta

            return total

        return hepsilon, dhepsilon

class Langevin(ModelLoader):
    """ Subclass for making a ModelLoader for a 1-D Langevin dynamics

    Methods:
        See ModelLoader in pyODEM/super_model/ModelLoader

    """

    def __init__(self, ini_file_name):
        """ Initialize the Langevin model, override superclass

        Args:
            ini_file_name: Name of a .ini file to load containing the model
            information.

        Attributes:
            See superclass for generic attributes.
            epsilons(array): Chosen from a specific list of tunable
                parameters from the .ini file.

        """

        ##remove .ini suffix
        if ".ini" in ini_file_name[-4:]:
            ini_file_name = ini_file_name[:-4]
        try:
            self.model = lmodel(ini_file_name)
        except:
            raise IOError("langevin_model package is not installed.")

        # get indices corresponding to epsilons to use
        self.use_params = np.where(self.model.fit_parameters)[0]
        self.epsilons = self.model.params[self.use_params]
        self.beta = 1.0 #set temperature

    def load_data(self,fname):
        """ Load a data file and format for later use

        Args:
            fname (string): Name of a file to load.

        Return:
            Array of floats: Values to be interpreted by the
                get_potentials_epsilon() methoc.

        """
        return np.load(fname)

    def get_potentials_epsilon(self, data):
        """ Return PotentialEnergy(epsilons)

        Potential Energy is calculated since as a factors * epsilons, as the
        Hamiltonian only depends linearly on each epsilon.

        """

        #list of constant pre factors to each model epsilons
        constants_list = []

        for i in self.use_params:
            constants_list.append(self.model.potential_functions[i](data) * -1.0 * self.beta)

        #compute the function for the potential energy
        def hepsilon(epsilons):
            total = np.zeros(np.shape(data)[0])
            for i in range(np.shape(epsilons)[0]):
                total += epsilons[i]*constants_list[i]

            return total

        #compute the function for the derivative of the potential energy
        def dhepsilon(epsilons):
            #first index is frame, second index is for each epsilon
            return constants_list

        return hepsilon, dhepsilon
