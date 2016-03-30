""" Loading data for Molecular Dynamics Simulations

Requires package: https://github.com/ajkluber/model_builder

"""
import numpy as np
from pyfexd.model_loaders import ModelLoader
try:
    import model_builder as mdb

class Protein(ModelLoader):
    """ Subclass for making a ModelLoader for a Protein Model
    
    Methods:
        See ModelLoader in pyfexd/super_model/ModelLoader
    
    """
    
    def __init__(self, ini_file_name):
        """ Initialize the Langevin model, override superclass
        
        Args:
            ini_file_name: Name of a .ini file to load containing the 
                model information.
        
        Attributes:
            See superclass for generic attributes.
            epsilons(array): Chosen from a specific list of tunable 
                parameters from the .ini file.
        
        """
        try:
            from langevin_model.model import langevin_model as lmodel
        except:
            raise IOError("model_builder package is not installed.")
        
        ##remove .ini suffix
        self.model, self.fittingopts = mdb.inputs.load_model(ini_file_name)
        
        # get indices corresponding to epsilons to use
        self.use_params = np.arange(len(self.model.Hamiltonian._pairs))
        self.epsilons = self.model.Hamiltonian.params[self.use_params]
        self.beta = 1.0 #set temperature
    
    def get_potentials_epsilon(self, data):
        """ Return PotentialEnergy(epsilons)  
        
        See superclass for full description of purpose.
        Override superclass. Potential Energy is easily calculated since
        for this model, all epsilons are linearly related to the 
        potential energy.
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
