""" Loading 1-Dimensional Langevin Dynamics Model set of functions will

Requires package: https://github.com/TensorDuck/langevin_model

"""
import numpy as np
from pyfexd.model_loaders import ModelLoader

class Langevin(ModelLoader):
    """ Subclass for making a ModelLoader for a 1-D Langevin dynamics
    
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
            raise IOError("langevin_model package is not installed. Please check path variables or install the relevant package from: https://github.com/TensorDuck/langevin_model")
        
        ##remove .ini suffix
        if ".ini" in ini_file_name[-4:]:
            ini_file_name = ini_file_name[:-4]
        self.model = lmodel(ini_file_name)
        
        # get indices corresponding to epsilons to use
        self.use_params = np.where(self.model.fit_parameters)[0] 
        self.epsilons = self.model.params[self.use_params]
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
            constants_list.append(self.model.potential_functions[i](data))

        
        def hepsilon(epsilons):
            total = np.zeros(np.shape(data)[0])
            
            for i in range(np.shape(epsilons)[0]):
                total += epsilons[i]*constants_list[i]
            
            total = -1.0 * self.beta * total
            
            return total
        
        
        
        return hepsilon
    
    
    
