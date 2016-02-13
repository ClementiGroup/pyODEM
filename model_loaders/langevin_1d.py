""" These set of functions will load in a simple 1-D Langevin data set"""
import numpy as np
import max)likelihood.model_loaders.model_loader as model_loader


""" USEFUL FUNCTIONS FOR GETTING DESIRED RESULTS FROM MODEL """

class langevin(model_loader):
    """Object for getting data sets and langevin based stuff from the """
    
    def __init__(self, ini_file_name):
        try:
            import langevin_model.model as lmodel
        except:
            raiseIOError("langevin_model package is not installed. Please check path variables or install the relevant package from: https://github.com/TensorDuck/langevin_model")
        
        ##remove .ini suffix
        if ".ini" in ini_file_name[-4:]:
            ini_file_name = ini_file_name[:-4]
        self.model = lmodel(ini_file_name)
        
    def input_config(self, data):
        self.data = data
    
    def append_config(self, data):
        try:
            self.data = np.append(self.data, data, axis=0)
        except:
            print "Failed to load data... data is the wrong size or shape"
    
    def load_config(self, fname):
        self.data = np.loadtxt(fname)
        
    def get_potentials_epsilon(self, data):
        pass
    
    
    
    