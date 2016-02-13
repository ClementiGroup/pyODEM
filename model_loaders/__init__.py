""" init file. 
Contains a superclass for loading and giving basic information to ml package
"""

import langevin_1d

class model_loader(object):
    def __init__(self):
        self.model = type('temp', (object,), {})()  ## potentials as a function of coordinates
        self.potentials_epsilon = []  ##potentials as a function of epsilons
        self.data = []
        
    def get_model(self):
        return self.model
    
    def get_potentials_epsilons(self):
        return self.potentials_epsilon
    
    def input_config(self, data):
        self.data = data
    
    def append_config(self, data):
        try:
            self.data = np.append(self.data, data, axis=0)
        except:
            print "Failed to load data... data is the wrong size or shape"
    
    def load_config(self, fname):
        self.data = np.loadtxt(fname)
    
    