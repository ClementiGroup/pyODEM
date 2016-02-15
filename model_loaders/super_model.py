class Model_Loader(object):
    def __init__(self):
        self.model = type('temp', (object,), {})()  ## potentials as a function of coordinates
        self.epsilons = []  ##potentials as a function of epsilons
        self.data = []
        
    def get_model(self):
        return self.model
    
    def get_potentials_epsilons(self):
        return self.epsilons
    
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
        """ Takes a 1-d array, outputs a function(epsilons_list) 
        
        get_potentials_epsilons(self, data) should take as input
        some data that is already properly formatted for the model
        in question. Then, it should calculate a function where
        the epsilons are the independent variables. the function
        is formatted to take a list of epsilons as an input and
        return a float number as its output.
        
        """
        
        def hepsilon(x):
            return 0.0
        
        return hepsilon
    