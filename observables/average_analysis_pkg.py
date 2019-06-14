import numpy as np
from pyODEM.observables import Observable

class AverageO(Observable):

    """ The class is used for observables, which are calculated as 
    an average over the trajectory.
    """
    
    def __init__(self):
        pass

    def compute_observed(self, data, wheight=None):
        """ Computes averages based on data for all frames
        
        Args:
             data  (2d numpy array): Contains calculated observables. The axis 0 correspond to                                                frames, axis 1 corresponds to values.
             wheight (2d numpy array): array of the same shape as data. 
                                       Right now is not used at all in the calculations.
                                       Added for consistancy with the same method in  the HistogramO Class            
            
        Returns:
             value (1D array) : array contains averages over frames
             seen  (1D array) :  1D array of True values
             stdev (1D array) : 1D array of  standard deviation for each value 
        
        
        """
        
        stdev = np.std(data, axis=0)      
        seen = [True for i in stdev]
        value = np.mean(data,axis=0)
        return value, stdev, seen
   
