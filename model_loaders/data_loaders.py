""" data_loaders.py contains a series of methods for handling basic data loading functions"""

import numpy as np

def load_array(fname):
    return np.loadtxt(fname)

