""" data_loaders.py contains methods for handling basic data loading """

import numpy as np

def load_array(fname):
    return np.loadtxt(fname)
