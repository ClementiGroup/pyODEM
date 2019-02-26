""" Initiate all the sub packages for calculating things """
import numpy as np
from mpi4py import MPI

from . import basic_functions

from . import model_loaders
from . import observables

from . import estimators



def Init():
    """ Optional (often required) for initializing MPI and numpy in parallel """

    try:
        MPI.Init()
    except:
        pass

    # should implement a way to seed numpy.random that is effective.
    # Google: numpy parallelization random seed for details.
