""" model_loaders contains modules for various models

Organized such that different models are separated into different files.
Keep all specific features of a model to each of its respective files.

Modules:
    data_loaders: Methods for loading data formats (e.g. .xtc)
    ModelLoader: SuperClass for each model object. Defines the necessary
        methods each model must have.

Specific Models Implemented:
    langevin_1d: For analyzing 1-dimensional langevin dynamics data.
        Associated with the github repository langevin_model.
    Protein: Uses model_builder to analyze protein trajectories using mdtraj. It
        assumes the Hamiltonian depends only linearly on the epsilons.
    ProteinNonLinear: Uses model_builder to analyze protein trajectories using
        mdtraj. It assumes the Hamiltonian does not depend linearly on the
        epsilons.

"""


#load all generic data loaders below
from data_loaders import *

#load the superclass
from super_model import ModelLoader

#load all subclasses below
from langevin_1d import Langevin
from proteins import Protein
from proteins import ProteinMultiTemperature
from proteins import ProteinNonLinear
from proteins import ProteinAwsem
from proteins import ProteinNonBonded

#load all the helper functions
from helper_functions import load_protein_nb
