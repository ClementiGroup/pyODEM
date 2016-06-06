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

"""


#load all generic data loaders below
from data_loaders import *

#load the superclass
from super_model import ModelLoader

#load all subclasses below
from langevin_1d import Langevin
from proteins import Protein
from proteins import ProteinNonLinear

