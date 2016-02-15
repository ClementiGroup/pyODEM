""" init file. 
Contains a superclass for loading and giving basic information to ml package
"""


#load all generic data loaders below
from data_loaders import *

#load the superclass
from super_model import Model_Loader

#load all subclasses below
from langevin_1d import Langevin


