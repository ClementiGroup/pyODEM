import os
import sys
from simtk.unit import angstrom
from simtk.unit import kilocalorie_per_mole

try:
    OPENAWSEM_LOCATION = os.environ["OPENAWSEM_LOCATION"]
    sys.path.append(OPENAWSEM_LOCATION)
    # print(OPENAWSEM_LOCATION)
except KeyError:
    print("Please set the environment variable name OPENAWSEM_LOCATION.\n Example: export OPENAWSEM_LOCATION='YOUR_OPENAWSEM_LOCATION'")
    exit()

from openmmawsem import *
from helperFunctions.myFunctions import *


def set_up_forces(oa, computeQ=False, submode=-1, contactParameterLocation=".", membrane_center=-0*angstrom):
    # apply forces
    forces = [
        contact_term(oa, parametersLocation=".", burialPartOn=False)
    ]
    return forces
