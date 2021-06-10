import pytest
import pyODEM
import numpy as np
import mdtraj as md
import os
import sys
import numpy as np

ml = pyODEM.model_loaders

OPENAWSEM_LOCATION = os.environ["OPENAWSEM_LOCATION"]
sys.path.append(OPENAWSEM_LOCATION)


try:
    have_openmmawsem = True
    from openmmawsem  import *
    from helperFunctions.myFunctions import *
except:
    have_openmmawsem = False
    print("OPENAWSEM PACKAGE WAS NOT FOUND.")
    exit


DATA_PATH = '../'
sequence = 'MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'

# Get energies for direct contacts
traj = md.load(f'{DATA_PATH}/movie.pdb')
topology = traj.top
openawsem_protein = ml.OpenAWSEMProtein()
openawsem_protein.prepare_system(
                   f'{DATA_PATH}/1pgb_openmmawsem.pdb',
                   os.path.abspath(f'{DATA_PATH}/params_direct_only/.'),
                   [contact_term],
                   sequence,
                   chains='A')
H_ref_direct = openawsem_protein.calculate_H_for_trajectory(traj)
np.savetxt('H_direct.txt', H_ref_direct)


# Get energies for mediated contacts
openawsem_protein = ml.OpenAWSEMProtein()
openawsem_protein.prepare_system(
                   f'{DATA_PATH}/1pgb_openmmawsem.pdb',
                   os.path.abspath(f'{DATA_PATH}/params_mediated_only/.'),
                   [contact_term],
                   sequence,
                   chains='A')
H_ref_mediated = openawsem_protein.calculate_H_for_trajectory(traj)
np.savetxt('H_mediated.txt', H_ref_mediated)

# Get energies for burial contacts
openawsem_protein = ml.OpenAWSEMProtein()
openawsem_protein.prepare_system(
                   f'{DATA_PATH}/1pgb_openmmawsem.pdb',
                   os.path.abspath(f'{DATA_PATH}/params_burial_only/.'),
                   [contact_term],
                   sequence,
                   chains='A')
H_ref_burial = openawsem_protein.calculate_H_for_trajectory(traj)
np.savetxt('H_burial.txt', H_ref_burial)

# Get energies for all three terms together
openawsem_protein = ml.OpenAWSEMProtein()
openawsem_protein.prepare_system(
                   f'{DATA_PATH}/1pgb_openmmawsem.pdb',
                   os.path.abspath(f'{DATA_PATH}/params_all/.'),
                   [contact_term],
                   sequence,
                   chains='A')
H_ref_total = openawsem_protein.calculate_H_for_trajectory(traj)
np.savetxt('H_total_nonbonded.txt', H_ref_total)