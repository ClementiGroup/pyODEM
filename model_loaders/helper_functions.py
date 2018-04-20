import numpy as np
from mpi4py import MPI
import mdtraj as md

from proteins import ProteinNonBonded
import pyODEM.basic_functions.util as util

def load_protein_nb(topf, dtrajs, traj_files, top_file):
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()

    all_indices = util.get_state_indices(dtrajs)

    n_states = len(all_indices)

    if n_states < size:
        # fewer states than cores
        print "TOO FEW STATES"
        pass
    else:
        # more states than cores
        print "GOOD"
        use_states = np.arange(rank, n_states, size)

    print "size: %d, rank: %d" % (size, rank)

    traj = md.load(traj_files, top=top_file)
    pmodel = ProteinNonBonded(topf)

    collected_data = []
    for state_idx in use_states:
        data = pmodel.load_data(traj[all_indices[state_idx]])
        collected_data.append([state_idx, data])

    return collected_data
