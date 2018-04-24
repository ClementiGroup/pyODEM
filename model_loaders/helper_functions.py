import numpy as np
from mpi4py import MPI
import mdtraj as md

from proteins import ProteinNonBonded
import pyODEM.basic_functions.util as util

def load_protein_nb(topf, dtrajs, traj_files, top_file, observable_object=None, obs_data=None):
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
        #print "GOOD"
        use_states = np.arange(rank, n_states, size)

    #print "size: %d, rank: %d" % (size, rank)

    traj = md.load(traj_files, top=top_file)
    pmodel = ProteinNonBonded(topf)

    collected_data = []
    for state_idx in use_states:
        stuff = {}
        stuff["index"] = state_idx
        data = pmodel.load_data(traj[all_indices[state_idx]])
        stuff["data"] = data
        if observable_object is not None:
            if obs_data is None:
                use_obs_data = [data for i in range(len(observable_object.observables))]
            else:
                use_obs_data = []
                for obs_dat in obs_data:
                    use_obs_data.append(obs_dat[all_indices[state_idx]])

            observed, obs_std = observable_object.compute_observations(use_obs_data)
            stuff["obs_result"] = observed
            stuff["obs_std"] = obs_std

        collected_data.append(stuff)

    comm.Barrier()

    return pmodel, collected_data

def load_distance_traces(traj_file, top_file, fit_pairs):
    traj = md.load(traj_file, top=top_file)

    top = traj.top

    ca_names = "name CA"
    residue_codes = "GAVLIMFWPSTCYNQDEKRH"
    for count in range(20):
        ca_names += " or name CA%s" % (residue_codes[count])
    all_ca_idxs = top.select(ca_names) # convert residue index to ca index

    all_observed = []
    for residue_pair in fit_pairs:
        ca_pairs = [all_ca_idxs[residue_pair[0]], all_ca_idxs[residue_pair[1]]]
        distance_trace = md.compute_distances(traj, [ca_pairs], periodic=False)[:,0]
        all_observed.append(distance_trace)

    return all_observed
