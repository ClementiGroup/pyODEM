import numpy as np
from mpi4py import MPI
import mdtraj as md

from .proteins import ProteinNonBonded
from .proteins import Protein
from .langevin_1d import Langevin

import pyODEM.basic_functions.util as util

def load_protein_nb(topf, dtrajs, traj_files, top_file, observable_object=None, obs_data=None, weights=None):
    """ Function for setting up objects for re-weighting a non-bonded CG model

    Assumes there are N frames that need to be re-weighted, with D discrete states.

    Args:
        topf (str): String to the top_file for computing the non-bonded interactions.
        dtrajs (np.ndarray): Length N integers of values 0...D-1 denoting the discrete state it is in.
        obs_data (list): Use if data set for computing observables is different from data for computing the energy. List contains arrays where each array-entry corresponds to the observable in the ExperimentalObservables object. Arrays are specified with first index corresponding to the frame and second index to the data. Default: Use the array specified in data for all observables.
        weights (np.ndarray): Length N float denoting weights for each frame when computing the observable.

    """
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()

    all_indices = util.get_state_indices(dtrajs)

    n_states = len(all_indices)

    if n_states < size:
        # fewer states than cores
        print("TOO FEW STATES")
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
        if weights is not None:
            this_wt = weights[all_indices[state_idx]]
        else:
            this_wt = None
        if observable_object is not None:
            if obs_data is None:
                use_obs_data = [data for i in range(len(observable_object.observables))]
            else:
                use_obs_data = []
                for obs_dat in obs_data:
                    use_obs_data.append(obs_dat[all_indices[state_idx]])

            observed, obs_std = observable_object.compute_observations(use_obs_data, weights=this_wt)
            stuff["obs_result"] = observed
            stuff["obs_std"] = obs_std

        collected_data.append(stuff)

    comm.Barrier()
    if observable_object is not None:
        observable_object.synchronize_obs_seen()
    comm.Barrier()
    return pmodel, collected_data

def load_protein(dtrajs, traj_files, model_name, observable_object=None, obs_data=None, weights=None):
    """ Function for setting up objects for re-weighting a non-bonded CG model

    Args:
        obs_data (list): Use if data set for computing observables is different from data for computing the energy. List contains arrays where each array-entry corresponds to the observable in the ExperimentalObservables object. Arrays are specified with first index corresponding to the frame and second index to the data. Default: Use the array specified in data for all observables.

    """
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()

    all_indices = util.get_state_indices(dtrajs)

    n_states = len(all_indices)

    if n_states < size:
        # fewer states than cores
        print("TOO FEW STATES")
        pass
    else:
        # more states than cores
        #print "GOOD"
        use_states = np.arange(rank, n_states, size)

    #print "size: %d, rank: %d" % (size, rank)
    pmodel = Protein(model_name)
    traj = md.load(traj_files, top=pmodel.model.mapping.topology)

    collected_data = []
    for state_idx in use_states:
        stuff = {}
        stuff["index"] = state_idx
        data = pmodel.load_data_from_traj(traj[all_indices[state_idx]])
        stuff["data"] = data
        if weights is not None:
            this_wt = weights[all_indices[state_idx]]
        else:
            this_wt = None
        if observable_object is not None:
            if obs_data is None:
                use_obs_data = [data for i in range(len(observable_object.observables))]
            else:
                use_obs_data = []
                for obs_dat in obs_data:
                    use_obs_data.append(obs_dat[all_indices[state_idx]])

            observed, obs_std = observable_object.compute_observations(use_obs_data, weights=this_wt)
            stuff["obs_result"] = observed
            stuff["obs_std"] = obs_std

        collected_data.append(stuff)

    comm.Barrier()
    if observable_object is not None:
        observable_object.synchronize_obs_seen()
    comm.Barrier()
    return pmodel, collected_data

def load_langevin(dtrajs, traj_files, model_name, observable_object=None, obs_data=None, weights=None):
    """ Function for setting up objects for re-weighting a non-bonded CG model

    Args:
        obs_data (list): Use if data set for computing observables is different from data for computing the energy. List contains arrays where each array-entry corresponds to the observable in the ExperimentalObservables object. Arrays are specified with first index corresponding to the frame and second index to the data. Default: Use the array specified in data for all observables.

    """
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()

    all_indices = util.get_state_indices(dtrajs)

    n_states = len(all_indices)

    if n_states < size:
        # fewer states than cores
        print("TOO FEW STATES")
        pass
    else:
        # more states than cores
        #print "GOOD"
        use_states = np.arange(rank, n_states, size)

    #print "size: %d, rank: %d" % (size, rank)
    lmodel = Langevin(model_name)
    all_data = lmodel.load_data(traj_files)

    collected_data = []
    for state_idx in use_states:
        stuff = {}
        stuff["index"] = state_idx
        data = all_data[all_indices[state_idx]]
        stuff["data"] = data
        if weights is not None:
            this_wt = weights[all_indices[state_idx]]
        else:
            this_wt = None
        if observable_object is not None:
            if obs_data is None:
                use_obs_data = [data for i in range(len(observable_object.observables))]
            else:
                use_obs_data = []
                for obs_dat in obs_data:
                    use_obs_data.append(obs_dat[all_indices[state_idx]])

            observed, obs_std = observable_object.compute_observations(use_obs_data, weights=this_wt)
            stuff["obs_result"] = observed
            stuff["obs_std"] = obs_std

        collected_data.append(stuff)

    comm.Barrier()
    if observable_object is not None:
        observable_object.synchronize_obs_seen()
    comm.Barrier()
    return lmodel, collected_data

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


def load_custom_protein(dtrajs, traj, cpmodel, observable_object=None, obs_data=None, weights=None):
    """ Function for setting up objects for re-weighting a non-bonded CG model

    Args:
        obs_data (list): Use if data set for computing observables is different from data for computing the energy. List contains arrays where each array-entry corresponds to the observable in the ExperimentalObservables object. Arrays are specified with first index corresponding to the frame and second index to the data. Default: Use the array specified in data for all observables.

    """
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()

    all_indices = util.get_state_indices(dtrajs)

    n_states = len(all_indices)

    if n_states < size:
        # fewer states than cores
        print("TOO FEW STATES")
        pass
    else:
        # more states than cores
        #print "GOOD"
        use_states = np.arange(rank, n_states, size)

    #print "size: %d, rank: %d" % (size, rank)

    collected_data = []
    for state_idx in use_states:
        stuff = {}
        stuff["index"] = state_idx
        data = cpmodel.load_data(traj[all_indices[state_idx]])
        stuff["data"] = data
        if weights is not None:
            this_wt = weights[all_indices[state_idx]]
        else:
            this_wt = None
        if observable_object is not None:
            if obs_data is None:
                use_obs_data = [data for i in range(len(observable_object.observables))]
            else:
                use_obs_data = []
                for obs_dat in obs_data:
                    use_obs_data.append(obs_dat[all_indices[state_idx]])

            observed, obs_std = observable_object.compute_observations(use_obs_data, weights=this_wt)
            stuff["obs_result"] = observed
            stuff["obs_std"] = obs_std

        collected_data.append(stuff)

    comm.Barrier()
    if observable_object is not None:
        observable_object.synchronize_obs_seen()
    comm.Barrier()
    return collected_data
