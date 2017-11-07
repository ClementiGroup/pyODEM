"""
Utility functions for organizing and preparing data for optimization
"""

import numpy as np

def get_state_indices(dtrajs):
    """
    Convert a discrete trajectory into frame index for each state.

    Args:
        dtrajs (numpy.ndarray): Discrete trajectory indexed from 0 to N-1 for N
            discrete states.

    Returns:
        equilibrium_frames (list): A list with len = N, where each entry gives
            a list of indices corresponding to that discrete state.

    """
    equilibrium_frames = []
    indices = np.arange(np.shape(dtrajs)[0])
    n_states = np.max(dtrajs) + 1
    for i in range(n_states):
        state_data = indices[dtrajs == i]
        if not state_data.size == 0:
            equilibrium_frames.append(state_data)

    total_check = 0
    for set_of_frames in equilibrium_frames:
        total_check += len(set_of_frames)
    assert total_check == np.shape(dtrajs)[0]
    if not len(equilibrium_frames) == n_states:
        print "Warning: in util.get_state_indices(), number of expected equilibrium frames not matching number of equilibrium frames.\n Check inputted dtrajs and verify each discrete index is represented correctly"

    return equilibrium_frames

### Functions for determining dEpsilon statistics ###

def bounds_simple(deps, all_epsilons, epsilon_info=None, highest=2., lowest=0.):
    bounds = []
    for idx,eps_value in enumerate(all_epsilons):
        low_val = eps_value - deps
        high_val = eps_value + deps
        if low_val < lowest:
            low_val = lowest
        if high_val > highest:
            high_val = highest
        bounds.append([low_val, high_val])
    return bounds

def bounds_slow_negative_only_nonnative(deps, all_epsilons, epsilon_info=None, highest=2., lowest=-2):
    bounds = []
    for i, eps_value in enumerate(all_epsilons):
        if epsilon_info[i] >= -0.09:
            this_low = -0.09
        else:
            this_low = lowest

        low_val = eps_value - deps
        high_val = eps_value + deps
        if low_val < this_low:
            low_val = this_low
        if high_val > highest:
            high_val = highest

        bounds.append([low_val, high_val])
    return bounds
