"""
Utility functions for organizing and preparing data for input
"""

import numpy as np

def get_state_indices(dtrajs):
    equilibrium_frames = []
    indices = np.arange(np.shape(dtrajs)[0])
    for i in range(np.max(dtrajs)+1):
        state_data = indices[dtrajs == i]
        if not state_data.size == 0:
            equilibrium_frames.append(state_data)

    total_check = 0
    for set_of_frames in equilibrium_frames:
        total_check += len(set_of_frames)
    assert total_check == np.shape(dtrajs)[0]

    return equilibrium_frames
