"""
Test spline-based model for odem optimization
"""

from json import load
import pytest
import pyODEM
import numpy as np
import mdtraj as md
import io
from pyODEM.model_loaders.hamiltonian_terms import TwoBodyBSpline

@pytest.fixture
def load_spline():
    n_bf = 30
    spline_range = (0.1, 1.4)
    spline = TwoBodyBSpline(n_bf, spline_range)
    return spline


class TestSpline:
    """
    Test all the spline-related functions
    """

    def test_atributes(self, load_spline):
        n_bf = 30
        spline = load_spline
        assert spline.n_bf == n_bf
        assert len(spline.basis_functions) == n_bf
        return

    def test_q(self, load_spline):
        spline = load_spline
        n_frames, n_pairs = 1000, 5
        d = np.abs(np.ones((n_frames, n_pairs)))
        spline._calculate_Q(d)
        print(spline.q.shape)
        assert spline.q.shape == (n_frames, n_pairs*spline.n_bf) 
        return

    def test_load_paramseters(self, load_spline):
        trial_input = io.StringIO('0 1 2 3 4 5 \n 6 7 8 9 10 11')
        spline = load_spline
        spline.n_bfs = 2
        spline.n_pairs = 6
        spline.load_paramters(trial_input)
        assert np.all(spline.params == np.arange(12))
        return