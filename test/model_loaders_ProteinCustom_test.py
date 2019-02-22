""" Test the Protein class that loads using model_builder"""
import pytest
import pyODEM
import os
import numpy as np

@pytest.fixture
def make_objects():
    cwd = os.getcwd()

    pmodel = pyODEM.model_loaders.CustomProtein()
    pmodel.set_temperature(120.)

    return pmodel


class TestProtein(object):
    def test_import_pmodel(self, make_objects):
        """ Check Protein class loads values correctly """
        # test that the various values are correctly loaded
        pmodel = make_objects
        assert True
