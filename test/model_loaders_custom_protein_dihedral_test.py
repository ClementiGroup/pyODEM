import pytest
import pyODEM
import numpy as np
import mdtraj as md

DATA_PATH = 'test_data/1UBQ_sample_data'

@pytest.fixture
def load_dihedral_model():
    """
    Creates model, sets temperature 130 K


    """
    model = pyODEM.model_loaders.CustomProteinDihedral()
    model.set_temperature(130)
    assert model.temperature == 130, "Incorrect temperature set"
    model.read_parameters_from_file('{}/dihedral_description'.format(DATA_PATH))
    model.set_epsilons_from_file('{}/dihedral_values'.format(DATA_PATH))
    return(model)

class TestDihedralProtein():
    """
    Test model loader, that loads information for dihedral
    angels and computes corresponding contribution to the

    """

    def test_attributes(self,load_dihedral_model):
        """
        Test attributes checks wheather
         pyODEM.model_loaders.CustomProteinDihedral() object correctly loads
         fixed and variable parameters from file via read_parameters_from_file and
         set_epsilons_from_file
        """
        model = load_dihedral_model
        assert model.particles == [[0,2,4,6],
                                   [0,2,4,6],
                                   [2,4,6,8],
                                   [2,4,6,8],
                                   [138,140,142,144],
                                   [138,140,142,144],
                                   [140,142,144,145],
                                   [140,142,144,145]
                                   ], "Incorect atom indexing"
        assert model.n == [1,3,1,3,1,3,1,3], "Incorrect values of n"
        assert np.all(np.isclose(model.phi0,
                            [6.233279955960006,
                            18.699839832973435,
                            7.135358546027862,
                            21.406075638083582,
                            7.231018297000419,
                            21.693054891001257,
                            3.6331384923818013,
                            10.899415459692111]))
        assert model.n_dihedrals == 8, "Wrong number of parameters"
        assert np.all(np.isclose(model.epsilons,[1.0,
                                                 0.5,
                                                 1.0,
                                                 0.5,
                                                 1.0,
                                                 0.5,
                                                 1.0,
                                                 0.5]))
        return

    def test_correct_dihedral_energies(self,load_dihedral_model):
        """
        Test wheather class reproduce correct energy values for dihedral
        angles
        """
        # Code for reference generation can be found at
        # http://localhost:8888/notebooks/Testing%20ground%20for%20new%20ODEM%20model%20loader.ipynb
        reference = np.array([
                                0.9050796031951904,
                                1.0124619007110596,
                                0.8160284161567688,
                                1.0647443532943726,
                                0.32823166251182556,
                                0.7980178594589233,
                                0.596301794052124,
                                1.6045031547546387,
                                3.769892454147339,
                                1.648215651512146,
                                2.2846686840057373,
                                3.9934020042419434,
                                5.379669189453125,
                                1.274632453918457,
                                3.342480421066284,
                                3.8135528564453125,
                                2.1070337295532227,
                                2.3992724418640137,
                                1.9360657930374146,
                                2.4567041397094727
                             ])
        model = load_dihedral_model
        parameters = model.get_epsilons()
        model.beta = 1
        traj = md.load('{}/sample_traj.xtc'.format(DATA_PATH),
                        top='{}/ref.pdb'.format(DATA_PATH))
        data = model.load_data(traj)
        hepsilon, dhepsilon = model.get_potentials_epsilon(data)
        result = hepsilon(parameters)
        assert np.all(np.isclose(np.array(result),reference))


    def test_correct_temperature(self,load_dihedral_model):
        """
        Test, that temperature and beta are correct
        """
        model = load_dihedral_model
        beta = 1.0/(model.GAS_CONSTANT_KJ_MOL*model.temperature)
        assert model.temperature == 130
        assert np.isclose(model.beta, beta)
