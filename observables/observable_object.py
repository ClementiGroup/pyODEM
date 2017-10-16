""" Superlcass Observable defined  """

class Observable(object):
    """ Super class all observable objects inherit from

    The observable class object holds information about each observable. The
    types of inputs required varies greatly between experiment types.
    Therefore, the compute_observation() method must be defined in all
    subclasses.

    """
    def __init__(self):
        pass

    def compute_observation(self):
        """ return the observation and observation_std"""
        return 0, 1
