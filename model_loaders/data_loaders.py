""" data_loaders.py contains methods for handling basic data loading """

import numpy as np

def load_array(fname):
    return np.loadtxt(fname)

class DataObjectBase(object):
    """ Super Class of array-like data loading objects

    The intention is for sub-classes of this class to use the same __getitem__
    method. Then the sub-classes will handle how these things are stored.

    """

    def __init__(self, list_of_lists, list_of_arrays, list_of_trajs):
        # check all the lists have the same length:
        list_size = None
        width = 0
        if list_of_lists is not None:
            list_size = len(list_of_lists[0])
            width += 1
            for this_list in list_of_lists:
                if len(this_list) == list_size:
                    pass
                else:
                    raise IOError("Not all lists have the same size")
        if list_of_arrays is not None:
            if list_size is None:
                list_size = np.shape(list_of_arrays)[0]
            for this_array in list_of_arrays:
                width += 1
                if np.shape(this_array)[0] == list_size:
                    pass
                else:
                    raise IOError("The first dimension of an array does not match the size of each list")

        if list_of_trajs is not None:
            if list_size is None:
                list_size = list_of_trajs[0].n_frames
            for this_traj in list_of_trajs:
                width += 1
                if this_traj.n_frames == list_size:
                    pass
                else:
                    raise IOError("The number of frames in the trajectory does not match the size of each list")

        # if all checks pass, store the lists and arrays for use later
        self.list_of_lists = list_of_lists
        self.list_of_arrays = list_of_arrays
        self.list_of_trajs = list_of_trajs
        if list_size is None:
            self.list_size = 0
        else:
            self.list_size = list_size

        self.width = width

    @property
    def shape(self):
        return (self.list_size, self.width)

    def __getitem__(self, args):
        return_list_stuff = None
        return_array_stuff = None
        return_traj_stuff = None
        #print args
        if self.list_of_lists is not None:
            return_list_stuff = []
            if isinstance(args, list) or isinstance(args, np.ndarray):
                # list of indices to use. Grab from each array and list of lists
                for this_list in self.list_of_lists:
                    this_list_selected = []
                    for idx in args:
                        this_list_selected.append(this_list[idx])
                    return_list_stuff.append(this_list_selected)
            elif isinstance(args, tuple):
                if not len(args) == 1:
                    "warning, something went wrong with parsing the __getitem__"
                for this_list in self.list_of_lists:
                    this_list_selected = []
                    for idx in args[0]:
                        this_list_selected.append(this_list[idx])
                    return_list_stuff.append(this_list_selected)

            else:
                # can use the args as a normal index (i.e. int, float or slice)
                for this_list in self.list_of_lists:
                    return_list_stuff.append(this_list[args])

        if self.list_of_arrays is not None:
            # for arrays, it's basically the same stuff
            return_array_stuff = []
            for this_array in self.list_of_arrays:
                return_array_stuff.append(this_array[args])

        if self.list_of_trajs is not None:
            return_traj_stuff = []
            for this_traj in self.list_of_trajs:
                return_traj_stuff.append(this_traj[args])

        return return_list_stuff, return_array_stuff, return_traj_stuff


class DataObjectList(DataObjectBase):
    def __init__(self, list_of_lists):
        super(DataObjectList, self).__init__(list_of_lists, None, None)

    def __getitem__(self, args):
        list_stuff, array_stuff, traj_stuff = super(DataObjectList, self).__getitem__(args)

        new_object = DataObjectList(list_stuff)
        # return a copy of itself

        return new_object

class DataObjectArrays(DataObjectBase):
    def __init__(self, list_of_lists):
        super(DataObjectList, self).__init__(None, list_of_lists, None)

    def __getitem__(self, args):
        list_stuff, array_stuff, traj_stuff = super(DataObjectList, self).__getitem__(args)

        new_object = DataObjectList(array_stuff)
        # return a copy of itself

        return new_object
