"""
Class for holding and using for calculating the Q_function
"""

class q_value(object):
    """ An object that holds a list of functions for calculating the q value, and can calulcate the q value for some set of new values"""
    def __init__(func_list):
        """initiate with a list of functions, or can use an empty list"""
        function_list = func_list #List of functions formatted for a single input variable f(r)
        num_func = len(function_list)
        
    def add_function(func):
        """Add a function to the list"""
        function_list.append(func)
        num_func += 1 
        
    def get_Q(value_list):
        """calculate the Q value for some list of new observed values"""
        if not len(value_list) == num_func:
            raise IOError("Number of values inputted not equal to number of functions")
        
        Q = 1.0
        for i in range(num_func):
            Q *= function_list[i](value_list[i])
        
        return Q