
class Estimators():
    """
    This class ia an interface between all other classes, that holds all the required
    information for the optimization
    """

    def __init__(self,Q_function,solver,epsilons):
        self.Q_function = Q_function
        self.solver = solver
        self.epsilons_old = epsilons
        self.Q_function_old = Q_function(epsilons)
        self.epsilons_new = None
        self.Q_function_new = None

    def find_solution(self,**kwargs):
        """
        Method is used to find new solution
        """
        self.epsilons_new = self.solver(self.Q_function,self.epsilons_old,**kwargs)
        self.Q_function_new = self.Q_function(self.epsilons_new)
