from pyODEM.observables import Observable

class AverageO(Observable):
    
    def __init__(self):
        pass

    def compute_observed(self, value, stdev):
        seen = [True for i in value]
        print(seen)
        return value, stdev, seen