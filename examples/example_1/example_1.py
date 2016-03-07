""" Basic example of a run through using the methods in this package"""

##import the basic things
import numpy as np
import pyfexd
bf = pyfexd.basic_functions
ml = pyfexd.model_loaders
observables = pyfexd.observables
ene = pyfexd.max_likelihood.estimate_new_epsilons

#load a histogram data
edges = np.loadtxt("edges.dat")
obs = observables.ExperimentalObservables()
obs.add_histogram("exp_data.dat", edges=edges, errortype="gaussian") #load and format the data distribution

qcalc = obs.get_q_function()

regdata = np.array([20, 110, 300, 90, 30])

print qcalc(regdata)
regdata[0] = 10
print qcalc(regdata)
regdata[0] = 10.0
print qcalc(regdata)
regdata[2] = 282.0
print qcalc(regdata)
