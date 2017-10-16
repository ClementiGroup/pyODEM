""" Example for analyzing a 1d langevin dynamics model

This script analyzes the langevin 1-d run and outputs a new set of parameters
based on the simulation run.
"""
import numpy as np
import scipy.stats as stats
import os

import pyfexd
ml = pyfexd.model_loaders
observables = pyfexd.observables
ene = pyfexd.estimators.max_likelihood_estimate


#load the model and load the data per the model's load_data method.
lmodel = ml.Langevin("simple.ini")
iteration = lmodel.model.iteration
data = lmodel.load_data("iteration_%d/position.dat" % iteration)


#load the observable object that calculates the observables of a set of simulation data
edges = np.loadtxt("edges.dat")
obs = observables.ExperimentalObservables()
obs.add_histogram("exp_data.dat", edges=edges, errortype="gaussian") #load and format the data distribution

#do a simple discretizaiton fo the data into equilibrium distribution states.
#In theory, the user will be able to specify any sort of equlibrium states for their data
hist, tempedges, slices = stats.binned_statistic(data, np.ones(np.shape(data)[0]), statistic="sum")
possible_slices = np.arange(np.min(slices), np.max(slices)+1)
equilibrium_frames = []
indices = np.arange(np.shape(data)[0])
for i in possible_slices:
    state_data = indices[slices == i]
    if not state_data.size == 0:
        equilibrium_frames.append(state_data)




#Now we can compute the set of epsilons that satisfy the max-likelihood condition
#set logq=True for using the logarithm functions
#currently, options are simplex, annealing, cg
solutions = ene(data, equilibrium_frames, obs, lmodel, solver="simplex", logq=False)
new_eps = solutions.new_epsilons
old_eps = solutions.old_epsilons
Qold = solutions.oldQ
Qnew = solutions.newQ
Qfunction = solutions.Qfunction_epsilon #-Q function
Qfunction_log = solutions.log_Qfunction_epsilon #-log(Q) function

print "Epsilons are: "
print new_eps
print old_eps

print ""
print "Qold: %g" %Qold
print "Qnew: %g" %Qnew

savestr = "iteration_%d/newton" % iteration
if not os.path.isdir(savestr):
    os.mkdir(savestr)
np.savetxt("%s/params" % savestr, np.append([1.0], new_eps))












