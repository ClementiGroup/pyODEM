""" Example for analyzing a 1d langevin dynamics model"""
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

print "Epsilons started as:"
print lmodel.epsilons

#load the observable object that calculates the observables of a set of simulation data
edges = np.loadtxt("edges.dat")
obs = observables.ExperimentalObservables()
obs.add_histogram("exp_data.dat", edges=edges, errortype="gaussian") #load and format the data distribution

#obsval, obsstd = obs.compute_observations(data, range=(-20.0, 20.0), nbins=400)


#do a simple discretizaiton fo the data into equilibrium distribution states.

hist, tempedges, slices = stats.binned_statistic(data, np.ones(np.shape(data)[0]), statistic="sum")
 

possible_slices = np.arange(np.min(slices), np.max(slices)+1)

equilibrium_frames = []
indices = np.arange(np.shape(data)[0])
for i in possible_slices:
    state_data = indices[slices == i]
    if not state_data.size == 0:
        equilibrium_frames.append(state_data)

print np.shape(equilibrium_frames)



#Now we can compute the set of epsilons that satisfy the max-likelihood condition
#neweps = ene(data, equilibrium_frames, obs, lmodel)


#debug
solutions = ene(data, equilibrium_frames, obs, lmodel)
new_eps = solutions.new_epsilons
old_eps = solutions.old_epsilons
Qold = solutions.Qfunction_epsilon(old_eps)
Qnew = solutions.Qfunction_epsilon(new_eps)

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



















