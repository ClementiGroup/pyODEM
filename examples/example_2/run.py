""" Example for analyzing a 1d langevin dynamics model"""
import numpy as np
import scipy.stats as stats

import pyfexd
ml = pyfexd.model_loaders
observables = pyfexd.observables
ene = pyfexd.max_likelihood.estimate_new_epsilons


#load the model and load the data per the model's load_data method.
lmodel = ml.Langevin("simple.ini")
data = lmodel.load_data("iteration_0/position.dat")

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



#Now we can compute the set of epsilons that satisfy the max-likelihood condition
#neweps = ene(data, equilibrium_frames, obs, lmodel)


#debug
new_eps, old_eps = ene(data, equilibrium_frames,obs,lmodel)



















