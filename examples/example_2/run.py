""" Example for analyzing a 1d langevin dynamics model"""

import max_likelihood.model_loaders as ml
import max_likelihood.observables as observables


lmodel = ml.Langevin("config.ini")

data = lmodel.load_data("iteration_0/positions.dat")

obs = observables.Obs()
obs.add_histogram(nbins=400, histrange=(-20.0,20.0))

obsval, obsstd = obs.compute_observations(data, range=(-20.0, 20.0), nbins=400)

slices = obs.observables[0].slices

possible_slices = np.arange(np.min(slices), np.max(slices)+1)

equilibrium_frames = []

for i in possible_slices:
    data[slices == i]
    equilibrium_frames.append(data)

#Now we can compute the set of epsilons that satisfy the max-likelihood condition
import max_likelihood.