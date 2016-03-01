""" Example for analyzing a 1d langevin dynamics model"""

import max_likelihood.model_loaders as ml
import max_likelihood.observables as observables


#load the model and load the data per the model's load_data method.
lmodel = ml.Langevin("config.ini")
data = lmodel.load_data("iteration_0/positions.dat")

#load the observable object that calculates the observables of a set of simulation data
obs = observables.Obs()
obs.add_histogram("observed_histogram.dat", nbins=400, histrange=(-20.0,20.0), error_type="gaussian") #load and format the data distribution

obsval, obsstd = obs.compute_observations(data, range=(-20.0, 20.0), nbins=400)


#do a simple discretizaiton fo the data into equilibrium distribution states.

slices = obs.observables[0].slices

possible_slices = np.arange(np.min(slices), np.max(slices)+1)

equilibrium_frames = []

for i in possible_slices:
    data[slices == i]
    equilibrium_frames.append(data)




#Now we can compute the set of epsilons that satisfy the max-likelihood condition
import max_likelihood.estimate_new_epsilons
