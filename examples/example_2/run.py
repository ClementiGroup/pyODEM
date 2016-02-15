""" Example for analyzing a 1d langevin dynamics model"""

import max_likelihood.model_loaders as ml
import max_likelihood.observables as obs


lmodel = ml.Langevin("config.ini")

data = lmodel.load_data("iteration_0/positions.dat")

hist_x_histogram(x, nbins=400, histrange=(-20.0,20.0))
obsval, obsstd = obs.histogram_distance(data, range=[(-20.0, 20.0)], nbins=400)



