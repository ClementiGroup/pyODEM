pyfexd
======

This package, pyfexd, adjusts model parameters based upon experimental data. The method was developed in the Clementi Group at Rice University. 

Main packages and methods to be aware of for the end user is:

model_loaders
-------------

Modules with methods for loading simulation data and analyzing it.

Currently supports:

Langevin: 1-D langevin dynamics data. See package TensorDuck/langevin_model.

observables
-----------

Modules for loading experimental results, computing Q values, and computing observables from simulation data.

Observable types currently supported:

ExperimentalObservables.add_histogram(): Adds a histogram data and associated observables for 1-D position data.

estimators
----------

Modules for estimating the Maximum Likelihood set of parameters for a model based on some experimental data (obsevables). 

Currently supports several different solvers, see documentation. 


Examples
--------

example_1: Compute the Q-value for the data files present in the folder.

example_2: Compute a new set of model parameters for the data files present in the folder.




