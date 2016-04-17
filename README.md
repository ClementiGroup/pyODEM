pyfexd
======

This package, pyfexd, adjusts model parameters based upon experimental data. The method was developed in the Clementi Group at Rice University. 

#Prerequisites
The `TensorDuck/langevin_model` repository needs to be  cloned and in your PYTHONPATH variable for using the `model_loaders.Langevin()` loader.

The `ajkluber/model_builder` repository needs to be  cloned and in your PYTHONPATH variable for using the `model_loaders.Proteins()` loader.

Main packages and methods to be aware of for the end user is:

##model_loaders

Modules with methods for loading simulation data and analyzing it.

`model_loaders.Langevin()`: 1-D langevin dynamics data. See package TensorDuck/langevin_model.

`model_loaders.Proteins()`: Loading a protein topology and its associated potential functions. See package ajkluber/model_builder.

##observables

Modules for loading experimental results, computing Q values, and computing observables from simulation data.


`observables.ExperimentalObservables.add_histogram()`: Adds a histogram data and associated observables for 1-D position data.

##estimators


Modules for estimating the Maximum Likelihood set of parameters for a model based on some experimental data (obsevables). 

`estimators.max_likelihood_estimate()`: Estimates the most likely set of model parameters given the data.

Currently supports several different solvers. Options are: simplex, cg, anneal and custom. See documentation. 


##Examples

example_1: Compute the Q-value for the data files present in the folder.

example_2: Compute a new set of model parameters for the data files present in the folder.

To Run, use execute multirun.sh in the folder to run a langevin 1-D simulation using the current set of starting parameters. There are 3-steps per an iteration in multirun.sh:

`python -m langevin_model.simulation sim --name $name --steps 1000000`: Executes a langevin 1-d simulation. You can change the number of steps it takes. More steps means more data to analyze so it will take longer to simulate and analyze, with ~O(N) scaling. The `time` function is there for diagnostic purposes. It will generate a new folder `iteration_%d` where `%d` is the iteration number.

`python -m run`: Executes the analysis script in the folder called `run.py`. This will analyze using the max_likelihood method and output a new set of parameteres into `iteration_%d/newton/params`

`python -m langevin_model.simulation next --name $name --start`: Update the input files for the next simulation run (i.e. update model parameters).




