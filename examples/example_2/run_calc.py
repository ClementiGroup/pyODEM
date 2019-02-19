""" Example for analyzing a 1d langevin dynamics model"""
import numpy as np
import scipy.stats as stats
import os
import mdtraj as md
import shutil
import configparser
import model_builder as mdb
import argparse as argparse
import pyemma.coordinates as coor

import pyfexd
model_name = "ww_domain.ini"

def save_model(inifile, newiter, newfile_location):
    config = configparser.SafeConfigParser(allow_no_value=True)
    config.read(inifile)

    config.set("fitting", "iteration", str(newiter))
    newparams = "%s/model_params" % newfile_location
    newpairwise = "%s/pairwise_params" % newfile_location

    config.set("model", "pairwise_params_file", newpairwise)
    config.set("model", "model_params_file", newparams)



    shutil.move(inifile,"1.%s" %inifile)

    with open(inifile,"w") as cfgfile:
        config.write(cfgfile)

parser = argparse.ArgumentParser()
parser.add_argument("--temperature", default=None, type=int)

args = parser.parse_args()
temperature = args.temperature

ml = pyfexd.model_loaders
observables = pyfexd.observables
ene = pyfexd.estimators.max_likelihood_estimate

cwd = os.getcwd()
#Pro6-Arg14,  Phe21-Ser28
#file_pairs = [[79,211], [336,444]]
#fit_pairs = [[9, 24], [37,51]]
#inverse = [False, False]
file_pairs = [[79,211]]
fit_pairs = [[9, 24]]
inverse = [False]



num_pairs = len(file_pairs)
assert len(fit_pairs) == num_pairs

# initialize observables object
obs = observables.ExperimentalObservables()

# initialize pmodel object
pmodel = ml.Protein(model_name)
pmodel.set_temperature(temperature)
iteration = pmodel.fittingopts["iteration"]

# add qdata to the beginning of everything
qdata = np.loadtxt("%s/q_analysis/I%d_T%d_qvalue.dat" % (cwd,iteration, temperature))

qexp_file = "%s/q_analysis/exp_data_shaw.dat" % (cwd)
qedges_file = "%s/q_analysis/edges_I0_T121.dat" % (cwd)

obs.add_histogram(qexp_file, edges=np.loadtxt(qedges_file), errortype="gaussian", scale=200.0)

for i in range(num_pairs):
    file_pair = file_pairs[i]
    if inverse[i]:
        suffix = "inverse"
    else:
        suffix = "distance"
    edges = np.loadtxt("%s/edges_%d-%d.dat"%(suffix,file_pair[0],file_pair[1]))
    obs.add_histogram("%s/exp-data_%d-%d.dat"%(suffix,file_pair[0],file_pair[1]), edges=edges, errortype="gaussian", scale=1000.0) #load and format the data distribution


temp_directory = "%s/iteration_%d/%d" % (cwd, iteration, temperature)
os.chdir(temp_directory)
#load the model and load the data per the model's load_data method.

data_directory = temp_directory
os.chdir(data_directory)
data = pmodel.load_data("traj.xtc")

traj = md.load("traj.xtc", top="conf.gro")

obs_data = []
obs_data.append(qdata)
for i in range(num_pairs):
    dist = md.compute_distances(traj, [fit_pairs[i]], periodic=False)[:,0]
    if inverse[i]:
        obs_data.append(1./dist)
    else:
        obs_data.append(dist)


#load the observable object that calculates the observables of a set of simulation data

#do a simple discretizaiton fo the data into equilibrium distribution states.
#In theory, the user will be able to specify any sort of equlibrium states for their data

all_dist = np.array(obs_data).transpose()
reg_space_obj = coor.cluster_regspace(all_dist, dmin=0.05)
dtrajs = np.array(reg_space_obj.dtrajs)[0,:]
assert np.min(dtrajs) == 0
assert np.shape(dtrajs)[0] == np.shape(data)[0]
print(("Number of equilibrium states are : %d" % (np.max(dtrajs))))
equilibrium_frames = []
indices = np.arange(np.shape(data)[0])
for i in range(np.max(dtrajs)+1):
    state_data = indices[dtrajs == i]
    if not state_data.size == 0:
        equilibrium_frames.append(state_data)

total_check = 0
for set_of_frames in equilibrium_frames:
    total_check += len(set_of_frames)
assert total_check == np.shape(data)[0]

bounds = []
for i,pairwise_pair in enumerate(pmodel.model.Hamiltonian._pairs):
    highest = 2
    if pairwise_pair.prefix_label == "LJ12GAUSSIAN":
        lowest = 0
    else:
        if pairwise_pair.eps >= -0.09:
            lowest = -0.1
        else:
            lowest = -highest
    lower_bound = pmodel.epsilons[i] - 0.3
    if lower_bound < lowest:
        lower_bound = lowest
    upper_bound = pmodel.epsilons[i] + 0.3
    if upper_bound > highest:
        upper_bound = highest
    bounds.append([lower_bound,upper_bound])

#Now we can compute the set of epsilons that satisfy the max-likelihood condition
#set logq=True for using the logarithm functions
#currently, options are simplex, annealing, cg
function_args = {"bounds":bounds, "gtol":0.001}
try:
    solutions = ene(data, equilibrium_frames, obs, pmodel, obs_data=obs_data, solver="bfgs", logq=True, kwargs=function_args)
    save_results = True
except:
    print("FAILURE TO OPTIMIZE")
    raise
    save_results = False

if save_results:
    new_eps = solutions.new_epsilons
    old_eps = solutions.old_epsilons
    Qold = solutions.oldQ
    Qnew = solutions.newQ
    Qfunction = solutions.log_Qfunction_epsilon

    print("Epsilons are: ")
    print(new_eps)
    print(old_eps)

    print("")
    print(("Qold: %g" %Qfunction(old_eps)))
    print(("Qnew: %g" %Qfunction(new_eps)))
    pmodel.save_model_parameters(new_eps)
    save_dir = "%s/newton_%d" % (cwd, iteration)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    os.chdir(save_dir)
    writer = mdb.models.output.InternalFiles(pmodel.model)
    writer.write_pairwise_parameters()
    f = open("info.txt", "w")
    f.write("computed at temperature: %d\n" % temperature)
    f.write("Qold: %g\n" % Qfunction(old_eps))
    f.write("Qnew: %g\n" % Qfunction(new_eps))
    f.close()
    ##save the new parameters
    os.chdir(cwd)
    save_model(model_name, iteration+1, "newton_%d" % iteration)
os.chdir(cwd)
