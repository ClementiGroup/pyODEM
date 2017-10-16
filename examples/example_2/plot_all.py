import numpy as np
import os
import mdtraj as md

import model_builder as mdb
import analysis_scripts.plot_package as pltpkg
import argparse

par = argparse.ArgumentParser()
par.add_argument("--iteration", type=int, default=0)
par.add_argument("--temperatures", type=int, nargs="+", default=None)
args = par.parse_args()

iteration = args.iteration
iters_use = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pair_list = [[9, 24], [1,29]]
pair_atoms_list = [[79,211], [11,254]]
suffixes = ["distance", "enhance"]

num_pairs_to_plot = len(pair_list)
assert num_pairs_to_plot == len(pair_atoms_list)
assert num_pairs_to_plot == len(suffixes)

#iteration =  2

cwd = os.getcwd()


if args.temperatures is None:
    temp_list_str = os.listdir("iteration_%d" % iteration)

    temperatures = [int(temp_str) for temp_str in temp_list_str]
    temperatures = np.sort(temperatures)
else:
    temperatures = args.temperatures
name1 = None
name2 = None





for idx in range(num_pairs_to_plot):
    name1 = None
    name2 = None
    pair = pair_list[idx]
    pair_atoms = pair_atoms_list[idx]

    data_dir = suffixes[idx]

    savedir = "%s/analysis-%s-%d-%d" % (cwd, data_dir, pair[0], pair[1])

    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    edges = np.loadtxt("%s/edges_%d-%d.dat"%(data_dir, pair_atoms[0], pair_atoms[1]))
    hist_ref = np.loadtxt("%s/exp-data_%d-%d.dat"%(data_dir, pair_atoms[0], pair_atoms[1]))[:,0]
    centers = edges[:-1] + edges[1:]
    centers /= 2.

    x = []
    y = []
    labels = []

    x.append(centers)
    y.append(hist_ref)
    labels.append("FRET")

    for temp in temperatures:
        traj_file = "iteration_%d/%d/traj.xtc" % (iteration, temp)
        top_file = "iteration_%d/%d/conf.gro" % (iteration, temp)

        if os.path.isfile(traj_file):
            traj = md.load(traj_file, top=top_file)
            dist = md.compute_distances(traj, [pair])
            hist, edges = np.histogram(dist, bins=edges, density=True)

            x.append(centers)
            y.append(hist)
            labels.append("temp-%d"%temp)

            if name1 is None:
                res1 = traj.top.atom(pair[0]).residue
                res2 = traj.top.atom(pair[1]).residue
                res1name = res1.name
                res2name = res2.name
                res1idx = res1.index + 1
                res2idx = res2.index + 1
                name1 = "%s%d" % (res1name, res1idx)
                name2 = "%s%d" % (res2name, res2idx)

    os.chdir(savedir)
    pltpkg.plot_simple(x,y,labels, "%s-%s"%(name1,name2) , "Distance (nm)"
        , "probability", save_file="iter-%d"%(iteration)
        , reference=True, show=False)

    os.chdir(cwd)
