import model_builder as mdb
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import os

savedir = "q_analysis"

def compute_Q_traj(traj, r0, pairs, potentials):
    distances = md.compute_distances(traj, pairs, periodic=False)
    Q_matrix = np.zeros(np.shape(distances))
    for idx in range(np.shape(distances)[0]):
        for jdx in range(np.shape(distances)[1]):
            if distances[idx,jdx] > (r0[jdx]*1.25):
                Q_matrix[idx,jdx] = 0
            else:
                Q_matrix[idx,jdx] = 1
            assert Q_matrix[idx, jdx] >= 0
    Q_val = np.sum(Q_matrix, axis=1) / float(len(pairs))
    assert np.shape(Q_val)[0] == traj.n_frames
    return Q_val
    
model, fitopts = mdb.inputs.load_model("ww_domain.ini")

pairs_list = []
r0_list = []
array_pairs = np.loadtxt("fip35_contacts.dat")
pairs = []
for idx in range(np.shape(array_pairs)[0]):
    pairs.append([array_pairs[idx,0]-1, array_pairs[idx,1]-1])

for pot in model.Hamiltonian._pairs:
    if [pot.atmi.index,pot.atmj.index] in pairs or [pot.atmj.index,pot.atmi.index] in pairs:
        r0 = pot.r0
        v0 = pot.dVdeps(r0)
        pairs_list.append([pot.atmi.index, pot.atmj.index])
        r0_list.append(r0)
if not len(pairs) == len(pairs_list):
    print("Some of our pairs are not found!!")
    print(len(pairs))
    print(len(pairs_list))
    for pa in pairs:
        if not pa in pairs_list:
            print(pa)
print(pairs_list)
print(pairs)

if not os.path.isdir(savedir):
    os.mkdir(savedir)

for iteration in [fitopts["iteration"]]:
    for temp in np.arange(50, 300,1):  
        print("Analyze Iteration: %d, Temperature : %d K" % (iteration,temp))  
        traj_file = "iteration_%d/%d/traj.xtc"%(iteration, temp)
        if os.path.isfile(traj_file):
            traj = md.load("iteration_%d/%d/traj.xtc"%(iteration,temp), top="iteration_%d/%d/conf.gro"%(iteration,temp))
            Q = compute_Q_traj(traj, r0_list, pairs_list, model.Hamiltonian._pairs)
            plt.figure()
            plt.plot(Q)
            plt.ylabel("Q fraction")
            plt.xlabel("frame")
            plt.axis([0, traj.n_frames, 0, 1.5])
            plt.savefig("%s/I%d_T%d_trace.png"%(savedir,iteration, temp))
            hist, edges = np.histogram(Q, range=(0,1), bins=25, density=False)
            hist = hist.astype(float)
            hist /= np.sum(hist)
            fe = -np.log(hist)
            min_val = np.min(fe)
            fe = fe - min_val
    
            centers = edges[1:] + edges[:-1]
            centers /= 2.
            
            fe_inf = np.isinf(fe)
            fe_save = []
            centers_save = []
            for i in range(np.shape(fe_inf)[0]):
                if not fe_inf[i]:
                    fe_save.append(fe[i])
                    centers_save.append(centers[i])
            
            fe = np.array(fe_save)
            centers = np.array(centers_save)
            
            try:
                assert np.all(fe>=0)
            except:
                print(fe)
                print(hist)
                raise
            max_val = np.max(fe)
            plt.figure()
            print(centers)
            print(fe)
            plt.plot(centers, fe)
            plt.ylabel("Free Energy (kT)")
            plt.xlabel("Q")
            plt.axis([0, 1, 0, max_val])
            plt.savefig("%s/I%d_T%d_hist.png" % (savedir,iteration, temp))
            np.savetxt("%s/I%d_T%d_qvalue.dat" % (savedir, iteration, temp), Q)
