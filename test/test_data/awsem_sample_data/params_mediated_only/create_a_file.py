import numpy as np

data = np.loadtxt('gamma_propper.dat')
data [210:, :] = 1.0
data [:210, :] = 0.0

np.savetxt('gamma.dat', data)
