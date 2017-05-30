import numpy as np
import matplotlib.pyplot as plt
from liveplot import LivePlot, Curve
from ising import Lattice
import time

lat = Lattice(32)
lat.fill_random()

lat.step_size = 50000
lat.T = 5.0
lat.run()
#lat.lattice.matshow()
#lat.energy.plot()
#lat.magnet.plot()

ti = 5
tf = 0.5
temp_paso = -0.05
temps = np.arange(ti, tf, temp_paso)

n_samples = 10000
magnet = np.zeros([len(temps),n_samples])

lat.step_size = 10000
for i, T in enumerate(temps):
    lat.T = T
    print(str(T) + ' '*20, end='\r')
    for j in range(n_samples):
        lat.run()
        magnet[i][j] = lat.current_magnet
    #plt.pause(0.0001)
