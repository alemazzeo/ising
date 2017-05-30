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
lat.lattice.matshow()
lat.energy.plot()
lat.magnet.plot()

m = list()
lat.step_size = 100
for i in range (100):
    lat.T = lat.T - 0.1
    m.append(lat._magnet[-1])
    lat.run()
    plt.pause(0.0001)
