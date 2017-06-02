import numpy as np
import matplotlib.pyplot as plt
from liveplot import LivePlot, Curve
from ising import Lattice
import time

# Setup

lat_size = 32

J = 1.0
B = 0.0

ti = 3.0
tf = 0.5
temp_paso = -0.05
temps = np.arange(ti, tf, temp_paso)
samples = list()

sample_size = 10000

preterm = 50000
term = 10000
step = 1500

# Initialization

lat = Lattice(lat_size)
lat.fill_random()

lat.J = J
lat.B = B
lat.T = ti

# Pretermalization

lat.run_until(preterm)

# Collecting samples

for T in np.nditer(temps):
    print('T= ' + str(T) + ' '*10, end='\r')

    # Set T and run until termalization
    lat.T = T
    lat.run_until(term)

    # Fill a sample
    sample = lat.run_sample(sample_size, step)
    # Store sample in a list
    samples.append(sample)
