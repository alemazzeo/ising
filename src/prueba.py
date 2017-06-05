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
tf = 2.8
temp_step = -0.05
temps = np.arange(ti, tf, temp_step)
samples = list()

sample_size = 10000

pretherm = 50000
therm = 10000
step = 1500

# Initialization

ising1 = Ising(lat_size)
ising1.fill_random()

ising1.J = J
ising1.B = B
ising1.T = ti

# Prethermalization

ising1.run_until(pretherm)

# Collecting samples

for T in np.nditer(temps):
    print('T= ' + str(T) + ' '*10, end='\r')

    # Set T and run until thermalization
    ising1.T = T
    ising1.run_until(therm)

    # Fill a sample
    sample = ising1.run_sample(sample_size, step)
    # Store sample in a list
    samples.append(sample)
