import matplotlib
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    try:
        matplotlib.use('qt4Agg')
    except ImportError:
        print("'Qt5Agg' or 'qt4Agg' request for interactive plot")
        raise

from ising import Simulation, Ising, State, Sample
from analysis import Analysis, Result
from bimodal import Bimodal
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, mode

sim1 = Simulation.load('../data/simulations/1stSweep0.npy')

plt.ion()
fig, ax = plt.subplots(1)

for i in range(35):
    res = Result(sim1[i][0])
    magnet = res._sample.magnet/1024
    ax.cla()

    ax.hist(magnet, normed=True)
    x = np.linspace(-1,1,100)
    params = beta.fit(magnet)
    rv = beta(*params)
    modes = mode(magnet)
    print (modes)
    ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

    ax.relim()
    ax.autoscale()
    plt.draw()
    plt.pause(0.01)
    print(i)
    print(*params)
    print('')
