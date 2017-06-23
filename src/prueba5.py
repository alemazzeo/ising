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
from analysis import Analysis, Result, Tools
import numpy as np
import matplotlib.pyplot as plt

sim1 = Simulation.load('../data/simulations/Simulation0.npy')
analysis = Analysis(sim1)


def test(fig, axs, result, state):

    for ax in axs:
        ax.cla()

    result.fit(axs[0], axs[1], axs[2])

    for ax in axs:
        ax.relim()
        ax.autoscale()
    plt.draw()


analysis.subplot(test, 3, 1)
