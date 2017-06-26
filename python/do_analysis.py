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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sim', type=str,
                    default='Simulation2')

params = parser.parse_args()

sim1 = Simulation.load(params.sim)
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
