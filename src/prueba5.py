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
import time

sim1 = Simulation.load('../data/simulations/TestSim0.npy')
analysis = Analysis(sim1)

def test(fig, axs, result, state):
    magnet = result.magnet_array/1024

    for ax in axs:
        ax.cla()

    pdf_type, mu, A = Tools.classificate(magnet, plot=True, ax=axs[0])
    print(pdf_type, mu, A)

    axs[0].axvline(mu, color='r', lw=5, alpha=0.5)
    if pdf_type == 'Bimodal':
        axs[0].axvline(-mu, color='r', lw=5, alpha=0.5)

    try:
        params, sigma = Tools.prefit(magnet, pdf_type, mu, A, plot=True, ax=axs[1])
    except:
        pass
    try:
        params = Tools.fit(magnet, pdf_type, params, plot=True, ax=axs[2])
    except:
        pass

    for ax in axs:
        ax.relim()
        ax.autoscale()
    plt.draw()


analysis.subplot(test, 3,1)
