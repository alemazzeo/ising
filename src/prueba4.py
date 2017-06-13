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
import time

sim1 = Simulation.load('../data/simulations/TestSim0.npy')
analysis = Analysis(sim1)

def classificate(data, ax):

    # Peaks of pdf
    peaks, xpeaks, ypeaks = Bimodal.estimate_pdf_peaks(data,plot=True, ax=ax)

    peaks = np.asarray(peaks)
    xpeaks = np.asarray(xpeaks)
    ypeaks = np.asarray(ypeaks)

    n = peaks.size

    mu = 0.0
    A = 0.1

    if n == 0:
        return 'Bimodal', mu, A

    elif n == 1:
        if xpeaks[0] > 0.5:
            mu = xpeaks[0]
            A = ypeaks[0]
            return 'Positive', mu, A

        elif xpeaks[0] < -0.5:
            mu = xpeaks[0]x
            A = ypeaks[0]
            return 'Negative', mu, A

        else:
            mu1 = abs(xpeaks[0])
            A = ypeaks[0]
            return 'Bimodal', mu, A

    else:

        xp = xpeaks[xpeaks>=0]
        Ap = ypeaks[xpeaks>=0]
        xn = xpeaks[xpeaks<0]
        An = ypeaks[xpeaks<0]

        print(xp, Ap, xn, An)

        if xp.size == 0:
            mu = xn[0]
            A = An[0]
            return 'Negative', mu, A

        elif xn.size == 0:
            mu = xp[-1]
            A = Ap[-1]
            return 'Positive', mu, A

        else:
            if -xn[0] > xp[-1]:
                mu = abs(xn[0])
                A = An[0]
            else:
                mu = xp[-1]
                A = Ap[-1]
            return 'Bimodal', mu, A

def test(fig, ax, result, state):
    magnet = result.magnet_array/1024
    ax.cla()

    # Histrogram
    Y, X, _ = ax.hist(magnet, normed=True, alpha=0.3)

    pdf_type, mu, A = classificate(magnet, ax)
    print(pdf_type, mu, A)

    ax.axvline(mu, color='r', lw=5, alpha=0.5)
    if pdf_type == 'Bimodal':
        ax.axvline(-mu, color='r', lw=5, alpha=0.5)
        if mu > 0.5:
            pass
        else:
            mins = [-1, 0.01, 0, -1, 0.01, 0]
            maxs = [1, 0.5, np.inf, 1, 0.5, np.inf]
            Bimodal.fit_bimodal(magnet, expected=[mu,0.3,A/2,-mu,0.3,A/2],
                                bounds=bounds, plot=True, ax=ax)

    elif pdf_type == 'Positive':
        Bimodal.fit_gaussian(magnet, expected=[mu,0.3,A], plot=True, ax=ax)

    elif pdf_type == 'Negative':
        Bimodal.fit_gaussian(magnet, expected=[mu,0.3,A], plot=True, ax=ax)

    ax.relim()
    ax.autoscale()
    plt.draw()

analysis.subplot(test, 1,1)
