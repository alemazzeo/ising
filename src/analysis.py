import matplotlib
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    try:
        matplotlib.use('qt4Agg')
    except ImportError:
        print("'Qt5Agg' or 'qt4Agg' request for interactive plot")
        raise

import numpy as np
import matplotlib.pyplot as plt

from ising import Sample, Simulation
from bimodal import Bimodal

class Analysis():
    def __init__(self, data):

        if isinstance(data, Simulation):
            self._sample_names = data._sample_names
            self._state_names = data._state_names
        elif isinstance(data, tuple):
            self._sample_names = data[0]
            self._state_names = data[1]

        self._samples = list()
        self._states = list()

        for name in self._sample_names:
            self._samples.append(Sample.load(name))
        for name in self._state_names:
            self._states.append(State.load(name))

class Result():
    def __init__(self, sample_name):
        self._sample_name = sample_name
        self._sample = Sample.load(sample_name)

        self._energy = self.fit_energy()

        self._magnet = self.fit_magnet()


    def fit_energy(self, plot=False):
        energy = self._sample.energy
        A = 0.3
        mu = np.mean(energy)
        sd = np.sqrt(np.var(energy))

        expected = [mu, sd, A]

        params, sigma = Bimodal.fit_gaussian(energy,
                                             expected=[mu, sd, A],
                                             plot=plot)

        mu = [params[0], sigma[0]]
        sd = [params[1], sigma[1]]
        A = [params[2], sigma[2]]

        return mu, sd, A

    def fit_magnet(self, plot=False):
        magnet = self._sample.magnet

        peaks, xpeaks, ypeaks = Bimodal.estimate_pdf_peaks(magnet,
                                                           plot=plot)

        return xpeaks
