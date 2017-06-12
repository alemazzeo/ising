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

from ising import Sample, State, Simulation
from bimodal import Bimodal

class Analysis():
    def __init__(self, data):

        if isinstance(data, Simulation):
            self._sample_names = data._sample_names
            self._state_names = data._state_names
        elif isinstance(data, tuple):
            self._sample_names = data[0]
            self._state_names = data[1]

        self._results = list()
        self._states = list()

        for name in self._sample_names:
            self._results.append(Result(name))
        for name in self._state_names:
            self._states.append(State.load(name))

        self._current = 0
        self._n = len(self._results)

        self._figs = list()
        self._axs = list()
        self._funcs = list()

    def __len__(self): return len(self._results)
    def __getitem__(self, key):
        return self._results[key], self._states[key]

    def subplot(self, func, *args, **kwargs):
        plt.ion()
        fig , ax = plt.subplots(*args, **kwargs)
        self._figs.append(fig)
        self._axs.append(ax)
        self._funcs.append(func)
        fig.canvas.mpl_connect('close_event',
                               lambda evt: self._on_close(evt, fig))
        fig.canvas.mpl_connect('key_press_event',
                               lambda evt: self._on_key_press(evt, fig))
        self._update()
        return fig, ax, func

    def _on_key_press(self, evt, fig):
        idx = self._figs.index(fig)
        if evt.key == 'left':
            self.left()
        if evt.key == 'right':
            self.right()

    def _on_close(self, evt, fig):
        idx = self._figs.index(fig)
        self._figs.pop(idx)
        self._axs.pop(idx)
        self._funcs.pop(idx)

    def _update(self):
        for i, func in enumerate(self._funcs):
            func(self._figs[i], self._axs[i], *self[self._current])

    @property
    def current(self): return self._current
    @current.setter
    def current(self, value):
        assert 0 < value < self._n
        self._current = value
        self._update()

    def right(self, n=1):
        if self._current + n < self._n:
            self.current += n

    def left(self, n=1):
        if self._current - n > 0:
            self.current -= n

class Result():
    def __init__(self, sample_name):
        self._sample_name = sample_name

    @property
    def magnet_array(self):
        sample = Sample.load(self._sample_name)
        return sample.magnet

    @property
    def energy_array(self):
        sample = Sample.load(self._sample_name)
        return sample.energy
