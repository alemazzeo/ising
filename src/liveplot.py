import numpy as np
import ctypes as C
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class LivePlot():
    def __init__(self, name='Untitled'):
        self._fig = None
        self._name = name
        self._subplots = dict()
        self._plots = dict()
        self._matplots = dict()

        self._events = {'close_event': None,
                        'button_press_event': None,
                        'motion_notify_event': None}

    def _generate(self):
        plt.ion()
        self._fig = plt.figure(self._name)

        for name, subplot in self._subplots.items():
            subplot._ax = self._fig.add_subplot(*subplot._args, **subplot._kargs)
            for name, curve in subplot._curves.items():
                dataset, = subplot._ax.plot(curve.xdata, curve.ydata, **curve._config)
                self._plots[name] = dataset

    def _update(self):
        for name, subplot in self._subplots.items():
            for name, curve in subplot._curves.items():
                self._plots[name].set_data(curve.xdata, curve.ydata)
            subplot._ax.relim()
            subplot._ax.autoscale()
        plt.draw()

    def _connect_event(self, **kargs):
        for event, function in kargs.items():
            self._events[event] = self._figure.canvas.mpl_connect(event, function)

    def _disconnect_event(self, *args):
        for event in args:
            event_id = self._events[event]
            self._fig.canvas.mpl_disconnect(event_id)

    def add_subplot(self, name, pos=111, **config):
        subplot = Subplot(name, self, pos, **config)
        self._subplots.update({name:subplot})
        return subplot

    def save(self, file_name=None, path='./Plots/', file_type='png'):
        self._figure.savefig(path + file_name + file_type)

class Subplot():
    def __init__(self, name, liveplot, *args, **kargs):
        self._name = name
        self._liveplot = liveplot
        self._args = args
        self._kargs = kargs
        self._curves = dict()
        self._ax = None

    def plot(self, curve, **config):
        self._curves.update({curve._name: curve})
        curve._connect(self, **config)

    def _update(self):
        self._liveplot._update()

class Curve():
    def __init__(self, name):
        self._name = name
        self._data = None
        self._xdata = None
        self._ydata = None
        self._config = dict()
        self._subplots = dict()

    @property
    def data(self):
        if isinstance(self._data, np.ndarray):
            return self._data
        elif isinstance(self._data, list):
            return np.array(self._data)

    @data.setter
    def data(self, value):
        if isinstance(value, np.ndarray):
            data = value
        else:
            data = np.array(value)
        if len(data.shape) == 1:
            self.ydata = data
        elif len(data.shape) == 2:
            self.xdata = data[0]
            self.ydata = data[1]

    @property
    def xdata(self):
        if self._xdata is None:
            if self._ydata is not None:
                return np.arange(len(self._ydata))
        else:
            return np.array(self._xdata)

    @xdata.setter
    def xdata(self, value):
        self._xdata = np.array(value)
        self._update()

    @property
    def ydata(self):
        return np.array(self._ydata)

    @ydata.setter
    def ydata(self, value):
        self._ydata = np.array(value)
        self._data = self._ydata
        self._update()

    def _update(self):
        for name, subplot in self._subplots.items():
            subplot._update()

    def _connect(self, subplot, **config):
        self._config.update(config)
        self._subplots.update({subplot._name: subplot})

    def _disconnect(self, subplot):
        self._subplots.pop(subplot._name)

    @property
    def label(self): return self._label
