import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class NumpyPlot(np.ndarray):
    def __new__(cls, input_array, **plotconfig):
        obj = np.asarray(input_array).view(cls)
        obj._plot_config = dict()
        obj._plot_config.update(plotconfig)
        obj._plot_data = None
        obj.figure = None
        obj.subplot = None
        obj._events = {'close_event': None,
                       'button_press_event': None,
                       'motion_notify_event': None}
        obj._active = False
        return obj

    def set_figure(self, *args, **kwargs):
        self.figure = plt.figure(*args, **kwargs)
        plt.ion()

    def add_subplot(self, *args, **kwargs):
        if self.figure is not None:
            self.subplot = self.figure.add_subplot(*args, **kwargs)

    def _exist_figure(self):
        if self.figure is None:
            self.set_figure()
            self._active = True
            self.connect_event(close_event= lambda evt: self._on_close(evt))

    def _exist_subplot(self):
        if self.subplot is None:
            self.add_subplot(111)

    def plot(self, *args, **kwargs):
        self._exist_figure()
        self._exist_subplot()

        self._plot_data, = self.subplot.plot(self, *args, **kwargs)
        self._update()

    def matshow(self, *args, **kwargs):
        self._exist_figure()
        self._exist_subplot()
        self._plot_data, = self.subplot.matshow(self, *args, **kwargs)
        self._update()

    def _update(self):
        if self._active:
            self._plot_data.set_xdata(np.arange(len(self)))
            self._plot_data.set_ydata(self)
            self.subplot.relim()
            self.subplot.autoscale()
            plt.figure(self.figure.number)
            plt.draw()

    def __array_wrap__(self, out_arr, context=None):
        self._update()
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def _on_close(self, evt):
        self._active = False

    def connect_event(self, **kargs):
        for event, function in kargs.items():
            self._events[event] = self.figure.canvas.mpl_connect(event, function)

    def disconnect_event(self, *args):
        for event in args:
            event_id = self._events[event]
            self.figure.canvas.mpl_disconnect(event_id)
