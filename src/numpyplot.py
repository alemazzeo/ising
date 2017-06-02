import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class NumpyPlot(np.ndarray):
    def __new__(cls, input_array, **plotconfig):
        obj = np.asarray(input_array).view(cls)
        obj._plot_config = dict()
        obj._plot_config.update(plotconfig)
        obj.figure = None
        obj.subplot = None
        obj._events = {'close_event': None,
                       'button_press_event': None,
                       'motion_notify_event': None}
        obj._active = False
        return obj

    def set_figure(self, *args, **kwargs):
        self.figure = plt.figure(*args, **kwargs)
        self.figure.clear()

    def add_subplot(self, *args, **kwargs):
        if self.figure is not None:
            self.figure.add_subplot(*args, **kwargs)

    def _exist_figure(self):
        if self.figure is None:
            self.set_figure()
            plt.ion()
            self._active = True
            self.connect_event(close_event=self._on_close)

    def plot(self, *args, **kwargs):
        self._exist_figure()
        if self.subplot is None:
            plt.plot(self, *args, **kwargs)
        else:
            self.subplot.plot(self, *args, **kwargs)
        self._update()

    def matshow(self, *args, **kwargs):
        self._exist_figure()
        if self.subplot is None:
            plt.matshow(self, *args, **kwargs)
        else:
            self.subplot.matshow(self, *args, **kwargs)
        self._update()

    def _update(self):
        if self._active:
            plt.figure(self.figure.number)
            plt.draw()

    def __array_wrap__(self, out_arr, context=None):
        self._update()

    def _on_close(self, evt):
        self._active = False

    def connect_event(self, **kargs):
        for event, function in kargs.items():
            self._events[event] = self.figure.canvas.mpl_connect(event, function)

    def disconnect_event(self, *args):
        for event in args:
            event_id = self._events[event]
            self.figure.canvas.mpl_disconnect(event_id)
