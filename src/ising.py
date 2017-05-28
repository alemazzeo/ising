import numpy as np
import ctypes as C
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import time
#import threading
#import os

class Lattice(C.Structure):
    _fields_ = [("_p_lattice", C.POINTER(C.c_int)),
                ("_n", C.c_int),
                ("_n2", C.c_int),
                ("_flips", C.c_int),
                ("_total_flips", C.c_int),
                ("_T", C.c_float),
                ("_J", C.c_float),
                ("_B", C.c_float),
                ("_p_dEs", C.c_float * 10),
                ("_p_exps", C.c_float * 10),
                ("_W", C.c_int),
                ("_N", C.c_int),
                ("_E", C.c_int),
                ("_S", C.c_int),
                ("_aligned", C.c_int),
                ("_p_energy", C.POINTER(C.c_float)),
                ("_p_magnet", C.POINTER(C.c_int))]

    def __init__(self, n, data='../datos/', lib='./libising.so'):

        # Origen de datos
        self._data = data

        # Valores por defecto
        self._step_size = 1000

        # Biblioteca de funciones
        self._lib = C.CDLL(lib)

        self.__init = self._lib.init
        self.__set_params = self._lib.set_params
        self.__info = self._lib.info
        self.__metropolis = self._lib.metropolis
        self.__pick_site = self._lib.pick_site
        self.__flip = self._lib.flip
        self.__find_neighbors = self._lib.find_neighbors
        self.__cost = self._lib.cost
        self.__try_flip = self._lib.try_flip
        self.__accept_flip = self._lib.accept_flip
        self.__calc_pi = self._lib.calc_pi
        self.__calc_energy = self._lib.calc_energy
        self.__calc_magnet = self._lib.calc_magnet
        self.__calc_lattice = self._lib.calc_lattice

        self.__init.restype = C.c_int
        self.__set_params.restype = C.c_int
        self.__info.restype = C.c_int
        self.__metropolis.restype = C.c_int
        self.__pick_site.restype = C.c_int
        self.__flip.restype = C.c_int
        self.__find_neighbors.restype = C.c_int
        self.__cost.restype = C.c_int
        self.__try_flip.restype = C.c_int
        self.__accept_flip.restype = C.c_int
        self.__calc_pi.restype = C.c_float
        self.__calc_energy.restype = C.c_float
        self.__calc_magnet.restype = C.c_int
        self.__calc_lattice.restype = C.c_int

        self.__init.argtypes = [C.POINTER(Lattice), C.c_int]
        self.__set_params.argtypes = [C.POINTER(Lattice), C.c_float, C.c_float, C.c_float]
        self.__info.argtypes = [C.POINTER(Lattice)]
        self.__metropolis.argtypes = [C.POINTER(Lattice), C.c_int]
        self.__pick_site.argtypes = [C.POINTER(Lattice)]
        self.__flip.argtypes = [C.POINTER(Lattice), C.c_int]
        self.__find_neighbors.argtypes = [C.POINTER(Lattice), C.c_int]
        self.__cost.argtypes = [C.POINTER(Lattice), C.c_int]
        self.__try_flip.argtypes = [C.POINTER(Lattice), C.c_float]
        self.__accept_flip.argtypes = [C.POINTER(Lattice), C.c_int, C.c_int]
        self.__calc_pi.argtypes = [C.POINTER(Lattice), C.c_int, C.c_int]
        self.__calc_energy.argtypes = [C.POINTER(Lattice), C.c_int]
        self.__calc_magnet.argtypes = [C.POINTER(Lattice), C.c_int]
        self.__calc_lattice.argtypes = [C.POINTER(Lattice)]

        # Memoria asignada a la red
        self._lattice = np.ones(n**2, dtype=C.c_int)
        self._p_lattice = self._lattice.ctypes.data_as(C.POINTER(C.c_int))
        self._energy = np.zeros(self._step_size, dtype=C.c_float)
        self._p_energy = self._energy.ctypes.data_as(C.POINTER(C.c_float))
        self._magnet = np.zeros(self._step_size, dtype=C.c_int)
        self._p_magnet = self._magnet.ctypes.data_as(C.POINTER(C.c_int))

        # Inicializa los valores
        self.__init(self, n)
        self.calc_lattice()

        # Lista de valores obtenidos (para graficar)
        self._list_T = list()
        self._list_E = list()
        self._list_M = list()
        self._list_C = list()

        # Figuras
        self._fig = None
        self._fig_energy = None
        self._fig_magnet = None

        # Subplot lattice
        self._subp_lattice = None
        # Dibujo de la matriz
        self._d_lat = None
        # Subplot energy
        self._subp_energy = None
        # Dibujo de energy
        self._d_energy = None
        # Subplot magnet
        self._subp_magnet = None
        # Dibujo de magnet
        self._d_magnet = None
        # Subplot energy autocorrelation
        self._subp_ac_energy = None
        # Dibujo de la matriz
        self._d_ac_energy = None

        # Flag de vista activa
        self._active_view = False
        # Modo de vista activa
        self._view_mode = 'all'
        # Ultimo spin seleccionado
        self._current_idx = -1
        # Flag de modo test
        self._test_mode = False
        # Máscara para vecinos y valor
        self._mask_v = 3
        self._mask = np.zeros(self._n**2, dtype=bool)
        # Diccionario de eventos conectados
        self._events = {'close_event': None,
                        'button_press_event': None,
                        'motion_notify_event': None}

    def _set(self, T=None, J=None, B=None):
        if T is None:
            T = self._T
        if J is None:
            J = self._J
        if B is None:
            B = self._B
        print (T)
        print (J)
        print (B)
        self._lib.set_params(self, T, J, B)

    @property
    def T(self): return self._T

    @T.setter
    def T(self, value): self._set(T=value)

    @property
    def J(self): return self._J

    @J.setter
    def J(self, value): self._set(J=value)

    @property
    def B(self): return self._B

    @B.setter
    def B(self, value): self._set(B=value)

    @property
    def nflips(self): return self._total_flips

    def run(self, step_size=None):
        if step_size is None:
            step_size = self._step_size
        # Prepara la memoria para alojar los resultados de E y M
        self._energy = np.zeros(step_size, dtype=C.c_float)
        self._p_energy = self._energy.ctypes.data_as(C.POINTER(C.c_float))
        self._magnet = np.zeros(step_size, dtype=C.c_int)
        self._p_magnet = self._magnet.ctypes.data_as(C.POINTER(C.c_int))
        # Llama a la función metropolis
        nflips = self.__metropolis(self, step_size)
        # Actualiza
        self._refresh()
        # Devuelve los pasos aceptados
        return nflips

    def fill_random(self, prob=0.5):
        # Da vuelta con probabilidad prob los elementos
        self._lattice[np.random.rand(self._n**2) > prob] *= -1
        # Actualiza
        self._refresh()

    def pick_site(self):
        return self.__pick_site(self)

    def find_neighbors(self, idx):
        self.__find_neighbors(self, idx)
        return self._W, self._N, self._E, self._S

    def cost(self, idx):
        self.find_neighbors(idx)
        return self.__cost(self, idx)

    def accept_flip(self, idx):
        aligned = self.cost(idx)
        self.__accept_flip(self, idx, aligned)
        self._refresh()

    def calc_energy(self, idx):
        return self.__calc_energy(self, idx)

    def calc_magnet(self, idx):
        return self.__calc_magnet(self, idx)

    def calc_lattice(self):
        self.__calc_lattice(self)

    def view(self, mode='lattice'):
        self._view_mode = mode
        # Si la vista no fue creada
        if not self._active_view:
            # Crea el marco interactivo
            plt.ion()

            # Crea la grilla para los subplots
            gs = gridspec.GridSpec(2,5)
            pos1 = gs[0:2,0:2]
            pos2 = gs[0,3:]
            pos3 = gs[1,3:]

            # Almacena el objeto figura
            self._fig = plt.figure()

            # Configura los subplots
            config1 = {#'xlim':   (0,1),
                       #'ylim':   (0,1),
                       #'xlabel': '$N$',
                       #'ylabel': '$E$',
                       'xscale': 'linear',
                       'yscale': 'linear',
                       'axisbg': 'w',
                       #'title':  'Lattice',
                       'aspect': 'auto'}
            config2 = {#'xlim':   (0,1),
                       #'ylim':   (0,1),
                       'xlabel': '$N$',
                       'ylabel': '$E$',
                       'xscale': 'linear',
                       'yscale': 'linear',
                       'axisbg': 'w',
                       #'title':  'Energy',
                       'aspect': 'auto'}
            config3 = {#'xlim':   (-5,5),
                       #'ylim':   (-2,2),
                       'xlabel': '$N$',
                       'ylabel': '$M$',
                       'xscale': 'linear',
                       'yscale': 'linear',
                       'axisbg': 'w',
                       #'title':  'Magnet',
                       'aspect': 'auto'}

            # Crea los subplot
            if mode == 'all':
                self._fig.subplots_adjust(wspace = .8, hspace=.4)
                self._plot_lattice(self._fig, pos1, **config1)
                self._plot_energy(self._fig, pos2, **config2)
                self._plot_magnet(self._fig, pos3, **config3)
            elif mode == 'lattice':
                self._plot_lattice(self._fig, 111, **config1)

            # Configura una función para detectar el cierre de la ventana
            self._connect_event('close_event')

            # Actualiza
            self._active_view = True
            self._refresh()

        else:
            # Fuerza el foco a la figura
            self._fig.canvas.get_tk_widget().focus_force()
            self._refresh()

    def _plot_lattice(self, fig=None, pos=None, **kargs):
        if fig is None:
            fig = self._fig
        if pos is None:
            pos = 111
        # Almacena el subplot para lattice
        self._subp_lattice = fig.add_subplot(pos, **kargs)
        # Transforma el array en matriz
        aux = self._reshape()
        # Almacena el objeto dibujo
        self._d_lat = self._subp_lattice.matshow(aux,
                                                 cmap='gray',
                                                 aspect='equal')

        return self._subp_lattice, self._d_lat

    def _plot_energy(self, fig=None, pos=None, **kargs):
        if fig is None:
            fig = self._fig
        if pos is None:
            pos = 111
        # Almacena el subplot para energy
        self._subp_energy = fig.add_subplot(pos, **kargs)
        # Almacena el objeto dibujo
        self._d_energy, = self._subp_energy.plot(self._energy)

        return self._subp_energy, self._d_energy

    def _plot_magnet(self, fig=None, pos=None, **kargs):
        if fig is None:
            fig = self._fig
        if pos is None:
            pos = 111
        # Almacena el subplot para energy
        self._subp_magnet = fig.add_subplot(pos, **kargs)
        # Almacena el objeto dibujo
        self._d_magnet, = self._subp_magnet.plot(self._magnet)

        return self._subp_magnet, self._d_magnet

    def _plot_ac_energy(self, fig=None, pos=None, **kargs):
        if fig is None:
            fig = self._fig
        if pos is None:
            pos = 111
        # Almacena el subplot para energy
        self._subp_ac_energy = fig.add_subplot(pos, **kargs)
        # Almacena el objeto dibujo
        self._d_ac_energy, = self._subp_ac_energy.plot(self._energy)

        return self._subp_ac_energy, self._d_ac_energy

    def _reshape(self):
        # Copia el array 1D y lo transforma en matriz
        aux = np.copy(self._lattice).astype(int)
        if self._test_mode:
            # Aplica la máscara
            aux[self._mask] *= self._mask_v
        aux = aux.reshape(self._n, self._n)
        return aux

    def _refresh(self):
        # Verifica que exista una ventana abierta
        if self._active_view:
            # Transforma el array en matriz
            aux = self._reshape()

            # Para el modo test
            if self._test_mode:
                # Setea los valores máximo y minimo
                self._d_lat.set_clim(vmin=-self._mask_v, vmax=self._mask_v)
            else:
                # Setea los valores máximo y minimo
                self._d_lat.set_clim(vmin=-1, vmax=1)

            # Actualiza el dibujo
            if self._view_mode == 'all':
                self._d_lat.set_array(aux)
                self._d_energy.set_xdata(self._energy)
                self._d_magnet.set_xdata(self._magnet)
            elif self._view_mode == 'lattice':
                self._d_lat.set_array(aux)
            plt.draw()

    def _refresh_all(self):
        pass

    def _refresh_lattice(self):
        pass

    def _refresh_energy(self):
        pass

    def _refresh_magnet(self):
        pass

    def plot_energy(self, **kargs):
        params = {'xlim': (0, self._flips)}
        params.update(kargs)
        fig = plt.figure()
        self._plot_energy(fig, 111, **params)
        plt.show()

    def test(self, active=True):
        if not self._active_view:
            self.view()
        self._test_mode = active

        if active:
            self._connect_event('button_press_event')
            self._connect_event('motion_notify_event')
        else:
            self._disconnect_event('button_press_event')
            self._disconnect_event('motion_notify_event')
            self._refresh()

    def _connect_event(self, *args, **kargs):
        # Eventos y funciones correspondientes por defecto
        events = {'close_event': lambda evt: self._close_view(evt),
                  'button_press_event': lambda evt: self._onclick(evt),
                  'motion_notify_event': lambda evt: self._onmotion(evt)}

        # Casos donde recibe solo el nombre del evento
        for element in args:
            event = element
            function = events[event]
            self._fig.canvas.mpl_connect(event, function)
            self._events[event] = self._fig.canvas.mpl_connect(event, function)

        # Casos donde recibe el evento y una nueva función a ejecutar
        for event, function in kargs.items():
            self._events[event] = self._fig.canvas.mpl_connect(event, function)

    def _disconnect_event(self, *args):
        for event in args:
            event_id = self._events[event]
            self._fig.canvas.mpl_disconnect(event_id)

    def _close_view(self, evt):
        # Detecta el cierre de la ventana
        self._active_view = False

    def _onclick(self, evt):
        if evt.button == 1:
            self._current_idx = int(evt.xdata+0.5) + int(evt.ydata+0.5) * self._n
            self.accept_flip(self._current_idx)
        elif evt.button == 3:
            self._current_idx = int(evt.xdata+0.5) + int(evt.ydata+0.5) * self._n
            cost = self.cost(self._current_idx)
            energy = self.calc_energy(self._current_idx)
            magnet = self.calc_magnet(self._current_idx)
            print('cost:   ' + str(cost))
            print('energy: ' + str(energy))
            print('magnet: ' + str(magnet))

    def _onmotion(self, evt):
        self._mask = np.ones(self._n**2, dtype=bool)
        if evt.inaxes != self._d_lat.axes: return
        idx = int(evt.xdata+0.5) + int(evt.ydata+0.5) * self._n
        W, N, E, S = self.find_neighbors(idx)
        self._mask[W] = False
        self._mask[N] = False
        self._mask[E] = False
        self._mask[S] = False
        self._refresh()

    def autocorrelate(x):
        fftx = npfft.fft(x)
        fftx_mean = np.mean(fftx)
        fftx_std = np.std(fftx)

        ffty = np.conjugate(fftx)
        ffty_mean = np.mean(ffty)
        ffty_std = np.std(ffty)

        result = np.fft.ifft((fftx - fftx_mean) * (ffty - ffty_mean))
        result = np.fft.fftshift(result)
        return [i / (fftx_std * ffty_std) for i in result.real]

class LivePlot():
    def __init__(self, name='Untitled'):
        self._figure = plt.figure(name)
        self._figure.clear()
        self._subplots = dict()
        self._curves = dict()
        self._mats = dict()

        self._events = {'close_event': None,
                        'button_press_event': None,
                        'motion_notify_event': None}

    def start(self):
        pass

    def stop(self):
        pass

    def _connect_event(self, **kargs):
        for event, function in kargs.items():
            self._events[event] = self._figure.canvas.mpl_connect(event, function)

    def _disconnect_event(self, *args):
        for event in args:
            event_id = self._events[event]
            self._fig.canvas.mpl_disconnect(event_id)

    def add_subplot(self, name, pos=111, **config):
        subplot = self._figure.add_subplot(pos, **config)
        self._subplots.update(name, subplot)
        return subplot

    def add_curve(self, name, subplot, xdata=None, ydata=None, **config):
        if xdata is None:
            curve, = self._subplots[subplot].plot(ydata, **config)
        else:
            curve, = self._subplots[subplot].plot(xdata, ydata, **config)

        self._curves.update(name, curve)
        return curve

    def add_matrix(self, name, subplot, matrix, **config):
        mat = self._subplots[subplot].plot(xdata, ydata, **config)
        self._mats.update(name, mat)
        return mat

    def save(self, file_name=None, path='./Plots/', file_type='png'):
        self._figure.savefig(path + file_name + file_type)

class Curve():
    def __init__(self, name, subplot, **config):
        self._data = None
        self._curve, = self._subplots[subplot].plot([], [] , **config)

    @property
    def data(self): return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self._curve.set_data(data[0], data[1])

    @property
    def label(self): return self._label

class Lattice():
    def __init__(self):
        pass

    def fill_random(self):
        pass

    def matrix_form(self):
        pass

    def calc_energy(self):
        pass

    def calc_magnet(self):
        pass

    def calc_all(self):
        pass
