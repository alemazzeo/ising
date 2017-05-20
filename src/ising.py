import numpy as np
import ctypes as C
import matplotlib.pyplot as plt
import time
import threading
import os

class lattice():

    def __init__(self, n, data='../datos/', lib='./libising.so'):
        # Tamaño de la red
        self._n = n
        # Origen de datos
        self._data = data

        # Biblioteca de funciones
        self._lib = C.CDLL(lib)

        # Memoria asignada a la red
        self._lattice = np.ones(self._n**2, dtype=C.c_int)

        # Memoria asignada a los valores actuales
        self._iter = 0
        self._current_T = 2.0
        self._current_E = C.c_int(0)
        self._current_M = C.c_int(0)
        self._current_C = C.c_float(0.0)

        # Lista de valores obtenidos (para graficar)
        self._list_iter = list()
        self._list_T = list()
        self._list_E = list()
        self._list_M = list()
        self._list_C = list()

        # Figura principal
        self._fig = None
        # Subplot lattice
        self._subp_lattice = None
        # Dibujo de la matriz
        self._draw = None
        # Flag de vista activa
        self._active_view = False
        # Ultimo spin seleccionado
        self._current_idx = -1
        # Flag de modo test
        self._test_mode = False
        # Máscara para vecinos y valor
        self._mask_v = 5
        self._mask = np.zeros(self._n**2, dtype=bool)
        # Diccionario de eventos conectados
        self._events = dict()

    def run(self, niter=1000):

        # Calcula las exponenciales para la T configurada
        T = self._current_T
        T_table = np.array([T, np.exp(-4/T), np.exp(-8/T)], dtype=C.c_float)

        # Prepara los argumentos que recibe la función
        args = (self._lattice.ctypes.data_as(C.POINTER(C.c_int)),
                C.c_int(self._n),
                T_table.ctypes.data_as(C.POINTER(C.c_float)),
                C.c_int(niter),
                C.byref(self._current_E),
                C.byref(self._current_M))

        # Llama a la función metropolis
        # int metropolis(int *lattice, int n, float *T, int pasos,
        #                int *energy, int *magnet);
        nflips = self._lib.metropolis(*args)

        # Actualiza
        self._refresh()
        # Devuelve los pasos aceptados
        return nflips

    def fill_random(self, prob=0.5):
        # Crea una matriz auxiliar de unos
        aux = np.ones(self._n**2, dtype=C.c_int)
        # Da vuelta con probabilidad prob los elementos
        self._lattice[np.random.rand(self._n**2) > prob] *= -1
        # Actualiza
        self._refresh()

    def pick_site(self):
        args = (self._lattice.ctypes.data_as(C.POINTER(C.c_int)),
                C.c_int(self._n))
        site = self._lib.pick_site(*args)
        return site

    def find_neighbors(self, idx, ctype=False):
        W = C.c_int(0)
        N = C.c_int(0)
        E = C.c_int(0)
        S = C.c_int(0)
        args = (self._lattice.ctypes.data_as(C.POINTER(C.c_int)),
                C.c_int(self._n),
                C.c_int(idx),
                C.byref(W),
                C.byref(N),
                C.byref(E),
                C.byref(S))
        self._lib.find_neighbors(*args)
        if ctype:
            return W, N, E, S
        else:
            return W.value, N.value, E.value, S.value

    def cost(self, idx):
        W, N, E, S = self.find_neighbors(idx, ctype=True)
        args = (self._lattice.ctypes.data_as(C.POINTER(C.c_int)),
                C.c_int(self._n),
                C.c_int(idx),
                C.byref(W),
                C.byref(N),
                C.byref(E),
                C.byref(S))
        return self._lib.cost(*args)

    def accept_flip(self, idx):
        opposites = self.cost(idx)
        args = (self._lattice.ctypes.data_as(C.POINTER(C.c_int)),
                C.c_int(self._n),
                C.c_int(idx),
                C.c_int(opposites),
                C.byref(self._current_E),
                C.byref(self._current_M))
        self._lib.accept_flip(*args)
        self._refresh()

    def calc_energy(self, idx):
        args = (self._lattice.ctypes.data_as(C.POINTER(C.c_int)),
                C.c_int(self._n),
                C.c_int(idx))
        return self._lib.calc_energy(*args)

    def calc_magnet(self, idx):
        args = (self._lattice.ctypes.data_as(C.POINTER(C.c_int)),
                C.c_int(self._n),
                C.c_int(idx))
        return self._lib.calc_magnet(*args)

    def view(self, lattice=True, energy=True, magnet=True, temp=True):
        # Si la vista no fue creada
        if not self._active_view:
            # Crea el marco interactivo
            plt.ion()
            # Almacena el objeto figura
            self._fig = plt.figure()
            # Almacena el subplot para lattice
            self._subp_lattice = self._fig.add_subplot(111)
            # Transforma el array en matriz
            aux = self._reshape()
            # Almacena el objeto dibujo
            self._draw = self._subp_lattice.matshow(aux, cmap='gray')
            # Configura una función para detectar el cierre de la ventana
            self._connect_event('close_event')
            # Actualiza
            self._active_view = True
            self._refresh()

        else:
            # Fuerza el foco a la figura
            self._fig.canvas.get_tk_widget().focus_force()
            self._refresh()

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
                self._draw.set_clim(vmin=-self._mask_v, vmax=self._mask_v)
            else:
                # Setea los valores máximo y minimo
                self._draw.set_clim(vmin=-1, vmax=1)

            # Actualiza el dibujo
            self._draw.set_array(aux)
            plt.draw()

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
        events = {'close_event': self._close_view,
                  'button_press_event': self._onclick,
                  'motion_notify_event': self._onmotion}

        # Casos donde recibe solo el nombre del evento
        for element in args:
            event = element
            function = events[event]
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
        if evt.inaxes != self._draw.axes: return
        idx = int(evt.xdata+0.5) + int(evt.ydata+0.5) * self._n
        W, N, E, S = self.find_neighbors(idx, ctype=False)
        self._mask[W] = False
        self._mask[N] = False
        self._mask[E] = False
        self._mask[S] = False
        self._refresh()
