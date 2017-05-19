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

        self._fig = None
        self._draw = None
        self._active_view = False
        self._current_idx = -1
        self._test_mode = False

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
            # Transforma el array en matriz
            aux = self._reshape()
            # Almacena el objeto dibujo
            self._draw = self._fig.add_subplot(111).matshow(aux, 
                                                            cmap='gray',
                                                            picker=1)
            # Setea los valores máximo y minimo (evita que se haga solo)
            self._draw.set_clim(vmin=-1, vmax=1)
            # Configura una función para detectar el cierre de la ventana
            self._ventana_interactiva()
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
        aux = aux.reshape(self._n, self._n)
        return aux

    def _ventana_interactiva(self):
        # Configura una función para detectar el cierre de la ventana
        self._fig.canvas.mpl_connect('close_event', 
                                     self._close_view)
        # Configura una función para detectar los clicks
        self._fig.canvas.mpl_connect('pick_event', 
                                     self._onpick)

    def _close_view(self, evt):
        # Detecta el cierre de la ventana
        self._active_view = False

    def _onpick(self, evt):
        self._current_idx = evt.ind

    def _refresh(self):
        # Verifica que exista una ventana abierta
        if self._active_view:
            # Transforma el array en matriz
            aux = self._reshape()
            # Actualiza el dibujo
            self._draw.set_array(aux)
            plt.draw()

    def test(self):
        if not self._active_view:
            self.view()
        self._test_mode = True
        
            
