import numpy as np
import ctypes as C
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from liveplot import LivePlot, Curve
import time

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

    def __init__(self, n, data='../datos/'):

        # Biblioteca de funciones
        self._lib = C.CDLL('./libising.so')

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

        # Origen de datos
        self._data = data

        # Otras variables internas
        self._step_size = None
        self._max_ntry = 1e6

        # Curvas para liveplot
        self.lattice = Curve('Lattice')
        self.energy = Curve('Energy')
        self.magnet = Curve('Magnetization')

        # Memoria asignada a la red
        self._lattice = np.ones(n**2, dtype=C.c_int)
        self._p_lattice = self._lattice.ctypes.data_as(C.POINTER(C.c_int))
        self.step_size = int(1.5 * n**2)

        # Inicializa los valores
        self.__init(self, n)
        self.calc_lattice()

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
        self.__set_params(self, T, J, B)

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

    @property
    def step_size(self): return self._step_size

    @step_size.setter
    def step_size(self, value):
        if self._step_size != value:
            self._step_size = value
            self._energy = np.zeros(self._step_size, dtype=C.c_float)
            self._p_energy = self._energy.ctypes.data_as(C.POINTER(C.c_float))
            self._magnet = np.zeros(self._step_size, dtype=C.c_int)
            self._p_magnet = self._magnet.ctypes.data_as(C.POINTER(C.c_int))

    @property
    def max_ntry(self): return self._max_ntry

    @max_ntry.setter
    def max_ntry(self, value): self._max_ntry = value

    def _update(self):
        self.lattice.data = np.copy(self._lattice).reshape([self._n, self._n])
        self.energy.data = self._energy[0:self._flips]
        self.magnet.data = self._magnet[0:self._flips]

    def fill_random(self, prob=0.5):
        self._lattice[np.random.rand(self._n**2) > prob] *= -1
        self.lattice.data = np.copy(self._lattice).reshape([self._n, self._n])

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

    def calc_energy(self, idx):
        return self.__calc_energy(self, idx)

    def calc_magnet(self, idx):
        return self.__calc_magnet(self, idx)

    def calc_lattice(self):
        self.__calc_lattice(self)

    def run(self):
        nflips = self.__metropolis(self, self._step_size)
        self._update()
        return nflips
