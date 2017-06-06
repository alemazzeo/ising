import matplotlib
matplotlib.use('Qt5Agg')

import ctypes as C
import numpy as np
import matplotlib.pyplot as plt
import os

class Sample(C.Structure):
    _fields_ = [("_sample_size", C.c_int),
                ("_step_size", C.c_int),
                ("_tolerance", C.c_float),
                ("_T", C.c_float),
                ("_J", C.c_float),
                ("_B", C.c_float),
                ("_p_energy", C.POINTER(C.c_float)),
                ("_p_magnet", C.POINTER(C.c_int)),
                ("_p_flips", C.POINTER(C.c_int)),
                ("_p_total_flips", C.POINTER(C.c_int)),
                ("_p_q", C.POINTER(C.c_float))]

    def __init__(self, sample_size, step_size=None, tolerance=None,
                 path='../data/samples/', name=None):

        self._sample_size = sample_size
        self._step_size = step_size
        self._tolerance = tolerance
        self._path = path
        self._name = name

        # Memoria asignada
        self._energy = np.zeros(self._sample_size, dtype=C.c_float)
        self._magnet = np.zeros(self._sample_size, dtype=C.c_int)
        self._flips = np.zeros(self._sample_size, dtype=C.c_int)
        self._total_flips = np.zeros(self._sample_size, dtype=C.c_int)
        self._q = np.zeros(self._sample_size, dtype=C.c_float)

        # Punteros
        self._p_energy = self._energy.ctypes.data_as(C.POINTER(C.c_float))
        self._p_magnet = self._magnet.ctypes.data_as(C.POINTER(C.c_int))
        self._p_flips = self._flips.ctypes.data_as(C.POINTER(C.c_int))
        self._p_total_flips = self._total_flips.ctypes.data_as(C.POINTER(C.c_int))
        self._p_q = self._q.ctypes.data_as(C.POINTER(C.c_float))

    def save(self, name=None, path=None):
        extension = '.sample'

        if path is None:
            path = self._path

        os.makedirs(path, exist_ok=True)

        if name is None:
            name = self._name

        if os.path.isfile(path+name+'0'+extension):
            i = 0
            newname = name + str(i)
            while os.path.isfile(path+newname+extension):
                i += 1
                newname = name + str(i)
            name = newname
        else:
            name = name + '0'

        fullname = path + name + extension

        params = [self._sample_size,
                  self._step_size,
                  self._tolerance,
                  self._T,
                  self._J,
                  self._B]

        data = [self._energy,
                self._magnet,
                self._flips,
                self._total_flips,
                self._q]

        np.save(fullname, [params,data])

        return "'" + fullname + "'" + ' has been successfully saved'

    @classmethod
    def load(cls, name, path='../data/samples'):
        extension = '.sample'
        if name[-len(extension):] != extension:
            name = name + extension

        params, data = np.load(path+name)

        load_sample = cls(sample_size = int(params[0]),
                         step_size = int(params[1]),
                         tolerance = float(params[2]),
                         name = name,
                         path = path)

        load_sample._T = float(params[3])
        load_sample._J = float(params[4])
        load_sample._B = float(params[5])

        load_sample._energy = data[0]
        load_sample._magnet = data[1]
        load_sample._flips = data[2]
        load_sample._total_flips = data[3]
        load_sample._q = data[4]

        return load_sample

    @property
    def energy(self): return self._energy

    @property
    def magnet(self): return self._magnet

    @property
    def flips(self): return self._flips

    @property
    def total_flips(self): return self._total_flips

    @property
    def q(self): return self._q

class Ising(C.Structure):
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
                ("_current_energy", C.c_float),
                ("_current_magnet", C.c_int),
                ("_p_energy", C.POINTER(C.c_float)),
                ("_p_magnet", C.POINTER(C.c_int))]

    def __init__(self, n, name='ising', path='../data/states/'):

        # Nombre y ruta
        self._name = name
        self._path = path

        # Biblioteca de funciones
        self._lib = C.CDLL('./libising.so')

        self.__init = self._lib.init
        self.__set_params = self._lib.set_params
        self.__info = self._lib.info
        self.__metropolis = self._lib.metropolis
        self.__run = self._lib.run
        self.__run_until = self._lib.run_until
        self.__run_sample = self._lib.run_sample
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
        self.__autocorrelation = self._lib.autocorrelation

        self.__init.restype = C.c_int
        self.__set_params.restype = C.c_int
        self.__info.restype = C.c_int
        self.__metropolis.restype = C.c_int
        self.__run.restype = C.c_int
        self.__run_until.restype = C.c_float
        self.__run_sample.restype = C.c_int
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
        self.__autocorrelation.restype = C.c_int

        self.__init.argtypes = [C.POINTER(Ising), C.c_int]
        self.__set_params.argtypes = [C.POINTER(Ising), C.c_float, C.c_float, C.c_float]
        self.__info.argtypes = [C.POINTER(Ising)]
        self.__metropolis.argtypes = [C.POINTER(Ising), C.c_int]
        self.__run.argtypes = [C.POINTER(Ising), C.c_int]
        self.__run_until.argtypes = [C.POINTER(Ising), C.c_int, C.c_float]
        self.__run_sample.argtypes = [C.POINTER(Ising), C.POINTER(Sample)]
        self.__pick_site.argtypes = [C.POINTER(Ising)]
        self.__flip.argtypes = [C.POINTER(Ising), C.c_int]
        self.__find_neighbors.argtypes = [C.POINTER(Ising), C.c_int]
        self.__cost.argtypes = [C.POINTER(Ising), C.c_int]
        self.__try_flip.argtypes = [C.POINTER(Ising), C.c_float]
        self.__accept_flip.argtypes = [C.POINTER(Ising), C.c_int, C.c_int]
        self.__calc_pi.argtypes = [C.POINTER(Ising), C.c_int, C.c_int]
        self.__calc_energy.argtypes = [C.POINTER(Ising), C.c_int]
        self.__calc_magnet.argtypes = [C.POINTER(Ising), C.c_int]
        self.__calc_lattice.argtypes = [C.POINTER(Ising)]
        self.__autocorrelation.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_int]

        # Otras variables internas
        self._step_size = None

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

    @property
    def current_magnet(self):
        return self._current_magnet

    @property
    def current_energy(self):
        return self._current_energy

    @step_size.setter
    def step_size(self, value):
        if self._step_size != value:
            self._step_size = value
            self._energy = np.zeros(self._step_size, dtype=C.c_float)
            self._p_energy = self._energy.ctypes.data_as(C.POINTER(C.c_float))
            self._magnet = np.zeros(self._step_size, dtype=C.c_int)
            self._p_magnet = self._magnet.ctypes.data_as(C.POINTER(C.c_int))

    @property
    def energy(self):
        return np.trim_zeros(self._energy)

    @property
    def magnet(self):
        return np.trim_zeros(self._magnet)

    def _update(self):
        pass

    def save(self, name=None, path=None):
        extension = '.state'

        if path is None:
            path = self._path

        os.makedirs(path, exist_ok=True)

        if name is None:
            name = self._name

        if os.path.isfile(path+name+'0'+extension):
            i = 0
            newname = name + str(i)
            while os.path.isfile(path+newname+extension):
                i += 1
                newname = name + str(i)
            name = newname
        else:
            name = name + '0'

        fullname = path + name + extension

        params = [self._n,
                  self._total_flips,
                  self._T,
                  self._J,
                  self._B]

        data = self._lattice

        np.save(fullname, [params,data])

        return "'" + fullname + "'" + ' has been successfully saved'

    @classmethod
    def load(cls, name, path='../data/states/'):
        extension = '.state'
        if name[-len(extension):] != extension:
            name = name + extension

        params, data = np.load(path+name)

        load_ising = cls(n = int(params[0]))
        load_ising._total_flips = int(params[1])

        load_ising.T = float(params[2])
        load_ising.J = float(params[3])
        load_ising.B = float(params[4])

        load_ising._lattice = data
        load_ising._p_lattice = load_ising._lattice.ctypes.data_as(C.POINTER(C.c_int))
        load_ising.calc_lattice()

        return load_ising

    def fill_random(self, prob=0.5):
        self._lattice[np.random.rand(self._n**2) > prob] *= -1
        self.calc_lattice()

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

    def run(self, step_size=None):
        if step_size is not None:
            self.step_size = step_size
        nflips = self.__run(self, self.step_size)
        self._update()
        return nflips

    def run_until(self, step_size=None, tolerance=10.0):
        if step_size is not None:
            self.step_size = step_size
        q = self.__run_until(self, self.step_size, tolerance)
        self._update()
        return q

    def run_sample(self, sample_size, step_size=None, tolerance=10.0):
        if step_size is not None:
            self.step_size = step_size
        data = Sample(sample_size, self.step_size, tolerance)
        self.__run_sample(self, data)
        return data

    @classmethod
    def plot_step(cls, data, ax=None, **kwargs):
        params = {'label': 'Last step',
                  'lw': 2,
                  'ls': '-'}
        params.update(kwargs)
        data = np.trim_zeros(data)
        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1)

        curve, = ax.plot(data, **params)
        return curve

    @classmethod
    def plot_lattice(cls, data1D, ax=None, **kwargs):
        params = {'vmin': -1,
                  'vmax': 1,
                  'cmap': 'gray'}
        params.update(kwargs)

        n = int(len(data1D)**0.5)
        data = np.reshape(data1D,(n,n))

        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1)

        curve = ax.matshow(data, **params)
        return curve

    @classmethod
    def plot_correlation(cls, data, ax=None):
        n = len(data)
        assert ax=None or len(ax) == 2
        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(2)

        x = data[0:int(n/100)]
        acorr = cls.autocorrelation(x)
        curve_d = cls.plot_step(x, ax=ax[0], label='Data')
        curve_a = cls.plot_step(acorr, ax=ax[1], label='Autocorrelation')

        for i in range(99):
            x = data[0:int(n*(i+2)/100)]
            acorr = cls.autocorrelation(x)
            curve_d.set_data(np.arange(x.size), x)
            curve_a.set_data(np.arange(acorr.size), acorr)
            ax[0].relim()
            ax[1].relim()
            ax[0].autoscale()
            ax[1].autoscale()
            plt.draw()
            plt.pause(0.00001)

    @classmethod
    def autocorrelation (cls, x):
        xp = x-np.mean(x)
        f = np.fft.fft(xp)
        p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
        pi = np.fft.ifft(p)
        return np.real(pi)[:int(x.size/2)]/np.sum(xp**2)

    @classmethod
    def autocorrelation2(cls, x):
        n = len(x)
        var = np.var(x)
        acorr = np.zeros(n, dtype=float)
        for k in range(n):
            acorr[k] = np.sum((x[0:n-k]-np.mean(x[0:n-k]))
                              *(x[k:n]-np.mean(x[k:n]))) / (n*var)
        return acorr
