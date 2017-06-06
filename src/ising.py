import matplotlib
matplotlib.use('Qt5Agg')

import ctypes as C
import numpy as np
import matplotlib.pyplot as plt
import os

class Utilities():
    @classmethod
    def plot_array1D(cls, data, ax=None, **kwargs):
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
    def plot_hist(cls, data, ax=None, **kwargs):
        params = {}
        params.update(kwargs)
        data = np.trim_zeros(data)
        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1)

        Y, X, _ = ax.plot(data, **params)
        return [Y, X], ax, fig

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
        return curve, ax, fig

    @classmethod
    def plot_correlation(cls, data, ax=None,
                         plot1_kw=dict(label='Data'),
                         plot2_kw=dict(label='Autocorrelation')):
        n = len(data)
        assert ax==None or len(ax) == 2
        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(2)

        x = data[0:int(n/100)]
        acorr = cls.autocorrelation(x)
        curve_d = cls.plot_step(x, ax=ax[0], **plot1_kw)
        curve_a = cls.plot_step(acorr, ax=ax[1], **plot2_kw)

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

        return curve_d, curve_a, ax, fig

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

    def __init__(self, n):

        self._n = n
        self._step_size = None

        # Biblioteca de funciones
        self._lib = C.CDLL('./libising.so')

        self.C_init = self._lib.init
        self.C_set_params = self._lib.set_params
        self.C_info = self._lib.info
        self.C_metropolis = self._lib.metropolis
        self.C_run = self._lib.run
        self.C_run_until = self._lib.run_until
        self.C_run_sample = self._lib.run_sample
        self.C_pick_site = self._lib.pick_site
        self.C_flip = self._lib.flip
        self.C_find_neighbors = self._lib.find_neighbors
        self.C_cost = self._lib.cost
        self.C_try_flip = self._lib.try_flip
        self.C_accept_flip = self._lib.accept_flip
        self.C_calc_pi = self._lib.calc_pi
        self.C_calc_energy = self._lib.calc_energy
        self.C_calc_magnet = self._lib.calc_magnet
        self.C_calc_lattice = self._lib.calc_lattice

        self.C_init.restype = C.c_int
        self.C_set_params.restype = C.c_int
        self.C_info.restype = C.c_int
        self.C_metropolis.restype = C.c_int
        self.C_run.restype = C.c_int
        self.C_run_until.restype = C.c_float
        self.C_run_sample.restype = C.c_int
        self.C_pick_site.restype = C.c_int
        self.C_flip.restype = C.c_int
        self.C_find_neighbors.restype = C.c_int
        self.C_cost.restype = C.c_int
        self.C_try_flip.restype = C.c_int
        self.C_accept_flip.restype = C.c_int
        self.C_calc_pi.restype = C.c_float
        self.C_calc_energy.restype = C.c_float
        self.C_calc_magnet.restype = C.c_int
        self.C_calc_lattice.restype = C.c_int

        self.C_init.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_set_params.argtypes = [C.POINTER(Ising), C.c_float, C.c_float, C.c_float]
        self.C_info.argtypes = [C.POINTER(Ising)]
        self.C_metropolis.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_run.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_run_until.argtypes = [C.POINTER(Ising), C.c_int, C.c_float]
        self.C_run_sample.argtypes = [C.POINTER(Ising), C.POINTER(Sample)]
        self.C_pick_site.argtypes = [C.POINTER(Ising)]
        self.C_flip.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_find_neighbors.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_cost.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_try_flip.argtypes = [C.POINTER(Ising), C.c_float]
        self.C_accept_flip.argtypes = [C.POINTER(Ising), C.c_int, C.c_int]
        self.C_calc_pi.argtypes = [C.POINTER(Ising), C.c_int, C.c_int]
        self.C_calc_energy.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_calc_magnet.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_calc_lattice.argtypes = [C.POINTER(Ising)]

        # Asigna la memoria para energy, magnet y lattice
        self.step_size = int(1.5 * n**2)
        self.fill_random()

        # Inicializa los valores
        self.C_init(self, n)
        self.C_calc_lattice(self)


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

    def _set(self, T=None, J=None, B=None):
        if T is None:
            T = self._T
        if J is None:
            J = self._J
        if B is None:
            B = self._B
        self.C_set_params(self, T, J, B)

    def assign_lattice(self, data):
        n = len(data)
        assert n == self._n**2
        self._lattice = np.array(data, dtype=C.c_int)
        self._p_lattice = self._lattice.ctypes.data_as(C.POINTER(C.c_int))

    def fill_random(self, prob=0.5):
        random = np.ones(self._n**2)
        random[np.random.rand(self._n**2) > prob] *= -1
        self.assign_lattice(random)
        self.C_calc_lattice(self)

    def run(self, step_size=None):
        if step_size is not None:
            self.step_size = step_size
        nflips = self.C_run(self, self.step_size)
        return nflips

    def run_until(self, step_size=None, tolerance=10.0):
        if step_size is not None:
            self.step_size = step_size
        q = self.C_run_until(self, self.step_size, tolerance)
        return q

    def run_sample(self, sample_size, step_size=None, tolerance=10.0):
        if step_size is not None:
            self.step_size = step_size
        data = Sample(sample_size, self.step_size, tolerance)
        self.C_run_sample(self, data)
        return data

class State(Ising):
    def __init__(self, n, name='ising', path='../data/states/'):

        super().__init__(n)

        # Nombre y ruta
        self._name = name
        self._path = path

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

        load_state = cls(n = int(params[0]))
        load_state._total_flips = int(params[1])

        load_state.T = float(params[2])
        load_state.J = float(params[3])
        load_state.B = float(params[4])

        load_state._lattice = data
        load_state._p_lattice = load_ising._lattice.ctypes.data_as(C.POINTER(C.c_int))
        load_state.calc_lattice()

        return load_state

class Simulation(State):
    def __init__(self, n, name='sim', path='../data/simulations'):

        State.__init__(self, n)
        self.fill_random()

        self.J = J
        self.B = B

        self._name = name
        self._path = path

        self._params = list()
        self._sample_names = list()
        self._state_names = list()

    def sweep(self, parameter, end, sweep_step=None,
                    sample_size=10000, ising_step='auto', therm='auto'):

        if therm == 'auto':
            therm = self._n2 * 50

        if ising_step == 'auto':
            ising_step = self._n2 * 2

        if parameter in ('T', 'Temperature'):
            start = self.T
            set_value = lambda value: self._set(T=value)
        elif parameter in ('J', 'Interaction'):
            start = self.J
            set_value = lambda value: self._set(J=value)
        elif parameter in ('B', 'Extern field'):
            start = self.B
            set_value = lambda value: self._set(B=value)
        else:
            print('Available parameters to sweep:')
            print('"Temperature"  or "T"')
            print('"Interaction"  or "J"')
            print('"Extern field" or "B"')
            return 0

        values = np.arange(start, end, sweep_step)
        n = float(len(values))

        for i, value in np.ndenumerate(values):
            text_bar = '*' * int(i/n) + '-' * int(1-i/n)
            print(parameter + '%.4f'%(value) + text_bar + '  ', end='\r')

            # Set T and run until thermalization
            set_value(value)
            self.run_until(therm)

            # Fill a sample
            sample = self.run_sample(sample_size, ising_step)
            # Store sample in a list
            self._samples.append(sample)
            self._params = [self.T, self.J, self.B]
        print('Sweep completed.')

    def save(self, name=None, path=None):

        extension = '.simulation'

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

        data = [self._params,
                self._sample_names,
                self._state_names]

        np.save(fullname, data)

        return "'" + fullname + "'" + ' has been successfully saved'

    @classmethod
    def load(cls, name, path='../data/simulations/'):
        extension = '.simulation'
        if name[-len(extension):] != extension:
            name = name + extension

        params, sample_names, state_names  = np.load(path+name)


        return load_sample
