import matplotlib
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    try:
        matplotlib.use('qt4Agg')
    except ImportError:
        raise

import ctypes as C
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

class Utilities():

    @classmethod
    def splitname(cls, fullname):
        fullname, extension = os.path.splitext(fullname)
        path, name = os.path.split(fullname)
        return path, name, extension

    @classmethod
    def newname(cls, fullname, default='../data/temp.npy'):
        path, name, extension = cls.splitname(fullname)
        dpath, dname, dext = cls.splitname(default)

        if extension == '':
            extension = dext

        if path == '':
            path = dpath

        os.makedirs(path, exist_ok=True)

        if name == '':
            name = dname

        if os.path.isfile(path+'/'+name+'0'+extension):
            i = 0
            newname = name + str(i)
            while os.path.isfile(path+'/'+newname+extension):
                i += 1
                newname = name + str(i)
            name = newname
        else:
            name = name + '0'

        return path + '/' + name + extension

    @classmethod
    def lastname(cls, fullname, default='../data/temp.npy'):
        path, name, extension = cls.splitname(cls.newname(fullname, default))
        return path + '/' + name[:-1] + str(int(name[-1])-1) + extension

    @classmethod
    def move(cls, files, dest, copy=False, verbose=False):
        changes = list()
        newlist = list()
        for fullname in files:
            path, name, extension = cls.splitname(fullname)
            dpath, dname, dextension = cls.splitname(dest)

            if (path==dpath and name[:len(dname)]==dname):
                newlist.append(fullname)
            else:
                newname = cls.newname(dest)
                if copy:
                    shutil.copyfile(fullname, newname)
                    if verbose:
                        print(fullname + ' copy to ' + newname)
                else:
                    os.rename(fullname, newname)
                    if verbose:
                        print(fullname + ' move to ' + newname)

                newlist.append(newname)
                changes.append([fullname, newname])

        return newlist, changes

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

        Y, X, _ = ax.hist(data, **params)
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
                ("_tolerance", C.c_double),
                ("_T", C.c_double),
                ("_J", C.c_double),
                ("_B", C.c_double),
                ("_p_energy", C.POINTER(C.c_double)),
                ("_p_magnet", C.POINTER(C.c_int)),
                ("_p_flips", C.POINTER(C.c_int)),
                ("_p_total_flips", C.POINTER(C.c_int)),
                ("_p_q", C.POINTER(C.c_double))]

    def __init__(self, sample_size, step_size=None, tolerance=None):

        self._sample_size = sample_size
        self._step_size = step_size
        self._tolerance = tolerance
        self._fullname = None

        # Memoria asignada
        self._energy = np.zeros(self._sample_size, dtype=C.c_double)
        self._magnet = np.zeros(self._sample_size, dtype=C.c_int)
        self._flips = np.zeros(self._sample_size, dtype=C.c_int)
        self._total_flips = np.zeros(self._sample_size, dtype=C.c_int)
        self._q = np.zeros(self._sample_size, dtype=C.c_double)

        # Punteros
        self._p_energy = self._energy.ctypes.data_as(C.POINTER(C.c_double))
        self._p_magnet = self._magnet.ctypes.data_as(C.POINTER(C.c_int))
        self._p_flips = self._flips.ctypes.data_as(C.POINTER(C.c_int))
        self._p_total_flips = self._total_flips.ctypes.data_as(C.POINTER(C.c_int))
        self._p_q = self._q.ctypes.data_as(C.POINTER(C.c_double))

    def save_as(self, fullname='', default='../data/samples/sample.npy'):
        fullname = Utilities.newname(fullname, default)
        self._fullname = fullname
        if self.save():
            return fullname
        else:
            return ''

    def save(self):
        if self._fullname is None:
            print('First call save_as')
            return False

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

        np.save(self._fullname, [params,data])
        return self._fullname

    @classmethod
    def load(cls, fullname, default='../data/samples/sample.npy'):
        path, name, extension = Utilities.splitname(fullname)
        dpath, dname, dextension = Utilities.splitname(default)

        if path == '':
            path = dpath
        if name == '':
            name = dname
        if extension == '':
            extension = dextension

        fullname = path + '/' + name + extension

        params, data = np.load(fullname)

        load_sample = cls(sample_size = int(params[0]),
                         step_size = int(params[1]),
                         tolerance = float(params[2]))

        load_sample._T = float(params[3])
        load_sample._J = float(params[4])
        load_sample._B = float(params[5])

        load_sample._energy = data[0]
        load_sample._magnet = data[1]
        load_sample._flips = data[2]
        load_sample._total_flips = data[3]
        load_sample._q = data[4]
        load_sample._fullname = fullname

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

    def __repr__(self): return self._fullname
    def __str__(self): return self._fullname + '\n' + self._print_params()

    def _print_params(self):
        return 'T=%.4f\nJ=%.4f\nB=%.4f'% (self._T, self._J, self._B)

    def view_energy(self, ax=None, **kwargs):
        Utilities.plot_hist(self.energy, ax=ax, **kwargs)

    def view_magnet(self, ax=None, **kwargs):
        Utilities.plot_hist(self.magnet, ax=ax, **kwargs)

    def view_flips(self, ax=None, **kwargs):
        Utilities.plot_array1D(self.flips, ax=ax, **kwargs)

    def view_total_flips(self, ax=None, **kwargs):
        Utilities.plot_array1D(self.total_flips, ax=ax, **kwargs)

    def view_q(self, ax=None, **kwargs):
        Utilities.plot_array1D(self.q, ax=ax, **kwargs)

class Ising(C.Structure):
    _fields_ = [("_p_lattice", C.POINTER(C.c_int)),
                ("_n", C.c_int),
                ("_n2", C.c_int),
                ("_flips", C.c_int),
                ("_total_flips", C.c_int),
                ("_T", C.c_double),
                ("_J", C.c_double),
                ("_B", C.c_double),
                ("_p_dEs", C.c_double * 10),
                ("_p_exps", C.c_double * 10),
                ("_W", C.c_int),
                ("_N", C.c_int),
                ("_E", C.c_int),
                ("_S", C.c_int),
                ("_aligned", C.c_int),
                ("_current_energy", C.c_double),
                ("_current_magnet", C.c_int),
                ("_p_energy", C.POINTER(C.c_double)),
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
        self.C_run_until.restype = C.c_double
        self.C_run_sample.restype = C.c_int
        self.C_pick_site.restype = C.c_int
        self.C_flip.restype = C.c_int
        self.C_find_neighbors.restype = C.c_int
        self.C_cost.restype = C.c_int
        self.C_try_flip.restype = C.c_int
        self.C_accept_flip.restype = C.c_int
        self.C_calc_pi.restype = C.c_double
        self.C_calc_energy.restype = C.c_double
        self.C_calc_magnet.restype = C.c_int
        self.C_calc_lattice.restype = C.c_int

        self.C_init.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_set_params.argtypes = [C.POINTER(Ising), C.c_double, C.c_double, C.c_double]
        self.C_info.argtypes = [C.POINTER(Ising)]
        self.C_metropolis.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_run.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_run_until.argtypes = [C.POINTER(Ising), C.c_int, C.c_double]
        self.C_run_sample.argtypes = [C.POINTER(Ising), C.POINTER(Sample)]
        self.C_pick_site.argtypes = [C.POINTER(Ising)]
        self.C_flip.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_find_neighbors.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_cost.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_try_flip.argtypes = [C.POINTER(Ising), C.c_double]
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
            self._energy = np.zeros(self._step_size, dtype=C.c_double)
            self._p_energy = self._energy.ctypes.data_as(C.POINTER(C.c_double))
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

    def calc_lattice(self):
        self.C_calc_lattice(self)

    def fill_random(self, prob=0.5):
        random = np.ones(self._n**2)
        random[np.random.rand(self._n**2) > prob] *= -1
        self.assign_lattice(random)
        self.calc_lattice()

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
    def __init__(self, n):

        super().__init__(n)
        self._fullname = None

    def save_as(self, fullname='', default='../data/states/state.npy'):
        fullname = Utilities.newname(fullname, default)
        self._fullname = fullname
        if self.save():
            return fullname
        else:
            return ''

    def save(self):
        if self._fullname is None:
            print('First call save_as')
            return False

        params = [self._n,
                  self._total_flips,
                  self._T,
                  self._J,
                  self._B]

        data = self._lattice

        np.save(self._fullname, [params,data])
        return self._fullname

    @classmethod
    def load(cls, fullname, default='../data/states/state.npy'):
        path, name, extension = Utilities.splitname(fullname)
        dpath, dname, dextension = Utilities.splitname(default)

        if path == '':
            path = dpath
        if name == '':
            name = dname
        if extension == '':
            extension = dextension

        fullname = path + '/' + name + extension

        params, data = np.load(fullname)

        load_state = cls(n = int(params[0]))
        load_state._total_flips = int(params[1])

        load_state.T = float(params[2])
        load_state.J = float(params[3])
        load_state.B = float(params[4])

        load_state._lattice = data
        load_state._p_lattice = load_state._lattice.ctypes.data_as(C.POINTER(C.c_int))
        load_state.calc_lattice()
        load_state._fullname = fullname

        return load_state

class Simulation():
    def __init__(self, state):

        if isinstance(state, State):
            self._state = state
        elif isinstance(state, int):
            self._state = State(state)

        self._state.fill_random()

        self._fullname = None

        self._params = list()
        self._sample_names = list()
        self._state_names = list()

    def sweep(self, parameter, end, sweep_step='auto',
                    sample_size=100, ising_step='auto', therm='auto'):

        if therm == 'auto':
            therm = self._state._n2 * 50

        if ising_step == 'auto':
            ising_step = self._state._n2 * 2

        if parameter in ('T', 'Temperature'):
            start = self._state.T
            set_value = lambda value: self._state._set(T=value)
        elif parameter in ('J', 'Interaction'):
            start = self._state.J
            set_value = lambda value: self._state._set(J=value)
        elif parameter in ('B', 'Extern field'):
            start = self._state.B
            set_value = lambda value: self._state._set(B=value)
        else:
            print('Available parameters to sweep:')
            print('"Temperature"  or "T"')
            print('"Interaction"  or "J"')
            print('"Extern field" or "B"')
            return 0

        if sweep_step is 'auto':
            sweep_step = 0.1

        values = np.arange(start, end, sweep_step)
        n = float(len(values))
        m = 50
        temp_samples = '../data/simulations/temp/samples/sample.npy'
        temp_states = '../data/simulations/temp/states/state.npy'

        for i, value in np.ndenumerate(values):
            j = int(m*(i[0]+1)/n)
            text_bar = '*' * j + '-' * (m-j)
            print(parameter + ' = %.4f - '%(value) + text_bar + '  ', end='\r')

            # Set T and run until thermalization
            set_value(value)
            self._state.run_until(therm)

            # Fill a sample
            sample = self._state.run_sample(sample_size, ising_step)

            sample_name = sample.save_as(temp_samples)
            state_name = self._state.save_as(temp_states)

            self._params.append([self._state.T,
                                 self._state.J,
                                 self._state.B])

            self._sample_names.append(sample_name)
            self._state_names.append(state_name)

        print('Sweep completed.' + ' ' * (m+len(parameter)))

    def save_as(self, fullname, default='../data/simulations/sim.npy', verbose=False):
        if len(self._state_names) == 0:
            return 'Simulation is empty'

        if self._fullname is not None and fullname != self._fullname:
            if verbose: print('Saving a simulation copy with new name')
            copy_data = True
        else:
            copy_data = False

        fullname = Utilities.newname(fullname, default)

        self._fullname = fullname
        if self.save(copy=copy_data, verbose=verbose):
            return fullname
        else:
            return ''

    def save(self, copy=False, verbose=False):
        if self._fullname is None:
            print('First call save_as')
            return False

        self.save_data(copy=copy, verbose=verbose)

        data = [self._params,
                self._sample_names,
                self._state_names]

        if verbose: print('Saving simulation header file...', end='')
        np.save(self._fullname, data)
        if verbose: print('Done')
        return self._fullname

    def save_data(self, copy=False, verbose=False):
        path, name, extension = Utilities.splitname(self._fullname)
        sample_names = path + '/' + name + '/samples/sample.npy'
        state_names = path + '/' + name + '/states/state.npy'

        if verbose: print('Moving new samples to folder...')
        self._sample_names, ch1 = Utilities.move(files=self._sample_names,
                                                 dest=sample_names,
                                                 copy=copy,
                                                 verbose=verbose)

        if verbose: print('Moving new states to folder...')
        self._state_names, ch2 = Utilities.move(files=self._state_names,
                                                dest=state_names,
                                                copy=copy,
                                                verbose=verbose)

        return ch1, ch2

    def __len__(self): return len(self._sample_names)
    def __getitem__(self, key):
        return self._sample_names[key], self._state_names[key]

    def __repr__(self): return self._fullname
    def __str__(self): return self._fullname

    def iter(self, a, b=None, step=1):
        start = 0 if b is None else a
        end = a if b is None else b
        i = start
        while i < end:
            yield self[i]
            i += step

    @classmethod
    def load(cls, fullname, default='../data/simulations/sim.npy'):
        path, name, extension = Utilities.splitname(fullname)
        dpath, dname, dextension = Utilities.splitname(default)

        fullname = path + '/' + name + extension

        params, sample_names, state_names  = np.load(fullname)

        state = State.load(state_names[-1])
        simulation = cls(state)
        simulation._params = params
        simulation._sample_names = sample_names
        simulation._state_names = state_names
        simulation._fullname = fullname

        return simulation
