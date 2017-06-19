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

from tools import Tools


class Sample(C.Structure):
    _fields_ = [("_n", C.c_int),
                ("_sample_size", C.c_int),
                ("_step_size", C.c_int),
                ("_tolerance", C.c_double),
                ("_T", C.c_double),
                ("_J", C.c_double),
                ("_B", C.c_double),
                ("_p_energy", C.POINTER(C.c_double)),
                ("_p_magnet", C.POINTER(C.c_int)),
                ("_p_flips", C.POINTER(C.c_int)),
                ("_p_total_flips", C.POINTER(C.c_int)),
                ("_p_q", C.POINTER(C.c_double)),
                ("_v2", C.c_int)]

    def __init__(self, sample_size, step_size=None, tolerance=None, v2=False):

        self._sample_size = sample_size
        self._step_size = step_size
        self._tolerance = tolerance
        self._fullname = None
        self._v2 = v2

        # Memory alloc
        self._energy = np.zeros(self._sample_size, dtype=C.c_double)
        self._magnet = np.zeros(self._sample_size, dtype=C.c_int)
        self._flips = np.zeros(self._sample_size, dtype=C.c_int)
        self._total_flips = np.zeros(self._sample_size, dtype=C.c_int)
        self._q = np.zeros(self._sample_size, dtype=C.c_double)

        # Pointers
        self._p_energy = self._energy.ctypes.data_as(C.POINTER(C.c_double))
        self._p_magnet = self._magnet.ctypes.data_as(C.POINTER(C.c_int))
        self._p_flips = self._flips.ctypes.data_as(C.POINTER(C.c_int))
        self._p_total_flips = self._total_flips.ctypes.data_as(C.POINTER(C.c_int))
        self._p_q = self._q.ctypes.data_as(C.POINTER(C.c_double))

    def save_as(self, fullname='', default='../data/samples/sample.npy'):
        fullname = Tools.newname(fullname, default)
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
                  self._B,
                  self._v2]

        data = [self._energy,
                self._magnet,
                self._flips,
                self._total_flips,
                self._q]

        np.save(self._fullname, [params,data])
        return self._fullname

    @classmethod
    def load(cls, fullname, default='../data/samples/sample.npy'):
        path, name, extension = Tools.splitname(fullname)
        dpath, dname, dextension = Tools.splitname(default)

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
        load_sample._v2 = params[6]

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
    def magnet(self): return self._magnet / (self._n**2)

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
        Tools.plot_hist(self.energy, ax=ax, **kwargs)

    def view_magnet(self, ax=None, **kwargs):
        Tools.plot_hist(self.magnet, ax=ax, **kwargs)

    def view_flips(self, ax=None, **kwargs):
        Tools.plot_array1D(self.flips, ax=ax, **kwargs)

    def view_total_flips(self, ax=None, **kwargs):
        Tools.plot_array1D(self.total_flips, ax=ax, **kwargs)

    def view_q(self, ax=None, **kwargs):
        Tools.plot_array1D(self.q, ax=ax, **kwargs)

class Ising(C.Structure):
    _fields_ = [("_p_lattice", C.POINTER(C.c_int)),
                ("_n", C.c_int),
                ("_n2", C.c_int),
                ("_flips", C.c_int),
                ("_total_flips", C.c_int),
                ("_T", C.c_double),
                ("_J", C.c_double),
                ("_B", C.c_double),
                ("_p_dEs", C.c_double * 18),
                ("_p_exps", C.c_double * 18),
                ("_current_energy", C.c_double),
                ("_current_magnet", C.c_int),
                ("_p_energy", C.POINTER(C.c_double)),
                ("_p_magnet", C.POINTER(C.c_int)),
                ("_v2", C.c_int)]

    def __init__(self, n, v2=False):

        self._n = n
        self._step_size = None
        self._v2 = v2

        # C Library
        self._lib = C.CDLL('./libising.so')

        # Functions
        self.C_init = self._lib.init
        self.C_set_params = self._lib.set_params
        self.C_info = self._lib.info
        self.C_metropolis = self._lib.metropolis
        self.C_metropolis_v2 = self._lib.metropolis_v2
        self.C_run = self._lib.run
        self.C_run_v2 = self._lib.run_v2
        self.C_run_until = self._lib.run_until
        self.C_run_until_v2 = self._lib.run_until_v2
        self.C_run_sample = self._lib.run_sample
        self.C_run_sample_v2 = self._lib.run_sample_v2
        self.C_pick_site = self._lib.pick_site
        self.C_flip = self._lib.flip
        self.C_flip_v2 = self._lib.flip_v2
        self.C_first_neighbors = self._lib.first_neighbors
        self.C_second_neighbors = self._lib.second_neighbors
        self.C_cost = self._lib.cost
        self.C_try_flip = self._lib.try_flip
        self.C_accept_flip = self._lib.accept_flip
        self.C_calc_pi = self._lib.calc_pi
        self.C_calc_energy = self._lib.calc_energy
        self.C_calc_magnet = self._lib.calc_magnet
        self.C_calc_lattice = self._lib.calc_lattice
        self.C_calc_lattice_v2 = self._lib.calc_lattice_v2

        # Return types
        self.C_init.restype = C.c_int
        self.C_set_params.restype = C.c_int
        self.C_info.restype = C.c_int
        self.C_metropolis.restype = C.c_int
        self.C_metropolis_v2.restype = C.c_int
        self.C_run.restype = C.c_int
        self.C_run_v2.restype = C.c_int
        self.C_run_until.restype = C.c_double
        self.C_run_until_v2.restype = C.c_double
        self.C_run_sample.restype = C.c_int
        self.C_run_sample_v2.restype = C.c_int
        self.C_pick_site.restype = C.c_int
        self.C_flip.restype = C.c_int
        self.C_flip_v2.restype = C.c_int
        self.C_first_neighbors.restype = C.c_int
        self.C_second_neighbors.restype = C.c_int
        self.C_cost.restype = C.c_int
        self.C_try_flip.restype = C.c_int
        self.C_accept_flip.restype = C.c_int
        self.C_calc_pi.restype = C.c_double
        self.C_calc_energy.restype = C.c_double
        self.C_calc_magnet.restype = C.c_int
        self.C_calc_lattice.restype = C.c_int
        self.C_calc_lattice_v2.restype = C.c_int

        # Arguments types
        self.C_init.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_set_params.argtypes = [C.POINTER(Ising), C.c_double,
                                      C.c_double, C.c_double]
        self.C_info.argtypes = [C.POINTER(Ising)]
        self.C_metropolis.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_metropolis_v2.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_run.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_run_v2.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_run_until.argtypes = [C.POINTER(Ising), C.c_int, C.c_double]
        self.C_run_until_v2.argtypes = [C.POINTER(Ising), C.c_int, C.c_double]
        self.C_run_sample.argtypes = [C.POINTER(Ising), C.POINTER(Sample)]
        self.C_run_sample_v2.argtypes = [C.POINTER(Ising), C.POINTER(Sample)]
        self.C_pick_site.argtypes = [C.POINTER(Ising)]
        self.C_flip.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_flip_v2.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_first_neighbors.argtypes = [C.POINTER(Ising), C.c_int, C.c_int * 4]
        self.C_second_neighbors.argtypes = [C.POINTER(Ising), C.c_int, C.c_int * 4]
        self.C_cost.argtypes = [C.POINTER(Ising), C.c_int, C.c_int * 4]
        self.C_try_flip.argtypes = [C.POINTER(Ising), C.c_double]
        self.C_accept_flip.argtypes = [C.POINTER(Ising), C.c_int, C.c_int]
        self.C_calc_pi.argtypes = [C.POINTER(Ising), C.c_int, C.c_int]
        self.C_calc_energy.argtypes = [C.POINTER(Ising), C.c_int, C.c_int]
        self.C_calc_magnet.argtypes = [C.POINTER(Ising), C.c_int]
        self.C_calc_lattice.argtypes = [C.POINTER(Ising)]
        self.C_calc_lattice_v2.argtypes = [C.POINTER(Ising)]

        # Step size and first fill
        self.step_size = int(1.5 * n**2)
        self.fill_random()

        # Init values and calc magnet and energy
        self.C_init(self, n)
        self.calc_lattice()

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
        if self._v2:
            self.C_calc_lattice_v2(self)
        else:
            self.C_calc_lattice(self)

    def fill_random(self, prob=0.5):
        random = np.ones(self._n**2)
        random[np.random.rand(self._n**2) > prob] *= -1
        self.assign_lattice(random)
        self.calc_lattice()

    def run(self, step_size=None):
        if step_size is not None:
            self.step_size = step_size
        if self._v2:
            nflips = self.C_run_v2(self, self.step_size)
        else:
            nflips = self.C_run(self, self.step_size)
        return nflips

    def run_until(self, step_size=None, tolerance=10.0):
        if step_size is not None:
            self.step_size = step_size
        if self._v2:
            q = self.C_run_until_v2(self, self.step_size, tolerance)
        else:
            q = self.C_run_until(self, self.step_size, tolerance)
        return q

    def run_sample(self, sample_size, step_size=None, tolerance=10.0):
        if step_size is not None:
            self.step_size = step_size
        if self._v2:
            data = Sample(sample_size, self.step_size, tolerance, v2=True)
            self.C_run_sample_v2(self, data)
        else:
            data = Sample(sample_size, self.step_size, tolerance)
            self.C_run_sample(self, data)
        return data


class State(Ising):
    def __init__(self, n, v2=False):
        super().__init__(n, v2)
        self._fullname = None

    def save_as(self, fullname='', default='../data/states/state.npy'):
        fullname = Tools.newname(fullname, default)
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
                  self._B,
                  self._v2]

        data = self._lattice

        np.save(self._fullname, [params, data])
        return self._fullname

    @classmethod
    def load(cls, fullname, default='../data/states/state.npy'):
        path, name, extension = Tools.splitname(fullname)
        dpath, dname, dextension = Tools.splitname(default)

        if path == '':
            path = dpath
        if name == '':
            name = dname
        if extension == '':
            extension = dextension

        fullname = path + '/' + name + extension

        params, data = np.load(fullname)

        load_state = cls(n=int(params[0]), v2=params[5])
        load_state._total_flips = int(params[1])

        load_state.T = float(params[2])
        load_state.J = float(params[3])
        load_state.B = float(params[4])
        load_state._v2 = params[5]

        load_state._lattice = data
        pointer = load_state._lattice.ctypes.data_as(C.POINTER(C.c_int))
        load_state._p_lattice = pointer
        load_state.calc_lattice()
        load_state._fullname = fullname

        return load_state

    def plot_lattice(self, ax=None, **kwargs):
        curve, ax, fig = Tools.plot_lattice(self._lattice, ax=ax, **kwargs)
        return curve, ax, fig

    
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

    @property
    def T(self): return self._state.T

    @T.setter
    def T(self, value): self._state.T = value

    @property
    def J(self): return self._state.J

    @J.setter
    def J(self, value): self._state.J = value

    @property
    def B(self): return self._state.B

    @B.setter
    def B(self, value): self._state.B = value

    def therm(self, T=None, J=None, B=None, therm='auto'):
        if therm == 'auto':
            therm = self._state._n2 * 50

        self._state._set(T=T, J=J, B=B)
        self._state.run_until(therm)

    def sweep(self, T=None, sweep_step='auto', sample_size=100,
              ising_step='auto', therm='auto', J=None, B=None):

        if therm == 'auto':
            therm = self._state._n2 * 50

        if ising_step == 'auto':
            ising_step = self._state._n2 * 2

        if T is not None and B is None and J is None:
            start = self._state.T
            end = T
            parameter = 'Temperature'
            set_value = lambda value: self._state._set(T=value)

        elif J is not None and T is None and B is None:
            start = self._state.J
            end = J
            parameter = 'Interaction'
            set_value = lambda value: self._state._set(J=value)

        elif B is not None and T is None and J is None:
            start = self._state.B
            end = B
            parameter = 'Extern Field'
            set_value = lambda value: self._state._set(B=value)

        else:
            raise ValueError

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
            print(parameter + ' = %.4f - ' % (value) + text_bar + '  ',
                  end='\r')

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

    def save_as(self, fullname, default='../data/simulations/sim.npy',
                verbose=False):
        if len(self._state_names) == 0:
            return 'Simulation is empty'

        if self._fullname is not None and fullname != self._fullname:
            if verbose: print('Saving a simulation copy with new name')
            copy_data = True
        else:
            copy_data = False

        fullname = Tools.newname(fullname, default)

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
        path, name, extension = Tools.splitname(self._fullname)
        sample_names = path + '/' + name + '/samples/sample.npy'
        state_names = path + '/' + name + '/states/state.npy'

        if verbose: print('Moving new samples to folder...')
        self._sample_names, ch1 = Tools.move(files=self._sample_names,
                                             dest=sample_names,
                                             copy=copy,
                                             verbose=verbose)

        if verbose: print('Moving new states to folder...')
        self._state_names, ch2 = Tools.move(files=self._state_names,
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
        path, name, extension = Tools.splitname(fullname)
        dpath, dname, dextension = Tools.splitname(default)

        fullname = path + '/' + name + extension

        params, sample_names, state_names = np.load(fullname)

        state = State.load(state_names[-1])
        simulation = cls(state)
        simulation._params = params
        simulation._sample_names = sample_names
        simulation._state_names = state_names
        simulation._fullname = fullname

        return simulation
