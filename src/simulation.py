import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt

from ising import Ising
from bimodal import Bimodal as bm

class Simulation(Ising):
    def __init__(self, n, **kwargs):
        _config = {'J': 1.0,
                   'B': 0.0}

        _config.update(kwargs)

        Ising.__init__(self, n)
        self.fill_random()

        self.J = _config['J']
        self.B = _config['B']

        self._temps = np.np.array([], dtype=float)
        self._samples = list()

    def sweep_temps(self, tf, ti=None, temp_step=0.05,
                    sample_size=10000, ising_step='auto', therm='auto'):

        if therm == 'auto':
            therm = self._n2 * 50

        if ising_step == 'auto':
            ising_step = self._n2 * 2

        temps = np.arange(ti, tf, temp_step)
        n = float(len(temps))
        samples = list()

        for i, T in np.ndenumerate(temps):
            text_bar = '*' * int(i/n) + '-' * int(1-i/n)
            print(text_bar + 'T= ' + str(T) + ' '*10, end='\r')
            # Set T and run until thermalization
            self.T = T
            self.run_until(therm)

            # Fill a sample
            sample = self.run_sample(sample_size, ising_step)
            # Store sample in a list
            samples.append(sample)

        self._temps = np.concatenate((self._temps, temps))
        self._samples.append(samples)

    def

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
                 path='../datos/', name=None):

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

        if name is None:
            name = self._name

        if path is None:
            path = self._path

        os.makedirs(path, exist_ok=True)

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

        np.save(path+name, [params,data])

    @classmethod
    def load(cls, name, path='../datos/'):

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
