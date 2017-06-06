import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt
import os

from ising import Ising
from bimodal import Bimodal as bm

class Simulation(Ising):
    def __init__(self, n, name='sim', path='../data/simulations'):

        Ising.__init__(self, n)
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
