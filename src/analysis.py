import matplotlib
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    try:
        matplotlib.use('qt4Agg')
    except ImportError:
        print("'Qt5Agg' or 'qt4Agg' request for interactive plot")
        raise

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, norm, exponnorm
from scipy.signal import find_peaks_cwt

from ising import Sample, State, Simulation

class Analysis():
    def __init__(self, data):

        if isinstance(data, Simulation):
            self._sample_names = data._sample_names
            self._state_names = data._state_names
        elif isinstance(data, tuple):
            self._sample_names = data[0]
            self._state_names = data[1]

        self._results = list()
        self._states = list()

        for name in self._sample_names:
            self._results.append(Result(name))
        for name in self._state_names:
            self._states.append(State.load(name))

        self._current = 0
        self._n = len(self._results)

        self._figs = list()
        self._axs = list()
        self._funcs = list()

    def __len__(self): return len(self._results)
    def __getitem__(self, key):
        return self._results[key], self._states[key]

    def subplot(self, func, *args, **kwargs):
        plt.ion()
        fig , ax = plt.subplots(*args, **kwargs)
        self._figs.append(fig)
        self._axs.append(ax)
        self._funcs.append(func)
        fig.canvas.mpl_connect('close_event',
                               lambda evt: self._on_close(evt, fig))
        fig.canvas.mpl_connect('key_press_event',
                               lambda evt: self._on_key_press(evt, fig))
        self._update()
        return fig, ax, func

    def _on_key_press(self, evt, fig):
        idx = self._figs.index(fig)
        if evt.key == 'left':
            self.left()
        if evt.key == 'right':
            self.right()

    def _on_close(self, evt, fig):
        idx = self._figs.index(fig)
        self._figs.pop(idx)
        self._axs.pop(idx)
        self._funcs.pop(idx)

    def _update(self):
        for i, func in enumerate(self._funcs):
            func(self._figs[i], self._axs[i], *self[self._current])

    @property
    def current(self): return self._current
    @current.setter
    def current(self, value):
        assert 0 < value < self._n
        self._current = value
        self._update()

    def right(self, n=1):
        if self._current + n < self._n:
            self.current += n

    def left(self, n=1):
        if self._current - n > 0:
            self.current -= n

class Result():
    def __init__(self, sample_name):
        self._sample_name = sample_name

    @property
    def magnet_array(self):
        sample = Sample.load(self._sample_name)
        return sample.magnet

    @property
    def energy_array(self):
        sample = Sample.load(self._sample_name)
        return sample.energy

class Tools():

    @classmethod
    def hist(cls, data, ax=None, **kwargs):
        params = {'normed': True,
                  'bins': 'doane',
                  'label': 'Input data',
                  'alpha': 0.3}
        params.update(kwargs)

        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1)

        return ax.hist(data, **params)

    @classmethod
    def plot_pdf(cls, x, y_pdf, ax=None, **kwargs):
        params = {'label': 'PDF',
                  'lw': 2,
                  'ls': '--',
                  'alpha': 0.7}
        params.update(kwargs)

        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1)

        curve, = ax.plot(x, y_pdf, **params)
        return curve

    @classmethod
    def plot_curve(cls, x, curve, c_params, ax=None, **kwargs):
        params = {'label': 'Curve',
                  'lw': 2,
                  'ls': '--',
                  'alpha': 0.7}
        params.update(kwargs)

        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1)

        curve, = ax.plot(x, curve(x,*c_params), **params)
        return curve


    @classmethod
    def estimate_pdf(cls, data, plot=False, ax=None):
        y, x = np.histogram(data, bins='sqrt', density=True)
        x = (x[1:] + x[:-1]) / 2
        f = gaussian_kde(data)
        if plot:
            if ax is None:
                plt.ion()
                fig, ax = plt.subplots(1)
            cls.hist(data, ax=ax)
            cls.plot_pdf(x, f(x), ax=ax)
        return f, x, y

    @classmethod
    def estimate_pdf_peaks(cls, data, widths=None, plot=False, ax=None):
        f, x, y = cls.estimate_pdf(data,plot=plot, ax=ax)
        y_pdf = f(x)

        if widths is None:
            widths = np.arange(10, int(len(x)/2))
        peakind = find_peaks_cwt(y_pdf, widths)

        if plot:
            if ax is None:
                plt.ion()
                fig, ax = plt.subplots(1)
            cls.hist(data, ax=ax)
            cls.plot_pdf(x, y_pdf, ax=ax, label='PDF', color='blue')
            for peak in peakind:
                ax.axvline(x[peak])

        return peakind, x[peakind], y_pdf[peakind]

#    @classmethod
#    def gauss(cls, x, mu, sigma, A):
#        return A*np.exp(-(x-mu)**2/2/sigma**2)

    @classmethod
    def gauss(cls, x, mu, sigma, A):
        return A * norm.pdf(x, loc=mu, scale=sigma)

    @classmethod
    def bimodal_gauss(cls, x, mu, sigma1, sigma2, A, dA):
        return (cls.gauss(x, mu, sigma1, A*dA) +
                cls.gauss(x, -mu, sigma2, A*(1-dA)))

    @classmethod
    def expnorm(cls, x, mu, sigma, lamb, A):
        return A * exponnorm.pdf(x, 1/(lamb*sigma), loc=mu, scale=sigma)

    @classmethod
    def bimodal_expnorm(cls, x, mu, sigma1, sigma2, lamb1, lamb2, A, dA):
        return (cls.expnorm(x, mu, sigma1, lamb1, A*dA) +
                cls.expnorm(1-x, 1+mu, sigma2, lamb2, A*(1-dA)))

    @classmethod
    def classificate(cls, data, plot=False, ax=None):

        if plot:
            if ax is None:
                plt.ion()
                fig, ax = plt.subplots(1)

        # Peaks of pdf
        peaks, xpeaks, ypeaks = cls.estimate_pdf_peaks(data,plot=plot, ax=ax)

        peaks = np.asarray(peaks)
        xpeaks = np.asarray(xpeaks)
        ypeaks = np.asarray(ypeaks)

        n = peaks.size

        mu = 0.0
        A = 0.1

        if n == 0:
            return 'Bimodal', mu, A

        elif n == 1:
            if xpeaks[0] > 0.5:
                mu = xpeaks[0]
                A = ypeaks[0]
                return 'Positive', mu, A

            elif xpeaks[0] < -0.5:
                mu = xpeaks[0]
                A = ypeaks[0]
                return 'Negative', mu, A

            else:
                mu1 = abs(xpeaks[0])
                A = ypeaks[0]
                return 'Bimodal', mu, A

        else:

            xp = xpeaks[xpeaks>=0]
            Ap = ypeaks[xpeaks>=0]
            xn = xpeaks[xpeaks<0]
            An = ypeaks[xpeaks<0]

            if xp.size == 0:
                mu = xn[0]
                A = An[0]
                return 'Negative', mu, A

            elif xn.size == 0:
                mu = xp[-1]
                A = Ap[-1]
                return 'Positive', mu, A

            else:
                if -xn[0] > xp[-1]:
                    mu = abs(xn[0])
                    A = An[0]
                else:
                    mu = xp[-1]
                    A = Ap[-1]
                return 'Bimodal', mu, A

    @classmethod
    def prefit(cls, data, pdf_type, mu, A, plot=False, ax=None):
        y, x = np.histogram(data, bins='doane', density=True)
        x = (x[1:] + x[:-1]) / 2

        if plot:
            if ax is None:
                plt.ion()
                fig, ax = plt.subplots(1)

        if pdf_type == 'Bimodal':
            # params: mu, sigma1, sigma2, A, dA
            par0 = [mu, 0.10, 0.10, A, 0.5]
            if -0.1 < mu < 0.1:
                mins = [0, 0.01, 0.01, 0.0000, 0.45]
                maxs = [1, 1.00, 1.00, np.inf, 0.55]
            elif -0.3 < mu < 0.3:
                mins = [0, 0.01, 0.01, 0.0000, 0.35]
                maxs = [1, 1.00, 1.00, np.inf, 0.65]
            else:
                mins = [0, 0.01, 0.01, 0.0000, 0.20]
                maxs = [1, 1.00, 1.00, np.inf, 0.80]
            bounds = (mins, maxs)
            params, cov = curve_fit(cls.bimodal_gauss, x, y, par0, bounds=bounds)
            if plot:
                y1 = cls.gauss(x, params[0], params[1], params[3]*params[4])
                y2 = cls.gauss(x, -params[0], params[2], params[3]*(1-params[4]))
                y3 = y1 + y2
                ax.plot(x, y1)
                ax.plot(x, y2)
                ax.plot(x, y3)
        elif pdf_type == 'Positive':
            # params: mu, sigma, A
            par0 = [mu, 0.10,      A]
            mins = [+0, 0.01, 0.0000]
            maxs = [+1, 0.50, np.inf]
            bounds = (mins, maxs)
            params, cov = curve_fit(cls.gauss, x, y, par0, bounds=bounds)
            if plot:
                y = cls.gauss(x, *params)
                ax.plot(x, y)
        elif pdf_type == 'Negative':
            # params: mu, sigma, A
            par0 = [mu, 0.10,      A]
            mins = [-1, 0.01, 0.0000]
            maxs = [+0, 0.50, np.inf]
            bounds = (mins, maxs)
            params, cov = curve_fit(cls.gauss, x, y, par0, bounds=bounds)
            if plot:
                y = cls.gauss(x, *params)
                ax.plot(x, y)

        sigma = np.sqrt(np.diag(cov))

        return params, sigma

    @classmethod
    def fit(cls, data, pdf_type, preparams, plot=False, ax=None):
        y, x = np.histogram(data, bins='doane', density=True, range=(-1,1))
        x = (x[1:] + x[:-1]) / 2

        if plot:
            if ax is None:
                plt.ion()
                fig, ax = plt.subplots(1)

        if pdf_type == 'Bimodal':
            # params: mu, sigma1, sigma2, lamb1, lamb2, A, dA
            p0 = np.asarray(preparams)
            pmax = p0 * 1.2
            pmin = p0 * 0.8
            p0 = p0.tolist()
            pmax = pmax.tolist()
            pmin = pmin.tolist()

            mu__, s1__, s2__, A__, dA__ = pmin
            _mu_, _s1_, _s2_, _A_, _dA_ = p0
            __mu, __s1, __s2, __A, __dA = pmax

            l1__ = 1.0
            _l1_ = 49.0
            __l1 = 50.0
            l2__ = 1.0
            _l2_ = 49.0
            __l2 = 50.0

            mins = [mu__, s1__, s2__, l1__, l2__, A__, dA__]
            par0 = [_mu_, _s1_, _s2_, _l1_, _l2_, _A_, _dA_]
            maxs = [__mu, __s1, __s2, __l1, __l2, __A, __dA]
            bounds = (mins, maxs)
            params, cov = curve_fit(cls.bimodal_expnorm, x, y, par0, bounds=bounds)
            print(params)
            if plot:
                y1 = cls.expnorm(x,   params[0], params[1], params[3], params[5]*params[6])
                y2 = cls.expnorm(1-x, 1+params[0], params[2], params[4], params[5]*(1-params[6]))
                y3 = y1 + y2

                ax.plot(x, y1, label='A')
                ax.plot(x, y2, label='B')
                ax.plot(x, y3, label='C')

                ax.legend(loc='best')
        elif pdf_type == 'Positive':
            return
        elif pdf_type == 'Negative':
            return

        #sigma = np.sqrt(np.diag(cov))

        return params#, sigma
