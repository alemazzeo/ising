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
from scipy.stats import gaussian_kde, exponnorm
from scipy.signal import find_peaks_cwt
from scipy.misc import factorial

class Bimodal():

    @classmethod
    def gauss(cls, x, mu, sigma, A):
        return A*np.exp(-(x-mu)**2/2/sigma**2)

    @classmethod
    def bimodal_gauss(cls, x, mu1, sigma1, A1, mu2, sigma2, A2):
        return cls.gauss(x, mu1, sigma1, A1) + cls.gauss(x, mu2, sigma2, A2)

    @classmethod
    def poisson(cls, k, lamb):
        return (lamb**k/factorial(k)) * np.exp(-lamb)

    @classmethod
    def bimodal_poisson(cls, k, A, lamb):
        return (A * cls.poisson(k, lamb) + (1-A) * cls.poisson(20-k,lamb))

    @classmethod
    def expnorm(cls, x, mu, sigma, lamb, A):
        return A * exponnorm.pdf(x, 1/(lamb*sigma), loc=mu, scale=sigma)

    @classmethod
    def fit_gaussian(cls, data, expected=None, bounds=(-np.inf,np.inf), plot=False, ax=None):
        y, x = np.histogram(data, density=True)
        x = (x[1:] + x[:-1]) / 2

        params, cov = curve_fit(cls.gauss, x, y, expected, bounds=bounds)
        sigma = np.sqrt(np.diag(cov))

        if plot:
            if ax is None:
                plt.ion()
                fig, ax = plt.subplots(1)
            cls.hist(data, ax=ax)
            cls.plot_gauss(x, *params[0:3], ax=ax, label='Gaussian', color='blue')
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=4, mode="expand", borderaxespad=0.)
        return params, sigma

    @classmethod
    def fit_bimodal(cls, data, expected=None, bounds=(-np.inf,np.inf), plot=False, ax=None):
        y, x = np.histogram(data, density=True)
        x = (x[1:] + x[:-1]) / 2

        params, cov = curve_fit(cls.bimodal_gauss, x, y, expected, bounds=bounds)
        sigma = np.sqrt(np.diag(cov))

        if plot:
            if ax is None:
                plt.ion()
                fig, ax = plt.subplots(1)
            cls.hist(data, ax=ax)
            cls.plot_gauss(x, *params[0:3], ax=ax, label='Gaussian 1', color='blue')
            cls.plot_gauss(x, *params[3:6], ax=ax, label='Gaussian 2', color='black')
            cls.plot_bimodal_gauss(x, *params, ax=ax, label='Bimodal', color='red')
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=4, mode="expand", borderaxespad=0.)
        return params, sigma

    @classmethod
    def gaussian_pdf(cls, data, plot=False, ax=None):
        y, x = np.histogram(data, bins='sqrt', density=True)
        x = (x[1:] + x[:-1]) / 2
        f = gaussian_kde(data)
        if plot:
            if ax is None:
                plt.ion()
                fig, ax = plt.subplots(1)
            cls.hist(data, ax=ax)
            cls.plot_pdf(data, ax=ax)
        return f, x, y

    @classmethod
    def estimate_pdf_peaks(cls, data, widths=None, plot=False, ax=None):
        f, x, y = cls.gaussian_pdf(data)
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

    @classmethod
    def hist(cls, data, ax=None, **kwargs):
        params = {'normed': True,
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

        curve, = ax.plot(x, y_pdf, label='PDF',
                    color='blue', lw=2, ls='--', alpha=0.5)
        return curve

    @classmethod
    def plot_gauss(cls, x, mu, sigma, A, ax=None, annotate=True, **kwargs):
        _default = {'label': 'Gauss',
                    'lw': 2,
                    'ls': '--',
                    'alpha': 0.5}
        _default.update(kwargs)

        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1)

        curve, = ax.plot(x, cls.gauss(x, mu, sigma, A), **_default)

        if annotate:
            text = '$\mu=%.2f$\n$\sigma=%.2f$'%(mu, sigma)
            ax.annotate(text, xy=(mu, A/6.0), xytext=(0, 0),
                        textcoords='offset points',
                        size=12, ha='center', va='bottom')

        return curve

    @classmethod
    def plot_bimodal_gauss(cls, x, mu1, sigma1, A1, mu2, sigma2, A2, ax=None, **kwargs):
        _default = {'label': 'Gauss',
                    'lw': 3,
                    'ls': '-',
                    'alpha': 0.5}
        _default.update(kwargs)

        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1)

        params = (mu1, sigma1, A1, mu2, sigma2, A2)
        curve, = ax.plot(x, cls.bimodal_gauss(x, *params), **_default)

        return curve
