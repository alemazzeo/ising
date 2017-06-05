import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks_cwt

class Bimodal():
    @classmethod
    def gauss(cls, x, mu, sigma, A):
        return A*np.exp(-(x-mu)**2/2/sigma**2)

    @classmethod
    def bimodal(cls, x, mu1, sigma1, A1, mu2, sigma2, A2):
        return cls.gauss(x, mu1, sigma1, A1) + cls.gauss(x, mu2, sigma2, A2)

    @classmethod
    def bimodal_symmetry(cls, x, mu, sigma1, A1, sigma2, A2):
        return cls.bimodal(x,-mu, sigma1, A1, mu, sigma2, A2)

    @classmethod
    def fit_bimodal(cls, data, expected=None, symmetry=False, plot=False):
        y, x = np.histogram(data, bins='sqrt', density=True)
        x = (x[1:] + x[:-1]) / 2

        if symmetry:
            params, cov = curve_fit(cls.bimodal_symmetry, x, y, expected)
            params = [params[0], params[1], params[2],
                      params[0], params[3], params[4]]
            sigma = np.sqrt(np.diag(cov))
            sigma = [sigma[0], sigma[1], sigma[2],
                     sigma[0], sigma[3], sigma[4]]
        else:
            params, cov = curve_fit(cls.bimodal, x, y, expected)
            sigma = np.sqrt(np.diag(cov))

        if plot:
            plt.ion()
            fig, ax = plt.subplots(1)
            cls.hist(data, ax=ax)
            cls.plot_gauss(x, *params[0:3], ax=ax, label='Gauss 1', color='blue')
            cls.plot_gauss(x, *params[3:6], ax=ax, label='Gauss 2', color='black')
            cls.plot_bimodal(x, *params, ax=ax, label='Bimodal', color='red')
            ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=4, mode="expand", borderaxespad=0.)
        return params, sigma

    @classmethod
    def gaussian_pdf(cls, data, plot=False):
        y, x = np.histogram(data, bins='sqrt', density=True)
        x = (x[1:] + x[:-1]) / 2
        f = gaussian_kde(data)
        if plot:
            plt.ion()
            fig, ax = plt.subplots(1)
            cls.hist(data, ax=ax)
            cls.plot_pdf(data, ax=ax)
        return f, x, y

    @classmethod
    def estimate_pdf_peaks(cls, data, widths=None, plot=False):
        f, x, y = cls.gaussian_pdf(data)
        y_pdf = f(x)

        if widths is None:
            widths = np.arange(1,10)
        peakind = find_peaks_cwt(y_pdf, widths)

        if plot:
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
                   'bins': 'sqrt',
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
    def plot_bimodal(cls, x, mu1, sigma1, A1, mu2, sigma2, A2, ax=None, **kwargs):
        _default = {'label': 'Gauss',
                    'lw': 3,
                    'ls': '-',
                    'alpha': 0.5}
        _default.update(kwargs)

        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1)

        params = (mu1, sigma1, A1, mu2, sigma2, A2)
        curve, = ax.plot(x, cls.bimodal(x, *params), **_default)

        return curve
