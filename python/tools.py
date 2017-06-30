import matplotlib
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    try:
        matplotlib.use('qt4Agg')
    except ImportError:
        raise

import numpy as np
import matplotlib.pyplot as plt

import os
import shutil

from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, norm, exponnorm
from scipy.signal import find_peaks_cwt


class Tools():

    @classmethod
    def file_exist(cls, fullname, default_path='./'):
        path, name, extension = cls.splitname(fullname)
        name0 = fullname + extension
        name1 = default_path + name
        name2 = default_path + name + extension

        if os.path.isfile(fullname):
            return fullname
        elif os.path.isfile(name0):
            return name0
        elif os.path.isfile(name1):
            return name1
        elif os.path.isfile(name2):
            return name2

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

        if os.path.isfile(path + '/' + name + '0' + extension):
            i = 0
            newname = name + str(i)
            while os.path.isfile(path + '/' + newname + extension):
                i += 1
                newname = name + str(i)
            name = newname
        else:
            name = name + '0'

        return path + '/' + name + extension

    @classmethod
    def lastname(cls, fullname, default='../data/temp.npy'):
        path, name, extension = cls.splitname(cls.newname(fullname, default))
        return path + '/' + name[:-1] + str(int(name[-1]) - 1) + extension

    @classmethod
    def move(cls, files, dest, copy=False, verbose=False):
        changes = list()
        newlist = list()
        for fullname in files:
            path, name, extension = cls.splitname(fullname)
            dpath, dname, dextension = cls.splitname(dest)

            if (path == dpath and name[:len(dname)] == dname):
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
        data = np.reshape(data1D, (n, n))

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
        assert ax is None or len(ax) == 2
        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(2)

        x = data[0:int(n / 100)]
        acorr = cls.autocorrelation(x)
        curve_d = cls.plot_step(x, ax=ax[0], **plot1_kw)
        curve_a = cls.plot_step(acorr, ax=ax[1], **plot2_kw)

        for i in range(99):
            x = data[0:int(n * (i + 2) / 100)]
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
    def autocorrelation(cls, x):
        xp = x - np.mean(x)
        f = np.fft.fft(xp)
        p = np.array([np.real(v)**2 + np.imag(v)**2 for v in f])
        pi = np.fft.ifft(p)
        return np.real(pi)[:int(x.size / 2)] / np.sum(xp**2)

    @classmethod
    def autocorrelation2(cls, x):
        n = len(x)
        var = np.var(x)
        acorr = np.zeros(n, dtype=float)
        for k in range(n):
            acorr[k] = np.sum((x[0:n - k] - np.mean(x[0:n - k]))
                              * (x[k:n] - np.mean(x[k:n]))) / (n * var)
        return acorr

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

        curve, = ax.plot(x, curve(x, *c_params), **params)
        return curve

    @classmethod
    def estimate_pdf(cls, data, plot=False, ax=None):
        y, x = np.histogram(data, bins='auto', density=True)
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
        f, x, y = cls.estimate_pdf(data, plot=plot, ax=ax)
        y_pdf = f(x)

        if widths is None:
            if int(len(x) / 2) > 15:
                widths = np.arange(5, int(len(x) / 2))
            else:
                widths = np.arange(5, 15)
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
    def gauss(cls, x, mu, sigma, A):
        return A * norm.pdf(x, loc=mu, scale=sigma)

    @classmethod
    def bimodal_gauss(cls, x, mu, sigma, ds, A, dA):
        return (cls.gauss(x, mu, sigma * ds, A * dA) +
                cls.gauss(x, -mu, sigma * (1 - ds), A * (1 - dA)))

    @classmethod
    def expnorm(cls, x, mu, sigma, lamb, A):
        return A * exponnorm.pdf(x, 1 / (lamb * sigma), loc=mu, scale=sigma)

    @classmethod
    def bimodal_expnorm(cls, x, mu, sigma, ds, lamb, dl, A, dA):
        return (cls.expnorm(1 - x, 1 - mu, sigma * ds,
                            lamb * dl, A * dA) +
                cls.expnorm(x + 1, 1 - mu, sigma * (1 - ds),
                            lamb * (1 - dl), A * (1 - dA)))

    @classmethod
    def plot_fit(cls, data, pdf_type, params, sigmas,
                 ax=None, hist=True, *args, **kwargs):
        print('pdf_type', pdf_type)
        print('params', params)
        print('\n')
        if ax is None:
            plt.ion()
            fig, ax = plt.subplots(1)
        if hist:
            ax.hist(data, bins='doane', normed=True,
                    range=(-1, 1), label='Data')

        if pdf_type == 'Norm Bimodal':
            x = np.linspace(-1, 1, 200)
            y1 = cls.gauss(x, params[0],
                           params[1] * params[2],
                           params[3] * params[4])
            y2 = cls.gauss(x, -params[0],
                           params[1] * (1 - params[2]),
                           params[3] * (1 - params[4]))
            y3 = cls.bimodal_gauss(x, *params)
            ax.plot(x, y1, label='Positive')
            ax.plot(x, y2, label='Negative')
            ax.plot(x, y3, label='Bimodal')

        elif pdf_type == 'Norm Positive':
            x = np.linspace(-1, 1, 200)
            y = cls.gauss(x, *params)
            ax.plot(x, y, label=pdf_type)

        elif pdf_type == 'Norm Negative':
            x = np.linspace(-1, 1, 200)
            y = cls.gauss(x, *params)
            ax.plot(x, y, label=pdf_type)

        elif pdf_type == 'ExponNorm Bimodal':
            x = np.linspace(-1, 1, 200)
            y = cls.bimodal_expnorm(x, *params)
            ax.plot(x, y, label='Bimodal')

            y1 = cls.expnorm(1 - x, 1 - params[0],
                             params[1] * params[2],
                             params[3] * params[4],
                             params[5] * params[6])
            y2 = cls.expnorm(1 + x, 1 - params[0],
                             params[1] * (1 - params[2]),
                             params[3] * (1 - params[4]),
                             params[5] * (1 - params[6]))

            ax.plot(x, y1, label='Positive')
            ax.plot(x, y2, label='Negative')

        elif pdf_type in ('ExponNorm Positive', 'ExponNorm Negative'):
            x = np.linspace(-1, 1, 200)
            if pdf_type == 'ExponNorm Negative':
                y = cls.expnorm(1 + x, 1 - params[0], params[1], params[2],
                                params[3])
            else:
                y = cls.expnorm(1 - x, 1 - params[0], params[1], params[2],
                                params[3])
            ax.plot(x, y, label=pdf_type)

        ax.legend(loc='best')

    @classmethod
    def classificate(cls, data, plot=False, ax=None):

        if plot:
            if ax is None:
                plt.ion()
                fig, ax = plt.subplots(1)

        # Peaks of pdf
        peaks, xpeaks, ypeaks = cls.estimate_pdf_peaks(data,
                                                       plot=plot,
                                                       ax=ax)

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
                mu = abs(xpeaks[0])
                A = ypeaks[0]
                return 'Bimodal', mu, A

        else:

            xp = xpeaks[xpeaks >= 0]
            Ap = ypeaks[xpeaks >= 0]
            xn = xpeaks[xpeaks < 0]
            An = ypeaks[xpeaks < 0]

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
    def prefit(cls, data, pdf_type, mu, A):
        y, x = np.histogram(data, bins='doane', density=True, range=(-1, 1))
        x = (x[1:] + x[:-1]) / 2

        if pdf_type == 'Bimodal':
            # params: mu, sigma, ds, A, dA
            if A > 3.0:
                A = 2.8
            par0 = [mu, 0.15, 0.5, A, 0.5]
            if -0.1 < mu < 0.1:
                mins = [0, 0.10, 0.45, 0.0, 0.45]
                maxs = [1, 0.80, 0.55, 3.0, 0.55]
            elif -0.3 < mu < 0.3:
                mins = [0, 0.10, 0.35, 0.0, 0.35]
                maxs = [1, 0.60, 0.65, 3.0, 0.65]
            elif -0.5 < mu < 0.5:
                mins = [0, 0.01, 0.20, 0.0000, 0.20]
                maxs = [1, 0.30, 0.80, np.inf, 0.80]
            else:
                mins = [0, 0.01, 0.00, 0.0000, 0.00]
                maxs = [1, 0.20, 1.00, np.inf, 1.00]
            bounds = (mins, maxs)
            params, cov = curve_fit(cls.bimodal_gauss, x, y,
                                    par0, bounds=bounds)

        elif pdf_type == 'Positive':
            # params: mu, sigma, A
            par0 = [mu, 0.10,      A]
            mins = [+0, 0.01, 0.0000]
            maxs = [+1, 0.50, np.inf]
            bounds = (mins, maxs)
            params, cov = curve_fit(cls.gauss, x, y, par0, bounds=bounds)

        elif pdf_type == 'Negative':
            # params: mu, sigma, A
            par0 = [mu, 0.10,      A]
            mins = [-1, 0.01, 0.0000]
            maxs = [+0, 0.50, np.inf]
            bounds = (mins, maxs)
            params, cov = curve_fit(cls.gauss, x, y, par0, bounds=bounds)

        sigmas = np.sqrt(np.diag(cov))

        return 'Norm ' + pdf_type, params, sigmas

    @classmethod
    def fit(cls, data, pdf_type, preparams, presigmas):
        y, x = np.histogram(data, bins='doane',
                            density=True, range=(-1, 1))
        x = (x[1:] + x[:-1]) / 2

        _, pdf_type = pdf_type.split(' ')

        if pdf_type == 'Bimodal':
            # params: mu, sigma1, sigma2, lamb1, lamb2, A, dA
            p0 = np.asarray(preparams)
            pmax = p0 * 1.3
            pmin = p0 * 0.7
            p0 = p0.tolist()
            pmax = pmax.tolist()
            pmin = pmin.tolist()

            mu__, s__, ds__, A__, dA__ = pmin
            _mu_, _s_, _ds_, _A_, _dA_ = p0
            __mu, __s, __ds, __A, __dA = pmax

            if(mu__ < 0):
                mu__ = 0
            if(__mu > 1):
                __mu = 1

            if -0.1 < _mu_ < 0.1:
                l__, _l_, __l = 2.0, 40.0, 160.0
                dl__, _dl_, __dl = 0.45, 0.50, 0.55
            elif -0.3 < _mu_ < 0.3:
                l__, _l_, __l = 2.0, 40.0, 160.0
                dl__, _dl_, __dl = 0.40, 0.50, 0.60
            elif -0.5 < _mu_ < 0.5:
                l__, _l_, __l = 2.0, 40.0, 160.0
                dl__, _dl_, __dl = 0.35, 0.50, 0.65
            else:
                l__, _l_, __l = 2.0, 40.0, 160.0
                dl__, _dl_, __dl = 0.20, 0.5, 0.80

            mins = [mu__, s__, ds__, l__, dl__, A__, dA__]
            par0 = [_mu_, _s_, _ds_, _l_, _dl_, _A_, _dA_]
            maxs = [__mu, __s, __ds, __l, __dl, __A, __dA]
            bounds = (mins, maxs)
            params, cov = curve_fit(cls.bimodal_expnorm, x, y, par0,
                                    bounds=bounds)

        elif pdf_type in ('Positive', 'Negative'):
            # params: mu, sigma, lamb, A

            p0 = np.asarray(preparams)
            if pdf_type == 'Positive':
                p0[0] = 1 - p0[0]
            else:
                p0[0] = 1 + p0[0]

            pmax = p0 * 1.2
            pmin = p0 * 0.8
            p0 = p0.tolist()
            pmax = pmax.tolist()
            pmin = pmin.tolist()

            mu__, s1__, A__, = pmin
            _mu_, _s1_, _A_, = p0
            __mu, __s1, __A, = pmax

            l__ = 1.0
            _l_ = 20.0
            __l = 50.0

            mins = [mu__, s1__, l__, A__]
            par0 = [_mu_, _s1_, _l_, _A_]
            maxs = [__mu, __s1, __l, __A]
            bounds = (mins, maxs)
            if pdf_type == 'Negative':
                params, cov = curve_fit(cls.expnorm, 1 + x, y, par0,
                                        bounds=bounds)
            else:
                params, cov = curve_fit(cls.expnorm, 1 - x, y, par0,
                                        bounds=bounds)

            params[0] = 1 - params[0]

        sigmas = np.sqrt(np.diag(cov))

        return 'ExponNorm ' + pdf_type, params, sigmas

    @classmethod
    def interpret(cls, pdf_type, params, sigmas):
        x = np.linspace(-1, 1, 1000)
        if pdf_type == 'ExponNorm Bimodal':
            mu, sigma, ds, lamb, dl, A, dA = params

            expnorm1 = cls.expnorm(1 - x, 1 - mu, sigma * ds,
                                   lamb * dl, A * dA)
            expnorm2 = cls.expnorm(1 + x, 1 - mu, sigma * (1 - ds),
                                   lamb * (1 - dl), A * (1 - dA))

            mode1 = x[np.argmax(expnorm1)]
            mode2 = x[np.argmax(expnorm2)]
            mean1 = mu + 1 / (lamb * dl)
            mean2 = mu + 1 / (lamb * (1 - dl))
            sd1 = np.sqrt(sigma * ds + (1 / (lamb * dl)**2)) / 2
            sd2 = np.sqrt(sigma * (1 - ds) + (1 / (lamb * (1 - dl))**2)) / 2
            return [mode1, sd1, dA], [mode2, sd2, 1 - dA]

        elif pdf_type in ('ExponNorm Positive', 'ExponNorm Negative'):
            mu, sigma, lamb, A = params

            if pdf_type == 'ExponNorm Positive':
                expnorm = cls.expnorm(1 - x, 1 - mu, sigma, lamb, A)
            else:
                expnorm = cls.expnorm(1 + x, 1 - mu, sigma, lamb, A)

            mode = x[np.argmax(expnorm)]
            mean = abs(mu) + 1 / lamb
            sd = np.sqrt(sigma + (1 / lamb**2)) / 2
            if pdf_type == 'ExponNorm Positive':
                return [mode, sd, 1.0], [0.0, 1.0, 0.0]
            else:
                return [0.0, 1.0, 0.0], [mode, sd, 1.0]

        elif pdf_type == 'Norm Bimodal':
            mu, sigma, ds, A, dA = params
            return [mu, sigma * ds, A * dA], [-mu, sigma * (1 - ds), A * (1 - dA)]
        elif pdf_type in ('Norm Positive', 'Norm Negative'):
            mu, sigma, A = params
            mean = abs(mu)
            sd = np.sqrt(sigma)
            if pdf_type == 'Norm Positive':
                return [mean, sd, 1.0], [0.0, 1.0, 0.0]
            else:
                return [0.0, 1.0, 0.0], [-mean, sd, 1.0]
        else:
            raise ValueError
