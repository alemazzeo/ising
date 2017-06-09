import matplotlib
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    try:
        matplotlib.use('qt4Agg')
    except ImportError:
        print("'Qt5Agg' or 'qt4Agg' request for interactive plot")
        raise

from ising import Simulation, Ising, State, Sample
from analysis import Analysis, Result
from bimodal import Bimodal
import numpy as np
import matplotlib.pyplot as plt

sim1 = Simulation.load('../data/simulations/1stSweep0.npy')

a = 20
x = np.linspace(0,a,100)
m = np.linspace(-1,1,100)
y1 = Bimodal.poisson(x,10.5)
y2 = Bimodal.poisson(a-x,10.5)

plt.ion()
res = Result(sim1[26][0])
magnet = res._sample.magnet/1024
fig, ax = plt.subplots(1)

Bimodal.hist((magnet+1)/2*a, ax=ax)
curve1, = ax.plot(x,y1)
curve2, = ax.plot(x,y2)
curve3, = ax.plot(x,y1+y2)

def foo(p,A):
    y1 = A * Bimodal.poisson(x,p)
    y2 = (1-A) * Bimodal.poisson(a-x,p)
    y3 = y1 + y2
    curve1.set_data(x,y1)
    curve2.set_data(x,y2)
    curve3.set_data(x,y3)
    ax.relim()
    ax.autoscale()
    return np.trapz(x, y3)

for j in range(25,30):
    ax.cla()
    res = Result(sim1[j][0])
    magnet = res._sample.magnet/1024
    Bimodal.hist((magnet+1)/2*a, ax=ax)
    curve1, = ax.plot(x,y1)
    curve2, = ax.plot(x,y2)
    curve3, = ax.plot(x,y1+y2)

    for i in range(100):
        foo(i/200*a,0.5)
        plt.draw()
        plt.pause(0.001)
#    for i in range(100):
#        foo((100-i)/200*a,0.5)
#        plt.draw()
#        plt.pause(0.001)
