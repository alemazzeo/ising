from ising import Simulation

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sim', type=str,
                    default='../data/simulations/Simulation2.npy')
parser.add_argument('-N', type=int, default=32)
parser.add_argument('-T', type=float, default=4.0)
parser.add_argument('-J', type=float, default=1.0)
parser.add_argument('-B', type=float, default=0.0)
parser.add_argument('-Tf', type=float, default=None)
parser.add_argument('-Jf', type=float, default=None)
parser.add_argument('-Bf', type=float, default=None)
parser.add_argument('-sample_size', type=int, default=1000)
parser.add_argument('-sweep_step', type=float, default=0.01)
parser.add_argument('-tolerance', type=float, default=1)

params = parser.parse_args()

tol = params.tolerance
sample = params.sample_size
sweep = params.sweep_step

Tf = params.Tf
Jf = params.Jf
Bf = params.Bf

sim = Simulation(32)
sim.therm(T=params.T, J=params.J, B=params.B)
sim.sweep(T=Tf, J=Jf, B=Bf, sample_size=sample,
          sweep_step=sweep, tolerance=tol)
sim.save_as(params.sim)
