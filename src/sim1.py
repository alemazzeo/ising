import numpy as np
from ising import Simulation, State
from analysis import Analysis, Result

ising1 = State(32)

ising1.T = 4.0
ising1.J = 0.6
ising1.B = 0.0

