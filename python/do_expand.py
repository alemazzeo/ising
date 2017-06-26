from ising import Simulation

import threading
from queue import Queue
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-sim', type=str,
                    default='../data/simulations/Simulation2.npy')
parser.add_argument('-threads', type=int, default=8)
parser.add_argument('-expand', type=int, default=100)

params = parser.parse_args()

sim1 = Simulation.load(params.sim)

print_lock = threading.Lock()

q = Queue()

def threader():
    while True:
        worker = q.get()
        sim1.expand(worker, params.expand)
        j = int(50*(worker+1)/len(sim1))
        text_bar = '*' * j + '-' * (50-j)
        print(text_bar, end='\r')
        q.task_done()

for x in range(params.threads):
    t = threading.Thread(target=threader)
    t.daemon = True
    t.start()

start = time.time()

for worker in range(len(sim1)):
    q.put(worker)

q.join()
