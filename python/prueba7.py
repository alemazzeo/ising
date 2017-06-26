from ising import Simulation, Ising, State, Sample

import threading
from queue import Queue
import time

sim1 = Simulation.load('../data/simulations/Simulation0.npy')

print_lock = threading.Lock()

q = Queue()

def threader():
    while True:
        worker = q.get()
        sim1.expand(worker, 100)
        j = int(50*(worker+1)/len(sim1))
        text_bar = '*' * j + '-' * (50-j)
        print(text_bar, end='\r')
        q.task_done()

for x in range(8):
    t = threading.Thread(target=threader)
    t.daemon = True
    t.start()

start = time.time()

for worker in range(len(sim1)):
    q.put(worker)

q.join()
