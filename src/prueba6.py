from ising import Simulation, Ising, State

sim = Simulation(32)          
sim.therm(T=4.0, J=0.6, B=0.0)                       
sim.sweep(T=0.1, sample_size=1000, sweep_step=-0.01) 
sim.save_as('Simulation')

sim = Simulation(32)
sim.therm(T=4.0, J=0.4, B=0.0)
sim.sweep(T=0.1, sample_size=1000, sweep_step=-0.01)
sim.save_as('Simulation')

sim = Simulation(32)
sim.therm(T=4.0, J=0.2, B=0.0)
sim.sweep(T=0.1, sample_size=1000, sweep_step=-0.01)
sim.save_as('Simulation')

sim = Simulation(32)
sim.therm(T=4.0, J=0.0, B=0.1)
sim.sweep(T=0.1, sample_size=1000, sweep_step=-0.01)
sim.save_as('Simulation')

sim = Simulation(32)
sim.therm(T=4.0, J=0.2, B=0.5)
sim.sweep(T=0.1, sample_size=1000, sweep_step=-0.01)
sim.save_as('Simulation')

sim = Simulation(32)
sim.therm(T=4.0, J=0.2, B=1.0)
sim.sweep(T=0.1, sample_size=1000, sweep_step=-0.01)
sim.save_as('Simulation')

sim = Simulation(32)
sim.therm(T=4.0, J=0.2, B=-0.5)
sim.sweep(T=0.1, sample_size=1000, sweep_step=-0.01)
sim.save_as('Simulation')
