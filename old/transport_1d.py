"""
Jack Farrell CU Boulder 2024

Current sweep at FIXED RESISTANCE (doesn't seem to be correct due to heating) in 1D nozzle model
Should take 20 mins to run
"""
import numpy as np
import matplotlib.pyplot as plt
from nozzle import Nozzle1D


Is = np.linspace(0.0,4,100) # current (mA)
V0 = -2.0 # being charitable, maybe we are only 2V away from CNP?
lee = 1e-7 # electron-electron collision mean free path (related to viscosity) (m)
Rs = 300 + Is**2 * 150# plugging in quadratic dependencce by hand


# sweep over currents
for n, I in enumerate(Is):

    # initialize nozzle solver 
    solver = Nozzle1D(
        dt = 0.001, 
        dx = 1/100, # pretty coarse grid
        R = Rs[n],
        I = I, 
        V0 = V0, 
        lee = lee
        )
    
    # set initial condition to be the previous solution
    if n > 0:
        solver.q = q

    # run the simulation
    while solver.wall_time() < 1/5: # for 1 / 4 a a minute;
        solver.step() # update simulation

    # grab the last state
    q = solver.q
    
    # log sweet progress
    print(f"completed current {I = }")

    # save the results in 1D_example_results
    solver.save_data(f"transport_1D_temperature/try4", fname = f"current_{I:.3f}")