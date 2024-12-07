"""
Jack Farrell, CU Boulder, 2024

2D channel flow of viscous electron gas through a nozzle using finite volume methods: Roe's approximate Riemann solver for 2D shallow water.  Dimensional splitting for x,y evolution. Extra resolution in x direction with minmod slope-limiter method
"""

########## imports
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

from .flux import FX, FY
from .bcs import apply_bc_x, apply_bc_y
from .utility import make_plot
from .config import *

########## functions
def save_data(q, n1,n2,eta,gamma,results_dir, x, y):
    
    # make sure the directory exists and create it if not
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # create a filename based on the parameters
    filename = f"n1={n1:.4f}-n2={n2:.4f}-eta={eta:.4f}-gamma={gamma:.4f}.npz"

    # save the data
    np.savez(
        f"{results_dir}/{filename}",
        q = q,
        x = x,
        y = y,
    )
    return "saved"





########### main solve function
def solve(n1=n1, n2=n2, results_dir=results_dir, eta=eta, gamma=gamma, draw_plot=draw_plot, stop_wall_time=stop_wall_time, save_after=save_after, save_increment=save_increment, h=h, hy=hy, k=k):

    # domain
    x = np.arange(0,LX,h)
    y = np.arange(0,LY,hy)
    Y,X = np.meshgrid(y,x)
    NX = len(x)
    NY = len(y)

    # constriction shape # TODO wrap in func
    mask = np.zeros_like(X)
    mask[Y < (0.342-0.094)/2 * np.exp(-(X-1/2)**2*100)] = 1
    mask[:,-int(NY/2):] = np.flip(mask[:,:int(NY/2)],1)
    width = 0.342 - 2*(0.342-0.094)/2 * np.exp(-(x-1/2)**2*100)


    # initialize the fields
    q = np.zeros((3, NX, NY))
    q[0, :, :] = n1 + (1 + np.tanh((X - 0.5) * 10)) / 2 * (n2 - n1)
    q[1, :, :] = 0.0
    q[2, :, :] = 0.0


    # a few helper functions
    def x_step(q,dt,h):
        """
        integrate conservation law in x direction
        """
        Q = apply_bc_x(q,NX,NY,mask,n1,n2)
        F = FX(Q,k,h,eta)
        q = Q[:,2:-2,:] - dt/h*(F[:,1:,:]-F[:,:-1,:])
        return q

    def y_step(q,dt,h):
        """
        integrate conservation law in y direction
        """
        Q = apply_bc_y(q,NX,NY,mask)
        F = FY(Q,k,hy,eta)
        q = Q[:,:,2:-2] - dt/h*(F[:,:,1:]-F[:,:,:-1])
        return q

    def relax_step(q,dt):
        """
        integrate the momentum relaxation term
        """
        q1 = np.copy(q)
        q1[1,:,:] = q[1,:,:] * np.exp(-gamma *dt)
        q1[2,:,:] = q[2,:,:] * np.exp(-gamma * dt)
        return q1


    # create a storage array to save the data
    storage = np.zeros((1, 3, NX, NY))

    # initialize step counter and simulation time and wall_time
    index = 0
    simtime = 0
    wall_time = 0

    # calculate the time in seconds to run for
    stop_wall_time_seconds = stop_wall_time * 3600

    # get the current time before sim starts
    start_time = datetime.datetime.now()

    # loop!
    while wall_time < stop_wall_time_seconds:

        # keep track of time in seconds
        wall_time = (datetime.datetime.now() - start_time).total_seconds()

        # use nested strang splitting to perform one full timestep
        q = relax_step(q, k / 2)
        q = y_step(q, k/2, hy)
        q = x_step(q, k, h)
        q = y_step(q, k / 2, hy)
        q = relax_step(q, k / 2)

        # once the save_increment is reached, 
        if (simtime + k) // save_increment != simtime // save_increment or simtime == 0:
            if wall_time > save_after * 3600:
                storage = np.concatenate([storage, np.expand_dims(q, 0)], axis=0)

            # log the time to console
            print(f"time elapsed {datetime.datetime.now() - start_time} ; simtime = {simtime:.4f}")

            # draw the plot if desired
            if draw_plot:
                make_plot(q, simtime, x, y, X, Y, mask)

        # iterate
        simtime += k
        index += 1

    # finally, save the data
    save_data(np.array(storage), n1, n2, eta, gamma, results_dir, x, y)

if __name__ == "__main__":
    solve()
