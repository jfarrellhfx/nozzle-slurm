"""
Jack Farrell CU Boulder 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from . import constants

class Nozzle1D():
    """
    1D nozzle model, equations of motion
        - dt(n) + dx(j) = - dx(w)/w
        - dt(j) + dx(j^2/n) + dx(n^2/n2) = eta * dx(dx(j/n)) - gamma * j - j^2/n * dx(w)/w
    """

    def __init__(self, dt=0.0001, dx=1/100, eta=None, gamma=None, R = None, j0 = None, I = None, V0 = 3, lee = None):
        """
        initializations
        """
        # announce starting!
        print("\nStarting 1D simulation, parameters:")

        # constants
        self.C = constants.C
        self.e = constants.e
        self.m = constants.m
        self.hbar = constants.hbar

        # basic parameters
        self.dt = dt
        self.dx =   dx
        self.eta = eta
        self.V0 = V0

        # gate things
        self.n0 = np.abs(self.V0 * self.C / -self.e)
        self.vs = np.sqrt((self.hbar/self.m*np.sqrt(np.pi*self.n0/2))**2 + 0 * (self.n0*self.e**2/self.m/self.C)) # no capacitive correction
        self.vT = np.sqrt(1.38e-23 * 300 / self.m) # thermal speed of sound

        self.VS = np.sqrt(self.vs**2 + self.vT**2)

        # domain
        self.x = np.arange(0, 1, self.dx)
        self.NX = len(self.x)
        
        # channel dimensions in real life
        self.L = 40e-6
        self.W = 0.342 * self.L

        # helper width 
        w = self.w(self.x)

        if I is None and j0 is None:
            raise ValueError("Must specify either j0 or I")
        elif I is None and j0 is not None:
            self.j0 = j0
        elif I is not None and j0 is None:
            self.I = I
            self.j0 = self.get_j(I)
        else:
            raise ValueError("Must specify either j0 or I, not both")

        if gamma is None and R is None:
            raise ValueError("Must specify either gamma or R")
        elif gamma is None and R is not None:
            self.gamma = self.get_gamma(R)
            self.R = R
        elif gamma is not None and R is None:
            self.gamma = gamma
        else:
            raise ValueError("Must specify either gamma or R, not both")
        
        if eta is None and lee is None:
            raise ValueError("Must specify either eta or lee")
        elif eta is None and lee is not None:
            self.eta = self.get_eta(lee)
            self.lee=lee
        elif eta is not None and lee is None:
            self.eta = eta
        else:
            raise ValueError("Must specify either eta or lee, not both")
    
        # initial conditions
        self.iteration = 0
        self.t = 0
        self.q = np.zeros((2, self.NX))
        self.q[0,:] = 1
        self.q[1,:] = -self.j0

        self.start_time = datetime.datetime.now()

        plt.show()

        

    def V(self, q):
        """
        helper function to calculate voltage
        """
        return (q[0,:] - q[0,0]) * self.e/self.C * self.n0


    def save_data(self, results_dir, fname = None):
        """
        save the final state to a npz file
        """

        if fname is None:
            path = f"{results_dir}/j={self.j0:.4f}---eta={self.eta:.4f}---gamma={self.gamma:.4f}---V0={self.V0:.4f}.npz"
        else:
            path = f"{results_dir}/{fname}.npz"

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)

        np.savez(path, x = self.x, q = self.q, vT = np.array([self.vT]), V = self.V(self.q), I = np.array([np.abs(self.j0* self.n0 * self.e * self.W * self.vs * 1000)]))
        pass

        
    def get_j(self, I):
        """
        get j0 from I
        """
        j0 = I / 1000 / self.n0 / self.e / self.vs / self.W
        #print(f"{j0=}")
        return j0
    

    def get_gamma(self, R):
        """
        get gamma from R
        """
        rho = R / 3.82 # geometric prefactor
        tau = self.m / (self.n0*self.e**2) * self.vs / self.L / rho
        gamma = 1/tau
        #print(f"{gamma=}")
        return gamma
    
    
    def get_eta(self, lee):
        """
        get eta from lee
        """
        eta = np.sqrt(2)/4*lee/self.L
        #print(f"{eta=}")
        return eta


    def w(self, x):
        """
        local channel width
        """
        return 0.342 - 2*(0.342-0.094)/2 * np.exp(-(x-1/2)**2*100)
    
    
    @staticmethod
    def minmod(a, b):
        """
        minmod for slope-limiting
        """
        return 1/2*(np.sign(a)+np.sign(b)) * np.minimum(np.abs(a),np.abs(b))
    
    
    def apply_bc(self, q):
        """
        apply boundary conditions
        """
        Q = np.zeros((2, q.shape[1]+4))
        Q[:,2:-2] = q
        Q[0,:2] = 1.0
        Q[1,:2] = q[1,1]
        Q[0,-2:] = q[0,-1]      
        Q[1,-2:] = -self.j0
        return Q
    
    
    def flux(self, dt=None):
        """
        roe flux with high-resolution correction
        """

        # default to global time step
        if dt is None:
            dt = self.dt
        q = self.q

        # apply boundary conditions
        Q = self.apply_bc(q)
        
        # jump in state vectors
        dQ = Q[:,1:] - Q[:,:-1]

        # zeroth order flux
        f = np.zeros_like(Q)
        f[0,:] = Q[1,:]
        f[1,:] = Q[1,:]**2/Q[0,:] + Q[0,:]**2/2 + self.vT**2/self.vs**2 * Q[0,:]
        favr = (f[:,1:] + f[:,:-1])/2

        # roe averaged quantities
        u_raw = Q[1,:]/Q[0,:]
        nsqrt = np.sqrt(Q[0,:])
        u = (nsqrt[1:]*u_raw[1:] + nsqrt[:-1]*u_raw[:-1])/ (nsqrt[1:] + nsqrt[:-1])
        c = np.sqrt((Q[0,1:] + Q[0,:-1])/2 + self.vT**2/self.vs**2)

        # matrices or right eigenvectors
        R = np.zeros((2,2,Q.shape[1]-1))
        R[0,0,:] = 1
        R[1,0,:] = u - c
        R[0,1,:] = 1
        R[1,1,:] = u + c

        # matrices of left eigenvectors
        L = np.zeros((2,2,Q.shape[1]-1))
        L[0,0,:] = 1/2/c*(u+c)
        L[1,0,:] = 1/2/c*(c-u)
        L[0,1,:] = -1/2/c
        L[1,1,:] = 1/2/c

        # vector of eigenvalues
        eigs = np.zeros((2,u.shape[-1]))
        eigs[0,:] = u - c
        eigs[1,:] = u + c

        # diagonal eigs matrix
        eigs_diag = np.zeros((2,2,u.shape[-1]))
        eigs_diag[0,0,:] = eigs[0,:]
        eigs_diag[1,1,:] = eigs[1,:]

        # godunov wave
        wave = np.einsum("ij...,jk...,kl...,l...->i...", R, np.abs(eigs_diag), L, dQ)

        # slope limiter term
        nu = dt / self.dx * eigs
        Ra = np.einsum("ij...,jk...,k...->ij...", R, L, dQ)
        sn = self.minmod(Ra[:,:,1:], Ra[:,:,:-1])
        S = 0.5*eigs*(np.sign(nu)-nu)
        high_res_plus = np.einsum("ij...,j...->i...", sn[:,:,:-1], ((eigs[:,1:-1]>0)*S[:,1:-1]))
        high_res_minus = np.einsum("ij...,j...->i...", sn[:,:,1], ((eigs[:,1:-1]<0)*S[:,1:-1]))

        # assemble the flx
        F = favr[:,1:-1] - 0.5 * wave[:,1:-1] + high_res_plus + high_res_minus

        # add viscosity
        du = Q[1,1:]/Q[0,1:]-Q[1,:-1]/Q[0,:-1]
        F[1,:] += -self.eta / self.dx * du[1:-1]       
        
        return F
    

    def wall_time(self):
        """
        time in cpu hours since start of simulation
        """
        return (datetime.datetime.now() - self.start_time).total_seconds() / 60


    def cons_step(self, dt=None):
        """
        solve the conservation law part
        """

        # default to global time step
        if dt is None:
            dt = self.dt
        
        F = self.flux()
        q = self.q - dt/self.dx * (F[:,1:] - F[:,:-1])
        self.q = q


    def source(self, q):
        """
        RHS of source terms equation
        """

        # calculate channel width
        xbc = np.zeros(self.NX+2)
        xbc[0],xbc[-1] = self.x[0], self.x[-1] 
        xbc[1:-1] = self.x
        w = self.w(xbc)

        # geometric source term from varying width
        geom_part = np.zeros_like(self.q)
        geom_part[0,:] = -(w[2:] - w[:-2])/(2*self.dx) / w[1:-1] * q[1,:]
        geom_part[1,:] = -(w[2:] - w[:-2])/(2*self.dx) / w[1:-1]* q[1,:]**2/q[0,:]

        # momentum relaxation
        relax_part = np.zeros_like(q)
        relax_part[1,:] = -self.gamma * q[1,:]
        
        return geom_part + relax_part

    
    def relax_step(self, dt=None):
        """
        solve source terms
        """

        # default to global time step
        if dt is None:
            dt = self.dt

        # midpoint method
        q = self.q
        q1 = q + dt/2 * self.source(q)
        q = q + dt * self.source(q1)
        self.q = q
        pass

        
    def step(self, dt=None):
        """
        one time step of simulation
        """

        # default to global time step
        if dt is None:
            dt = self.dt

        # update the state
        self.relax_step(self.dt/2)
        self.cons_step(self.dt)
        self.relax_step(self.dt/2)

        # update time and iterations
        self.t += dt
        self.iteration += 1

    
    def live_plot(self):
        """
        draw a plot that updates as the simulation progresses
        """
        # get voltage
        V = (self.q[0,:] - self.q[0,0]) * self.e/self.C * self.n0
        
        # if the figure does not exist, create it
        if not hasattr(self, "fig"):
            self.fig, self.axes = plt.subplots(1,2, figsize = (8,3))
            self.ax1,self.ax2 = self.axes
            self.ax1.plot(self.x, V)
            self.fig.canvas.manager.set_window_title("1D Nozzle Simulation")
            plt.suptitle(f"sim-time = {self.t:1.2f}")
            plt.show(block=False)

        # otherwise, just update the data
        else:
            # update the plot after some amount of iterations
            if self.iteration % 10000 == 0 or self.t == 0:
                
                # plot voltage
                self.ax1.cla()
                self.ax1.plot(self.x, V)
                self.ax1.set_xlabel("$x/L$")
                self.ax1.set_ylabel("$V(x)$ (V)")
                self.ax1.set_title("Voltage profile")

                # plot the mach number
                self.ax2.cla()
                self.ax2.set_xlabel("$x/L$")
                self.ax2.set_title("Mach number")
                self.ax2.set_ylabel("$M(x)$")
                self.ax2.axhline(y = 1, color = "black", zorder = -1, linestyle = "--")
                self.ax2.plot(self.x, np.abs(self.q[1,:]/self.q[0,:]/np.sqrt(self.q[0,:] + self.vT**2/self.vs**2 )))
            
                # configure plot
                plt.tight_layout()
                self.fig.canvas.start_event_loop(0.001)
                self.fig.canvas.draw_idle()

    def log(self):
        """
        log progress
        """
        if self.iteration % 10000 ==0 or self.t == 0:
            print(f"wall_time = {datetime.datetime.now() - self.start_time}, simtime = {self.t:1.4f}")
        pass
