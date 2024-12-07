"""
Jack Farrell, CU Boulder 2024
"""

########## IMPORTS ##########
import matplotlib.pyplot as plt
import numpy as np
import logging
import dedalus.public as de
import scipy as sp
import argparse, sys
import os
from os.path import abspath, dirname
os.chdir(dirname(abspath(__file__)))
from dedalus.tools.config import config
config['logging']['stdout_level'] = 'info'

########## CONSTANTS ##########
# simulation parameters
LX = 1
LY = 0.342
NX = 64
NY = 64
k = 0.0001
STOP_TIME = 100
TIMESTEPPER = de.SBDF1
SNAPSHOT_INTERVAL = 0.01
S = 10000
NU = 0.1
DTYPE = np.float64
DEALIAS = 3/2
geometric_prefactor = 8
anim = 0
# constants
e = 1.602e-19
hbar = 1.0456e-34
m = 9.11e-31 * 0.03

L = 40e-6
W = 0.342 * L

VDa = -5
VG = -8

d = 307e-9
e0 = 8.854e-12
er = 3.9
C = e0*er/d

# command-line overridable values
I = 4
R = 3000 
rho = R *  W / geometric_prefactor
results_dir = f"I={I:.3f}_c".replace(".","_")


n0 = 0.3e11 * 1e4

#n0 = 5e11 * 1e4


########## FUNCTIONS ##########
def smooth(A, n):
    A = sp.ndimage.filters.gaussian_filter(A, [0,n], mode = 'constant', cval = 1)
    A = sp.ndimage.filters.gaussian_filter(A, [n,0], mode = 'nearest', cval = 1)
    return A

def nozzle(X, Y, LX, LY):
    h = 0.098
    height = LY/2 - (2 * LY/2-h) / 2 * np.exp(-1/2*X**2/(0.427/6)**2)
    Gamma = np.zeros((X.shape[0], Y.shape[1]))
    Gamma[Y > height] = 1
    Gamma[Y <= -height] = 1
    #Gamma[:,0] = 1
    Gamma[:,Y.shape[1]//2:] = np.flip(Gamma[:,0:Y.shape[1]//2], axis = 1)
    return Gamma


if __name__ == "__main__":
    ends = []


    ########## COMMAND-LINE INTERFACE #########
    parser = argparse.ArgumentParser(prog='constrictionFlow', description='hydrodynamic electron flow through a de-Laval nozzle.')   
    parser.add_argument("--I", help="current (mA)", type=float, default=I)
    parser.add_argument("--n0", help="uniform background number density (cm^-2)", type=float, default=n0)
    parser.add_argument("--m", help="effective mass (kg)", type=float, default=m)
    parser.add_argument("--R", help="resistance (Ohms)", type=float, default=R)
    parser.add_argument("--results-dir", help="name of subdirectory for datasets")
    args = parser.parse_args()
    I, n0, m, R = float(args.I)/100, args.n0, args.m, args.R
    if args.results_dir != None:
        results_dir = str(args.results_dir)


    ########## PRELIMINARY CALCULATIONS ##########
    # Experimental Parameters
    vs = np.sqrt(e**2/m/C*n0 + n0 * hbar**2 * np.pi*n0/2/m)
    J0 = I/W/e / (n0 * vs) * 1e-3

    #J0 = 0.01
    tau = m/(n0*e**2) * vs / rho
    gamma = 1/tau 
        
    
    ########## SIMULATION SETUP ##########
    coords = de.CartesianCoordinates('x', 'y')
    dist = de.Distributor(coords, dtype=DTYPE)
    xbasis = de.Chebyshev(coords['x'], size=NX, bounds=(-LX/2,LX/2), dealias=DEALIAS)
    ybasis = de.RealFourier(coords['y'], size=NY, bounds=(-LY/2,LY/2), dealias=DEALIAS)
    X, Y = dist.local_grid(xbasis), dist.local_grid(ybasis)
    x,y = np.meshgrid(X,Y)

    n = dist.Field(name = 'n', bases = (xbasis, ybasis))
    u = dist.Field(name = 'u', bases = (xbasis, ybasis))
    v = dist.Field(name = 'v', bases = (xbasis, ybasis))
    c = dist.Field(name = 'c', bases = (xbasis, ybasis)) # passive tracer
    ux = dist.Field(name = 'ux', bases = (xbasis, ybasis))
    vx = dist.Field(name = 'vx', bases = (xbasis, ybasis))
    tauu = dist.Field(name = 'tauu', bases = ybasis)
    taun = dist.Field(name = 'tauu', bases = ybasis)
    tauv = dist.Field(name = 'tauv', bases = ybasis)
    tau3 = dist.Field(name = 'tau3', bases = ybasis)
    tau4 = dist.Field(name = 'tau4', bases = ybasis)
    tauc = dist.Field(name='tauc', bases = ybasis)

    f = dist.Field(name='f',  bases = (xbasis,ybasis))

  
    # tau method
    lift_basis = xbasis.derivative_basis(1)
    p = dist.Field(name = "p", bases = lift_basis)
    p['c'][-1] = 1

    # substitutions
    dx = lambda A: de.Differentiate(A, coords['x'])
    dy = lambda A: de.Differentiate(A, coords['y'])

    # geometry
    geom = dist.Field(name="geom", bases = (xbasis, ybasis))
    geom["g"] = nozzle(X, Y, LX, LY)
    geom["g"] = smooth(geom["g"], 2)




    f["g"][0,::NY//5] = 1  #np.sin(np.array(range(NY))/7)**2
    f['g'] = sp.ndimage.filters.gaussian_filter(f['g'], [2,2], mode='constant', cval = 0)

    #plt.pcolormesh(geom["g"])
    #plt.show()

    problem = de.IVP([n,u,c,v,ux,vx, tauu, tauv, tau3,tau4,tauc], namespace = locals())

    # equations 
    problem.add_equation("dt(n) + ux + dy(v)= 0")
    problem.add_equation("dt(u)+ tauu * p + gamma*u - NU*(dx(ux) + dy(dy(u)))= -dx(n**2)/2 - dx(u**2/n) - dy(u*v/n) - S*geom*u")
    problem.add_equation("dt(v) + tauv * p + gamma*v - NU*(dx(vx) + dy(dy(v)))= -dy(n**2)/2 - dx(u*v/n) - dy(v**2/n) - S*geom*v")
    problem.add_equation("dt(c) + tauc*p = -dx(c*u/n) - dy(c*v/n) - 0.1*c")
    problem.add_equation("ux - dx(u) + tau3 * p = 0")
    problem.add_equation("vx - dx(v) + tau4 * p = 0")

    # boundary conditions
    problem.add_equation("v(x =-LX/2) = 0")
    problem.add_equation("v(x = LX/2) = 0")
    problem.add_equation("u(x = -LX/2) = J0")
    problem.add_equation("ux(x=LX/2)=0")
    problem.add_equation("c(x=-LX/2) = f(x=-LX/2)")


    solver = problem.build_solver(TIMESTEPPER)
    solver.stop_wall_time = 32400


    # save data
    snapshots = solver.evaluator.add_file_handler(results_dir, sim_dt=SNAPSHOT_INTERVAL, max_writes=1e10)
    snapshots.add_task(n, name="density")
    snapshots.add_task(u, name="momentum")
    snapshots.add_task(c, name="tracer")

    # initial conditions
    n['g'] = 1
    u['g'] = 0
    v['g'] = 0
    c['g']= f['g']
    #plt.plot(c['g'][0,:])
    #plt.show()
   


    ########## SIMULATION LOOP ##########
    logger = logging.getLogger(__name__)
    print(f"\nSimulation Parameters:\ngamma = {gamma}\nJ0 = {J0}\n\nn0 = {n0/1e15}")
    Gamma = nozzle(X, Y, LX, LY)
    Gamma[Gamma==0.0] = np.nan


    
    if anim:
        plt.show(block=False)
        fig, axes = plt.subplots(2,2, figsize = (10, 8))
        ax1,ax2 = axes[0]
        ax3, ax4 = axes[1]
    try:
        while solver.proceed:
            timestep = k
            solver.step(timestep)
            if (solver.iteration-1)%5000 == 0:
                logger.info('t={},  Iteration={}'.format(solver.sim_time, solver.iteration))
               
                if anim:
                    # figure
                    n_ = n
                    u_ = u
                    c_ = c
                    c_.change_scales(1)
                    n_.change_scales(1)
                    u_.change_scales(1)
                    ax1.cla()
                    ax2.cla()
                    ax3.cla()
                    ax4.cla()
                    ax1.set_ylabel("$U(x)$")
                    ax2.set_ylabel("$u(x)/v_s$")
                    ax1.set_xlabel("$x$")
                    ax2.set_xlabel("x")
                

               
                    V = (n_['g']-1) * n0 * e /C
                

                    veloc = u_['g']/n_['g']
                    ends.append(np.max(veloc))


                    ax1.plot(X, V[:,32])
                    #ax2.plot(ends)
                    ax2.pcolormesh(x,y, veloc.T, cmap = "bwr", vmin = -1, vmax = 1)
                    ax3.pcolormesh(x,y,c_['g'].T, cmap = "Blues", vmin = 0, vmax = np.max(c['g']))
                    ax3.pcolormesh(x,y,Gamma.T, cmap = "Greys", vmin = 0, vmax = 1)
                    ax4.plot(ends)
                    #plt.ylabel("$U(x) (V) $")
                    #plt.tight_layout()
                    #veloc = ((u.set_scales(1)/n.set_scales(1))["g"])
                    #plt.plot(X, veloc['g'][c.shape[1] //2])
                    fig.canvas.start_event_loop(0.001)
                    fig.canvas.draw_idle()
                    
                
                
            
    except:
        logger.error("Ending loop")
        np.savez(results_dir+"/parameters.npz", X, Y, k)
        raise
    finally:
        solver.log_stats()
        np.savez(results_dir+"/parameters.npz", X, Y, k)
        #plt.plot(X[:,:], n['g'][:,64][:128])
        #plt.show()
