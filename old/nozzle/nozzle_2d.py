"""
Jack Farrel CU Boulder 2024
2D nozzle model using finite-volume methods with a simple dimensional-splitting algorithm.  We use Roe's approximate Riemann solver for the shallow-water equations with minmod slope limiting to make a high-resolution method.  Relaxation terms are incorporated in an additional strang-splitting step.
"""
# imports
import numpy as np
import matplotlib.pyplot as plt
from . import constants
import datetime

# Nozzle2D class
class Nozzle2D():
    """
    2D nozzle model
    """

    def __init__(self, dt, dx, dy, eta, gamma, R, I, j0, V0, lee):
        """
        initializations
        """
        # announcements
        print("\nStarting 2D simulation, parameters:")
        
        # constants
        self.C = constants.C
        self.e = constants.e
        self.m = constants.m
        self.hbar = constants.hbar

        # basic parameters
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.eta = eta
        self.V0 = V0

        # gate things
        self.n0 = np.abs(self.V0 * self.C / -self.e)
        self.vs = np.sqrt((self.hbar/self.m*np.sqrt(np.pi*self.n0/2))**2 + (self.n0*self.e**2/self.m/self.C))
        
        # domain
        self.x = np.arange(0, 1, self.dx)
        self.NX = len(self.x)
        self.y = np.arange(0, 0.342, self.dy)
        self.NY = len(self.y)
        self.Y,self.X = np.meshgrid(self.y,self.x)

        
        # channel dimensions in real life
        self.L = 40e-6
        self.W = 0.342 * self.L

        # channel geometry
        self.mask = np.zeros_like(self.X)
        self.mask[self.Y < (0.342-0.094)/2 * np.exp(-(self.X-1/2)**2*100)] = 1
        self.mask[:,-int(self.NY/2):] = np.flip(mask[:,:int(self.NY/2)],1)


        # should only specify one of I or j0
        if I is None and j0 is None:
            raise ValueError("Must specify either j0 or I")
        elif I is None and j0 is not None:
            self.j0 = j0
        elif I is not None and j0 is None:
            self.I = I
            self.j0 = self.get_j(I)
        else:
            raise ValueError("Must specify either j0 or I, not both")

        # should only specify one of gamma or R
        if gamma is None and R is None:
            raise ValueError("Must specify either gamma or R")
        elif gamma is None and R is not None:
            self.gamma = self.get_gamma(R)
            self.R = R
        elif gamma is not None and R is None:
            self.gamma = gamma
        else:
            raise ValueError("Must specify either gamma or R, not both")
        
        # should only specify one of eta or lee
        if eta is None and lee is None:
            raise ValueError("Must specify either eta or lee")
        elif eta is None and lee is not None:
            self.eta = self.get_eta(lee)
            self.lee=lee
        elif eta is not None and lee is None:
            self.eta = eta
        else:
            raise ValueError("Must specify either eta or lee, not both")
        
        # geometry
        self.mask = np.zeros_like(self.X)
        self.mask[self.Y < (0.342-0.094)/2 * np.exp(-(self.X-1/2)**2*100)] = 1
        self.mask[:,-int(self.NY/2):] = np.flip(self.mask[:,:int(NY/2)],1)
        self.width = 0.342 - 2*(0.342-0.094)/2 * np.exp(-(self.x-1/2)**2*100)
        
        # initial conditions
        self.q = np.zeros((2, self.NX))
        self.q[0,:] = 1
        self.q[1,:] = -self.j0
        self.q[2,:] = 0

        # other initializations
        self.start_time = datetime.datetime.now()
        self.iteration = 0
        self.t = 0

        pass


    def V(self, q):
        """
        helper function to calculate voltage
        """
        return (q[0,:] - q[0,0]) * self.e/self.C * self.n0
    

    def save_data(self, results_dir, fname = None):
        """
        save the state and some other useful things to a npz file
        """

        # default filename

        #if fname is None:
        # TODO finish
        pass


    @staticmethod
    def minmod(a, b):
        """
        minmod for slope-limiting
        """
        return 1/2*(np.sign(a)+np.sign(b)) * np.minimum(np.abs(a),np.abs(b))
    

    def apply_bc_x(self, q):
        """
        apply boundary conditions in x direction
        """
        # initialize bigger array with ghost cells to implement boundary conditions
        qbc = np.zeros((3,self.NX+4,self.NY))
        q1 = np.copy(q)
        q1[1,:,:][self.mask == 1] = 0
        q1[2,:,:][self.mask == 1] = 0
        qbc[:,2:-2,:] = q1[:,:,:]

        # default zeroth order extrapolation
        qbc[:,0,:] = q1[:,0,:]
        qbc[:,-1,:] = q1[:,-1,:]

        # density boundary conditions
        qbc[0,0,:] = 1.0 # left
        qbc[0,-1,:] = q1[0,-1,:] # right

        # current boundary conditions
        qbc[1,0,:] = q1[1,0,:] # left
        qbc[1,-1,:] = self.j0 # right
        
        # transverse current boundary conditions
        qbc[2,0,:] = 0 # left
        qbc[2,-1,:] = 0 # right

        # fill in the rest of the boundary
        qbc[:,-2,:] = np.copy(qbc[:,-1,:])
        qbc[:,1,:] = np.copy(qbc[:,0,:])
        return qbc
    
    
    def apply_bc_y(self, q):
        """
        boundary conditions in y direction
        """
        # build the boundary condition array
        qbc = np.zeros((3,self.NX,self.NY+4))
        q1 = np.copy(q)
        q1[1,:,:][self.mask == 1] = 0
        q1[2,:,:][self.mask == 1] = 0
        qbc[:,:,2:-2] = q1[:,:,:]

        # default zeroth order extrapolation
        qbc[:,:,0] = q1[:,:,0]
        qbc[:,:,-1] = q1[:,:,-1]

        # density boundary conditions
        qbc[1,:,0] = 0 # 
        qbc[1,:,-1] = 0
        qbc[2,:,0] = 0
        qbc[2,:,-1] = 0

        # fill in the rest of the boundary
        qbc[:,:,1] = np.copy(qbc[:,:,0])
        qbc[:,:,-2] = np.copy(qbc[:,:,-1])
        return qbc
    
    
    def FX(self,q, dt=None):
        """
        finite volume flux across horizontal edges of grid points
        """
        if dt is None:
            dt = self.dt

        Q = self.apply_bc_x(q)
        dQ = Q[:,1:,:]-Q[:,:-1,:]
        f = np.zeros_like(Q)
        f[0, :, :] = Q[1, :, :]
        f[1, :, :] = Q[1, :, :]**2 / Q[0, :, :] + 0.5 * Q[0, :, :]**2
        f[2, :, :] = Q[1, :, :] * Q[2, :, :] / Q[0, :, :]
        favr = 0.5*(f[:,:-1,:] + f[:,1:,:])
        usqrt = Q[1,:,:]/Q[0,:,:]
        vsqrt = Q[2,:,:]/Q[0,:,:]
        nsqrt = np.sqrt(Q[0,:,:])
        uavr = (usqrt[:-1,:]*nsqrt[:-1,:] + usqrt[1:,:]*nsqrt[1:,:])/(nsqrt[:-1,:]+nsqrt[1:,:])
        vavr = (vsqrt[:-1,:]*nsqrt[:-1,:] + vsqrt[1:,:]*nsqrt[1:,:])/(nsqrt[:-1,:]+nsqrt[1:,:])
        cavr =np.sqrt((Q[0,:-1,:]+Q[0,1:,:])/2)

        # calculate the value of L(u) and R(u) for each cell
        u, v, c = uavr, vavr, cavr
        L_values = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))
        R_values = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))

        L_values[0, 0, :, :] = (u + c) / (2 * c)
        L_values[1, 0, :, :] = -v
        L_values[2, 0, :, :] = (c-u)/(2*c)
        L_values[0, 1, :, :] = -1/(2*c)
        L_values[1, 1, :, :] = 0
        L_values[2, 1, :, :] = 1/(2*c)
        L_values[0, 2, :, :] = 0
        L_values[1, 2, :, :] = 1
        L_values[2, 2, :, :] = 0

        R_values[0, 0, :, :] = 1
        R_values[1, 0, :, :] = u-c
        R_values[2, 0, :, :] = v
        R_values[0, 1, :, :] = 0
        R_values[1, 1, :, :] = 0
        R_values[2, 1, :, :] = 1
        R_values[0, 2, :, :] = 1
        R_values[1, 2, :, :] = u+c
        R_values[2, 2, :, :] = v

        u, v, c = uavr, vavr, cavr

        # calculate the fluxes
        # build an array of the eigenvalues for each cell
        eigs = np.zeros((3, uavr.shape[0], uavr.shape[1]))
        eigs[0, :, :] = uavr - cavr
        eigs[1, :, :] = uavr
        eigs[2, :, :] = uavr + cavr

        # Create a diagonal matrix out of the vector of the first index of eigs
        eigs_diag = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))
        eigs_diag[0, 0, :, :] = eigs[0, :, :]
        eigs_diag[1, 1, :, :] = eigs[1, :, :]
        eigs_diag[2, 2, :, :] = eigs[2, :, :]

        # build an array of nu values for each cell
        nu = np.zeros((3, uavr.shape[0], uavr.shape[1]))
        nu = dt / self.dx * eigs
        
        # build an array of Aabs values for each cell
        #Aabs = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))
        Aabs = np.einsum('imkl,mjkl,jnkl->inkl', R_values, np.abs(eigs_diag), L_values)

        alpha = np.einsum('ijkl,jkl->ikl', L_values, dQ)
        alpha_diag = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))
        alpha_diag[0, 0, :, :] = alpha[0, :, :]
        alpha_diag[1, 1, :, :] = alpha[1, :, :]
        alpha_diag[2, 2, :, :] = alpha[2, :, :]
        Ra = np.einsum("ij...,jk...->ik...", R_values, alpha_diag)
        sn = self.minmod(Ra[:,:,1:,:], Ra[:,:,:-1,:])
        S = 0.5*eigs*(np.sign(nu)-nu)
        high_res_plus = np.einsum("ijkl,jkl->ikl", sn[:,:,:-1,:], ((eigs[:,1:-1,:]>0)*S[:,1:-1,:]))
        high_res_minus = np.einsum("ijkl,jkl->ikl", sn[:,:,1:,:], ((eigs[:,1:-1,:]<0)*S[:,1:-1,:]))

        F = favr[:,1:-1,:] - 0.5 * np.einsum('ijkl,jkl->ikl', Aabs[:,:,1:-1,:], dQ[:,1:-1,:]) + high_res_plus + high_res_minus

        du = Q[1,1:,:]/Q[0,1:,:]-Q[1,:-1,:]/Q[0,:-1,:]     
        dv = Q[2,1:,:]/Q[0,1:,:]-Q[2,:-1,:]/Q[0,:-1,:]    
        F[1,:,:] += -self.eta / self.dx * du[1:-1,:]
        F[2,:,:] += -self.eta / self.dx * dv[1:-1,:]
        return F




def FY(self, q, dt=None):
    """
    finite volume flux in y-direction
    """
    if dt is None:
        dt = self.dt

    # apply boundary conditions
    Q = self.apply_bc_y(q)
    
    # construct zeroth-order fluxes
    f = np.zeros_like(Q)
    f[0, :, :] = Q[2, :, :]
    f[2, :, :] = Q[2, :, :]**2 / Q[0, :, :] + 1/2 * (Q[0, :, :])**2
    f[1, :, :] = Q[1, :, :] * Q[2, :, :] / Q[0, :, :]

    # average flux
    favr = 0.5*(f[:,:,:-1] + f[:,:,1:])

    # jump in the state vector for solving riemann problem
    dQ = Q[:,:,1:]-Q[:,:,:-1]

    # we'll just vectorize the calculations from fy
    usqrt = Q[1,:,:]/Q[0,:,:]
    vsqrt = Q[2,:,:]/Q[0,:,:]    
    nsqrt = np.sqrt(Q[0,:,:])
    uavr = (usqrt[:,:-1]*nsqrt[:,:-1] + usqrt[:,1:]*nsqrt[:,1:])/(nsqrt[:,:-1]+nsqrt[:,1:])
    vavr = (vsqrt[:,:-1]*nsqrt[:,:-1] + vsqrt[:,1:]*nsqrt[:,1:])/(nsqrt[:,:-1]+nsqrt[:,1:])
    cavr =np.sqrt(1/2*(Q[0,:,:-1]+Q[0,:,1:]))
    u, v, c = uavr, vavr, cavr

    # build values R_values, using r1, r2,r3 from f2 as the columns of the R matrix
    R_values = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))
    R_values[0, 0, :, :] = 1
    R_values[1, 0, :, :] = u
    R_values[2, 0, :, :] = v-c
    R_values[0, 1, :, :] = 0
    R_values[1, 1, :, :] = -1
    R_values[2, 1, :, :] = 0
    R_values[0, 2, :, :] = 1
    R_values[1, 2, :, :] = u
    R_values[2, 2, :, :] = v+c
    
    # build L values. L matrices should be matrix inverse of R matrices
    L_values = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))
    L_values[0, 0, :, :] = (v + c) / (2 * c)
    L_values[1, 0, :, :] = u
    L_values[2, 0, :, :] = (c - v) / (2 * c)
    L_values[0, 1, :, :] = 0
    L_values[1, 1, :, :] = -1
    L_values[2, 1, :, :] = 0
    L_values[0, 2, :, :] = -1 / (2 * c)
    L_values[1, 2, :, :] = 0
    L_values[2, 2, :, :] = 1 / (2 * c)

    # build eigs and eigs_diag, using w1, w2, w3 from fy
    eigs = np.zeros((3, uavr.shape[0], uavr.shape[1]))
    eigs[0, :, :] = v - c
    eigs[1, :, :] = v 
    eigs[2, :, :] = v + c

    # duplicate eigs vector in diagonal form
    eigs_diag = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))
    eigs_diag[0, 0, :, :] = eigs[0, :, :]
    eigs_diag[1, 1, :, :] = eigs[1, :, :]
    eigs_diag[2, 2, :, :] = eigs[2, :, :]

    # build an array of nu values for each cell
    nu = np.zeros((3, uavr.shape[0], uavr.shape[1]))
    nu = dt / self.dy * eigs
    
    # build an array of Aabs values for each cell
    Aabs = np.einsum('imkl,mjkl,jnkl->inkl', R_values, np.abs(eigs_diag), L_values)

    # coefficients 
    alpha = np.einsum('ijkl,jkl->ikl', L_values, dQ)
    alpha_diag = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))
    alpha_diag[0, 0, :, :] = alpha[0, :, :]
    alpha_diag[1, 1, :, :] = alpha[1, :, :]
    alpha_diag[2, 2, :, :] = alpha[2, :, :]

    # calculate R*alpha
    Ra = np.einsum("ij...,jk...->ik...", R_values, alpha_diag)

    # minmod slopes
    sn = self.minmod(Ra[:,:,:,1:], Ra[:,:,:,:-1])

    # calculate high resolution fluxes
    S = 0.5*eigs*(np.sign(nu)-nu)
    high_res_plus = np.einsum("ijkl,jkl->ikl", sn[:,:,:,:-1], ((eigs[:,:,1:-1]>0)*S[:,:,1:-1]))
    high_res_minus = np.einsum("ijkl,jkl->ikl", sn[:,:,:,1:], ((eigs[:,:,1:-1]<0)*S[:,:,1:-1]))

    # assemble and calculate the flux
    Flux = np.zeros_like(Q)
    Flux = favr[:,:,1:-1]-0.5 * np.einsum('ijkl,jkl->ikl', Aabs[:,:,:,1:-1], dQ[...,1:-1]) + high_res_plus + high_res_minus

    # add viscous terms
    du = Q[1,:,1:]/Q[0,:,1:]-Q[1,:,:-1]/Q[0,:,:-1]
    dv = Q[2,:,1:]/Q[0,:,1:]-Q[2,:,:-1]/Q[0,:,:-1]
    Flux[1,:,:] += -self.eta/self.dy*du[:,1:-1]
    Flux[2,:,:] += -self.eta/self.dy*dv[:,1:-1]
    return Flux


def x_step(self, q, dt=None):
    """
    x-direction step
    """
    if dt is None:
        dt = self.dt
    # calculate fluxes
    F = self.FX(self.q, dt)
    
    # update q
    q = q - dt/self.dx*(F[:,1:,:]-F[:,:-1,:])
    return q


def y_step(self, q, dt=None):
    """
    y-direction step
    """
    # set default timestep if it is not supplied
    if dt is None:
        dt = self.dt
        
    # calculate fluxes
    F = self.FY(self.q, dt)
    
    # update q
    q = q - dt/self.dy*(F[:,:,1:]-F[:,:,:-1])
    return q

def split_step(self, dt = None):
    """
    dimensional splitting step
    """
    # set default timestep if it is not supplied
    if dt is None:
        dt = self.dt
        
    
    self.q = self.y_step(self.q, dt)
    self.q = self.x_step(self.q, dt)
    self.q = self.y_step(self.q, dt)
    
    # update time
    self.t += dt
    self.iteration += 1
    pass


def relax_step(self, dt=None):
    # set default timestep if it is not supplied
    if dt is None:
        dt = self.dt

    self.q[1:,...] = np.exp(-dt*self.gamma) * self.q[1:,...]

    pass



