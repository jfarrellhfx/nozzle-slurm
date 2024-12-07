"""
Jack Farrell, CU Boulder, 2024

2D channel flow of viscous electron gas through a nozzle using finite volume methods: Roe's approximate Riemann solver for 2D shallow water.  Dimensional splitting for x,y evolution. High resolution in x direction with minmod slope limiter.

This self-contained script contains all necessary configuration options and functions...
"""

########## imports
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import datetime


########## configuration 
# setup, unlikely to change
LX, LY = 1.0, 0.342

# discretizations
h = LX/100
hy = LY/100
k = 0.0001

# how long to simulate
stop_wall_time = 14 # cpu-hours to run for

# saving data
results_dir = "/scratch/alpine/jafa3629/constriction/constriction_good_copy"
save_increment = 0.01 # simtime between snapshots
save_after = 13 # how long to wait before starting to save data

# physical parameters
eta = 0.0025 # viscosity
gamma = 3 # momentum relaxation rate

# boundary conditions
n1 = 3 # left
n2 = 1 # right

# command-line interface
# optionally specify any or all of n1, n2, eta, gamma, results_dir as command-line arguments
# when running in batches using shell scripts, it's easiest to pass integer values.  So divide by 10000 to have 0.01 accuracy. (e.g. if passed --n1=10000, n1 will be 1.000)
parser = argparse.ArgumentParser(prog='constrictionFlow', description='hydrodynamic electron flow through a de-Laval nozzle.')   
parser.add_argument("--n1", type=float, default=10000*n1)
parser.add_argument("--n2", type=float, default=10000*n2)
parser.add_argument("--results-dir", default=results_dir)
parser.add_argument("--eta", type=float, default=10000*eta)
parser.add_argument("--gamma", type=float, default=10000*gamma)
args = parser.parse_args()
n1, n2, results_dir, eta, gamma = args.n1/10000, args.n2/10000, args.results_dir, args.eta/10000, args.gamma/10000


# make sure the directory exists and create it if not
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
# format filename in readable way
filename = f"n1={n1:.4f}-eta={eta:.4f}-gamma={gamma:.4f}.npz"

# domain
x = np.arange(0,LX,h)
y = np.arange(0,LY,hy)
Y,X = np.meshgrid(y,x)
NX = len(x)
NY = len(y)

# constiction shape
mask = np.zeros_like(X)
mask[Y < (0.342-0.094)/2 * np.exp(-(X-1/2)**2*100)] = 1
mask[:,NY//2:] = np.flip(mask[:,:NY//2],1)
width = 0.342 - 2*(0.342-0.094)/2 * np.exp(-(x-1/2)**2*100)


########## functions 
def apply_bc_x(q):
    global dn_curr
    qbc = np.zeros((3,NX+4,NY))
    q1 = np.copy(q)
    
    q1[1,:,:][mask == 1] = 0
    q1[2,:,:][mask == 1] = 0

    qbc[:,2:-2,:] = q1[:,:,:]
    qbc[:,0,:] = q[:,0,:]
    qbc[:,-1,:] = q[:,-1,:]
    qbc[0,0,:] = n1
    qbc[1,0,:] = q[1,1,:]
    qbc[1,-1,:] = q[1,-2,:]
    qbc[0,-1,:] = n2
    qbc[2,0,:] = q1[2,0,:]
    qbc[2,-1,:] = q1[2,-1,:]
    qbc[1,-1,:] = q[1,-1,:] 
    qbc[:,-2,:] = np.copy(qbc[:,-1,:])
    qbc[:,1,:] = np.copy(qbc[:,0,:])
    return qbc

def minmod(a,b):
    return 1/2*(np.sign(a)+np.sign(b)) * np.minimum(np.abs(a),np.abs(b))

def apply_bc_y(q):
    qbc = np.zeros((3,NX,NY+2))
    q1 = np.copy(q)
    q1[1,:,:][mask == 1] = 0
    q1[2,:,:][mask == 1] = 0
    qbc[:,:,1:-1] = q1[:,:,:]
    qbc[:,:,0] = q[:,:,0]
    qbc[:,:,-1] = q[:,:,-1]
    qbc[1,:,0] = 0#-q[1,:,0]
    qbc[1,:,-1] = 0#-q[1,:,-1]
    qbc[2,:,0] = 0#-q[2,:,0]
    qbc[2,:,-1] = 0#-q[2,:,-1]
    return qbc

def A(q):
    A = np.zeros(q.shape)
    A[0,:,:] = q[1,:,:]
    A[1,:,:] = q[1,:,:]**2/q[0,:,:] + 1/2*q[0,:,:]**2
    A[2,:,:] = q[1,:,:]*q[2,:,:]/q[0,:,:]
    return A

def A1(q):
    A = np.zeros(3)
    A[0] = q[1]
    A[1] = q[1]**2/q[0] + 1/2*q[0]**2
    A[2] = q[1]*q[2]/q[0]
    return A

def B(q):
    B= np.zeros(3)
    B[0] = q[2]
    B[2] = q[2]**2/q[0] + 1/2*(q[0])**2
    B[1] = q[1]*q[2]/q[0]
    return B

def fx(ql,qr):
    nl,nr = ql[0],qr[0]
    ul,ur = ql[1]/nl,qr[1]/nr
    vl,vr = ql[2]/nl,qr[2]/nr
    u = (np.sqrt(nl)*ul + np.sqrt(nr)*ur)/(np.sqrt(nl) + np.sqrt(nr))
    # TODO: the correct "average" value of v is given on page 481 of Leveque
    v = (np.sqrt(nl)*vl + np.sqrt(nr)*vr)/(np.sqrt(nl) + np.sqrt(nr))
    c = np.sqrt(1/2*(nl+nr))
    r1= np.zeros((3))
    r2,r3 = np.copy(r1),np.copy(r1)
    r1[0] = 1
    r1[1] = u - c
    r1[2] = v
    r2[2]=1
    r3[0] = 1
    r3[1] = u + c
    r3[2] = v
    w1 = u - c
    w2 = u
    w3 = u + c
    (d1,d2,d3) = qr-ql
    x = -(-c*d1+d2-d1*u)/2/c
    y = d3 -d1*v
    z = -(-c*d1-d2+d1*u)/2/c
    wave = np.abs(w1)*x*r1 + np.abs(w2)*y*r2 + np.abs(w3)*z*r3
    f = 1/2*(A1(qr) + A1(ql) - wave) - eta/h*np.array([0,ur-ul,vr-vl])
    return f

def fy(ql,qr):
    nl,nr = ql[0],qr[0]
    ul,ur = ql[1]/nl,qr[1]/nr
    vl,vr = ql[2]/nl,qr[2]/nr
    u = (np.sqrt(nl)*ul + np.sqrt(nr)*ur)/(np.sqrt(nl) + np.sqrt(nr))
    # TODO: the correct "average" value of v is given on page 481 of Leveque
    v = (np.sqrt(nl)*vl + np.sqrt(nr)*vr)/(np.sqrt(nl) + np.sqrt(nr))
    c = np.sqrt(1/2*(nl+nr))
    r1= np.zeros((3))
    r2,r3 = np.copy(r1),np.copy(r1)
    r1[0] = 1
    r1[1] = u
    r1[2] = v-c
    r3[1]=1
    r2[0] = 1
    r2[1] = u
    r2[2] = v+c
    w1 = v - c
    w2 = v+c
    w3 = v
    (d1,d2,d3) = qr-ql
    x = -(-c*d1+d3-d1*v)/2/c
    y = -(-c*d1-d3+d1*v)/2/c
    z=d2-d1*u
    wave = np.abs(w1)*x*r1 + np.abs(w2)*y*r2 + np.abs(w3)*z*r3
    f = 1/2*(B(qr) + B(ql) - wave) - eta/hy*np.array([0,ur-ul,vr-vl])
    return f

w = lambda u,v,c: np.array([u-c,u, u+c])
R = lambda u,v,c: np.array([[1,0,1],[u-c,0,u+c],[v,1,v]])
L = lambda u,v,c: np.array([[(u+c)/2/c, -1/2/c, 0], [-v, 0 ,1], [(c-u)/2/c, 1/2/c, 0]])

def FX(Q):
    dQ = Q[:,1:,:]-Q[:,:-1,:]
    f = A(Q)
    favr = 0.5*(f[:,:-1,:] + f[:,1:,:])
    usqrt = Q[1,:,:]/Q[0,:,:]
    vsqrt = Q[2,:,:]/Q[0,:,:]
    nsqrt = np.sqrt(Q[0,:,:])
    uavr = (usqrt[:-1,:]*nsqrt[:-1,:] + usqrt[1:,:]*nsqrt[1:,:])/(nsqrt[:-1,:]+nsqrt[1:,:])
    vavr = (vsqrt[:-1,:]*nsqrt[:-1,:] + vsqrt[1:,:]*nsqrt[1:,:])/(nsqrt[:-1,:]+nsqrt[1:,:])
    cavr =np.sqrt(1/2*(Q[0,:-1,:]+Q[0,1:,:]))
    Ra = np.zeros((3,3, uavr.shape[0],uavr.shape[1]))
    for i in range(uavr.shape[0]):
        for j in range(uavr.shape[1]):
            u,v,c = uavr[i,j], vavr[i,j], cavr[i,j]
            alpha = np.matmul(L(u,v,c),dQ[:,i,j])
            Ra[:,:,i,j] = np.matmul(R(u,v,c),np.diag(alpha))
    sn = np.zeros((3,3,Ra.shape[2]-1,Ra.shape[3]))
    for i in range(sn.shape[2]):
        for j in range(sn.shape[3]):
            sn[:,:,i,j] = minmod(Ra[:,:,i+1,j],Ra[:,:,i,j])
    F = np.zeros((3, sn.shape[2]-1,sn.shape[3]))
    for i in range(F.shape[1]):
        for j in range(F.shape[2]):
            u = uavr[i+1,j]
            v = vavr[i+1,j]
            c = cavr[i+1,j]
            eigs = w(u,v,c)
            nu = k/h*eigs
            if np.max(np.abs(nu)) >= 1:
                print('Warning: CFL Condition')
            S = 0.5*eigs*(np.sign(nu)-nu)
            Aabs = R(u,v,c).dot(np.diag(np.abs(eigs))).dot(L(u,v,c))
            F[:,i,j] = favr[:,i+1,j] - 0.5*Aabs.dot(dQ[:,i+1,j])+ sn[:,:,i,j].dot((eigs>0)*S) + sn[:,:,i+1,j].dot((eigs<0)*S)
    du = Q[1,1:,:]/Q[0,1:,:]-Q[1,:-1,:]/Q[0,:-1,:]     
    dv = Q[2,1:,:]/Q[0,1:,:]-Q[2,:-1,:]/Q[0,:-1,:]    
    F[1,:,:] += -eta / h * du[1:-1,:]
    F[2,:,:] += -eta/ h * dv[1:-1,:]
    return F

def save_data(q):
    np.savez(
        f"{results_dir}/{filename}",
        q = q,
        x = x,
        y = y,
    )
    return "saved"

def x_sweep(q,dt):
    q1= np.copy(q)
    Q = apply_bc_x(q)
    F = FX(Q)
    q1 = q - dt/h*(F[:,1:,:]-F[:,:-1,:])
    #for i in range(2,q.shape[1]+2):
     #   for j in range(0,q.shape[2]):
      #      q1[:,i-2,j] = Q[:,i,j] - dt/h*(fx(Q[:,i,j], Q[:,i+1,j]) - fx(Q[:,i-1,j], Q[:,i,j]))
    return q1

def y_sweep(q,dt):
    Q = apply_bc_y(q)
    for i in range(0,q.shape[1]):
        for j in range(1,q.shape[2]+1):
            q[:,i,j-1] = Q[:,i,j] - dt/hy*(fy(Q[:,i,j], Q[:,i,j+1]) - fy(Q[:,i,j-1], Q[:,i,j]))
    return q

def relax_step(q,dt):
    q1 = np.copy(q)
    q1[1,:,:] = q[1,:,:] * np.exp(-gamma *dt)
    q1[2,:,:] = q[2,:,:] * np.exp(-gamma * dt)
    return q1





# initial conditions
q = np.ones((3,NX,NY))
#q[0,x<=LX/2,:] = n1
#q[0,x>LX/2,:] = n2

q[0,:,:] = n1 + (1+np.tanh((X-0.5)*10))/2 * (n2-n1)
#q[0,:,:][mask==1] = 0.0001
#local_vs = np.sqrt(q[0,:,:])/vs
q[1,:,:] = 0#J #0.001 #-6*J/width**3*(LY/2+width/2-Y)*(LY/2-width/2-Y)
q[2,:,:] = 0.00



#q[0,:,:][mask==1]=0.001
t = 0

# main loop
index = 0
simtime = 0
storage = np.zeros((1,3,NX,NY))
# plotting initialization
#fig, axes = plt.subplots(2,2, figsize=(10,8))
#plt.show(block=False)
#ax1,ax2 = axes[0]
#ax3,ax4 = axes[1]
diag_list = []
wall_time = 0
stop_wall_time_seconds = stop_wall_time * 3600
start_time = datetime.datetime.now()
while wall_time < stop_wall_time_seconds:

    wall_time = (datetime.datetime.now()-start_time).total_seconds()

    q_old = np.copy(q)
    
    
    q = relax_step(q, k/2)
    q = y_sweep(q, k/2)
    q = x_sweep(q, k)
    q = y_sweep(q, k/2)
    q = relax_step(q, k/2)
    

    

    # once the save_increment is reached, 
    if (simtime + k)//save_increment != simtime//save_increment or simtime == 0:
        if wall_time > save_after * 3600:
            storage = np.concatenate([storage, np.expand_dims(q,0)],axis=0)

        print(f"time elapsed {datetime.datetime.now()-start_time} ; simtime = {simtime:.4f}" )
    
    # finally 
    simtime += k
    index += 1
    # plotting loop
    # TODO incorporate into config in clean way
    #if index % 20 == 0:
    
        

        #local_vs =np.sqrt(n0*q[0,:,:]/2*np.pi)/m*hbar# / vs
        #veloc = q[1,:,:]/q[0,:,:]

        #ax1.cla()
        #ax2.cla()
        #ax3.cla()
        #ax4.cla()
        #ax1.plot(volts)
        #ax1.set_xlabel(r"$x$ ($\mu m$)")
        #ax1.set_ylabel("$V$ (V)")
        #ax1.set_title("Voltage across constriction")
        #ax2.plot(q[1,:,NY//2])

        #ax3.plot(local_vs(q)[:,NY//2]/vs)
        #ax3.plot(veloc[:,NY//2])
        #ax3.set_title(r"Speeds")


        #ax4.pcolormesh(q[1,:,:])
        
        #fig.canvas.start_event_loop(0.001)
        #fig.canvas.draw_idle()

# finally, after everything is finished, save the data 
save_data(np.array(storage))


