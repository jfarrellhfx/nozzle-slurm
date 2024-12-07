"""
Jack Farrell, CU Boulder, 2024

2D channel flow of viscous electron gas through a nozzle using finite volume methods: Roe's approximate Riemann solver for 2D shallow water.  Dimensional splitting for x,y evolution. High resolution in x direction with minmod slope limiter.

TODO: proper transverse Riemann solver for the shallow water equations
"""

# imports
import numpy as np
import os
import argparse
import datetime
from scipy.integrate import simps

print(1)
# simulation parameters
LX, LY = 1.0, 0.342 
h = LX/100
hy = LY/50
k = 0.0005
save_increment = 0.01

x = np.arange(0,LX,h)
y = np.arange(0,LY,hy)
NX = len(x)
NY = len(y)
Y,X = np.meshgrid(y,x)


# key variables
# these are taken as defaults if not supplied in command-line interface
V = 12 # source-drain voltage
eta = 0.001
results_dir = "results2"
stop_time = 32400/2


# command-line interface
parser = argparse.ArgumentParser(prog='constrictionFlow', description='hydrodynamic electron flow through a de-Laval nozzle.')   
parser.add_argument("--V", help="Voltage difference", type=float, default=100*V)
parser.add_argument("--results-dir", default=results_dir)
args = parser.parse_args()
V, results_dir = args.V/100, args.results_dir
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
filename = f"V={V:.3f}-eta={eta:.5f}.npz".replace(".","_")


# physical parameters
d = 307e-9
e0 = 8.854e-12
er = 3.9
C = e0*er/d
e = 1.602e-19
W = LY * 40e-6
m = 9.1e-31
V0 = -13
n0 = C*V0/(-e)
vs = np.sqrt(-e*V0/m)
I = 4
Res = 3000/10
J = -I / n0 / vs / 1000 / e / W
dn = C*V/e / n0


# initial conditions
q = np.ones((3,NX,NY))
q[0,:,:] = 1.0
q[1,:,:] = 0.00
q[2,:,:] = 0.00
q0 = np.copy(q)
print(J)

# functions 
def apply_bc_x(q):
    qbc = np.zeros((3,NX+4,NY))
    q1 = np.copy(q)
    qbc[:,2:-2,:] = q1[:,:,:]
    q1[1,:,:][mask == 1] = 0
    q1[2,:,:][mask == 1] = 0
    qbc[:,0,:] = q1[:,0,:]
    qbc[:,-1,:] = q1[:,-1,:]
    qbc[0,0,:] = 1
    qbc[0,-1,:] = 1 - dn
    qbc[2,0,:] = 0
    qbc[2,-1,:] = 0
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
    qbc[:,:,0] = q1[:,:,0]
    qbc[:,:,-1] = q1[:,:,-1]
    qbc[1,:,0] = 0
    qbc[1,:,-1] = 0
    qbc[2,:,0] = 0
    qbc[2,:,-1] = 0
    return qbc

def A(q):
    A = np.zeros(q.shape)
    A[0,:,:] = q[1,:,:]
    A[1,:,:] = q[1,:,:]**2/q[0,:,:] + 1/2*q[0,:,:]**2
    A[2,:,:] = q[1,:,:]*q[2,:,:]/q[0,:,:]
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
    

    f = 1/2*(A(qr) + A(ql) - wave) - eta/h*np.array([0,ur-ul,vr-vl])
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
            
    F[1:,:,:] += -eta / h * dQ[1:,1:-1,:]



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
    q1 = q - k/h*(F[:,1:,:]-F[:,:-1,:])
    #for i in range(2,q.shape[1]+2):
    #    for j in range(0,q.shape[2]):
    #        q1[:,i-2,j] = Q[:,i,j] - dt/h*(fx(Q[:,i,j], Q[:,i+1,j]) - fx(Q[:,i-1,j], Q[:,i,j]))
    return q1

def y_sweep(q,dt):
    Q = apply_bc_y(q)
    for i in range(0,q.shape[1]):
        for j in range(1,q.shape[2]+1):
            q[:,i,j-1] = Q[:,i,j] - dt/hy*(fy(Q[:,i,j], Q[:,i,j+1]) - fy(Q[:,i,j-1], Q[:,i,j]))
    return q

def relax_step(q,dt):
    q1 = np.copy(q)
    relax_coeff = n0*e**2/m*Res * W/vs
    q1[1,:,:] = q[1,:,:] - dt*relax_coeff*q[1,:,:]*q[0,:,:]
    q1[2,:,:] = q[2,:,:] - dt*relax_coeff*q[2,:,:]*q[0,:,:]
    return q1


mask = np.zeros_like(X)
mask[Y < (0.342-0.094)/2 * np.exp(-(X-1/2)**2*40)] = 1
mask[:,NY//2:] = np.flip(mask[:,:NY//2],1)




# main loop
index = 0
simtime = 0
storage = []
#fig, axes = plt.subplots(2,2, figsize=(10,8))
#plt.show(block=False)
#ax1,ax2 = axes[0]
#ax3,ax4 = axes[1]
diag_list = []
wall_time = 0
start_time = datetime.datetime.now()
while wall_time < stop_time:
    try: 
        wall_time = (datetime.datetime.now()-start_time).total_seconds()

        q_old = np.copy(q)
        q = relax_step(q, k/2)
        q = y_sweep(q, k/2)
        q = x_sweep(q, k)
        q = y_sweep(q, k/2)
        q = relax_step(q, k/2)

        index += 1
        simtime += k
        if (simtime - k)//save_increment != simtime//save_increment:
            storage.append(q)
        
        if index % 20 == 0:
            dndt = (q[0,:,:] - q_old[0,:,:]) / k
            diagnosis = np.max(np.abs(dndt))

            diag_list.append(diagnosis)
            current = np.sum(q[1,-1,:])* hy * e * 40e-6 * n0*vs * 1000
            print(f"Simtime = {simtime:.4f}, I = {current:.4f}")

            local_vs = np.sqrt(e**2*q[0,:,:]/m/C*n0)
            veloc = q[1,:,:]/q[0,:,:] * vs

            #ax1.cla()
            #ax2.cla()
            #ax3.cla()
            #ax4.cla()
            #ax1.plot(13 - (n0 * q[0,:,NY//2]) * e / C)
            #ax1.set_xlabel(r"$x$ ($\mu m$)")
            #ax1.set_ylabel("$V$ (V)")
            #ax1.set_title("Voltage across constriction")
            #ax2.pcolormesh(q[2,:,:])

            #ax3.plot(local_vs[:,NY//2])
            #ax3.plot(veloc[:,NY//2])
            #ax3.set_title(r"Speeds")


            #ax4.plot(q[1,-1,:])
            #ax4.set_xlabel("Iter.")
            #plt.plot(x*40, q[1,:,NY//2]/q[0,:,NY//2]*vs)
            #plt.plot(x*40, 13-n0*q[0,:,NY//2]*e/C)
            #plt.xlim(10,30)
            #fig.canvas.start_event_loop(0.001)
            #fig.canvas.draw_idle()
    except:
        save_data(np.array(storage))

save_data(np.array(storage))


