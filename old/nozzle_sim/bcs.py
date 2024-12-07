import numpy as np

def apply_bc_x(q,NX,NY,mask,n1,n2):
    """
    apply boundary conditions in x direction
    """
    #global mask, NX, NY, n1, n2
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



def apply_bc_y(q,NX,NY,mask):
    """
    boundary conditions in y direction
    """
    qbc = np.zeros((3,NX,NY+4))
    q1 = np.copy(q)
    q1[1,:,:][mask == 1] = 0
    q1[2,:,:][mask == 1] = 0
    qbc[:,:,2:-2] = q1[:,:,:]
    qbc[:,:,0] = q[:,:,0]
    qbc[:,:,-1] = q[:,:,-1]
    qbc[1,:,0] = 0#-q[1,:,0]
    qbc[1,:,-1] = 0#-q[1,:,-1]
    qbc[2,:,0] = 0#-q[2,:,0]
    qbc[2,:,-1] = 0#-q[2,:,-1]
    qbc[:,:,1] = np.copy(qbc[:,:,0])
    qbc[:,:,-2] = np.copy(qbc[:,:,-1])
    return qbc