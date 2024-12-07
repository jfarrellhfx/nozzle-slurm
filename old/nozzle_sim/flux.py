import numpy as np
from .utility import minmod


def FX(Q,k,h,eta):
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
    cavr =np.sqrt((Q[0,:-1,:]+Q[0,1:,:]))

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
    nu = k / h * eigs
    
    # build an array of Aabs values for each cell
    #Aabs = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))
    Aabs = np.einsum('imkl,mjkl,jnkl->inkl', R_values, np.abs(eigs_diag), L_values)
    Aabs_lim = minmod(Aabs[:,1:,:], Aabs[:,:-1,:])

    alpha = np.einsum('ijkl,jkl->ikl', L_values, dQ)
    alpha_diag = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))
    alpha_diag[0, 0, :, :] = alpha[0, :, :]
    alpha_diag[1, 1, :, :] = alpha[1, :, :]
    alpha_diag[2, 2, :, :] = alpha[2, :, :]
    Ra = np.einsum("ij...,jk...->ik...", R_values, alpha_diag)
    sn = minmod(Ra[:,:,1:,:], Ra[:,:,:-1,:])
    S = 0.5*eigs*(np.sign(nu)-nu)
    high_res_plus = np.einsum("ijkl,jkl->ikl", sn[:,:,:-1,:], ((eigs[:,1:-1,:]>0)*S[:,1:-1,:]))
    high_res_minus = np.einsum("ijkl,jkl->ikl", sn[:,:,1:,:], ((eigs[:,1:-1,:]<0)*S[:,1:-1,:]))

    #alpha_lim = minmod(alpha[:,1:,:], alpha[:,:-1,:])

  

    #high_res = 1/2*np.einsum("ijkl,jnkl,nkl->ikl", R_values[:,:,:-1,:], np.abs(eigs_diag[:,:,:-1,:])*(np.ones_like(eigs_diag[:,:,:-1,:]) - k/h*np.abs(eigs_diag[:,:,:-1,:])), alpha_lim)[:,1:,:]
    
    #high_res_plus =0* np.einsum('jikl,ikl->jkl', sn, (eigs[:,1:,:] > 0) * S[:,1:,:])
    #high_res_minus =0* np.einsum('jikl,ikl->jkl', sn, (eigs[:,:-1,:] < 0) * S[:,:-1,:])
    F = favr[:,1:-1,:] - 0.5 * np.einsum('ijkl,jkl->ikl', Aabs[:,:,1:-1,:], dQ[:,1:-1,:]) + high_res_plus + high_res_minus

    du = Q[1,1:,:]/Q[0,1:,:]-Q[1,:-1,:]/Q[0,:-1,:]     
    dv = Q[2,1:,:]/Q[0,1:,:]-Q[2,:-1,:]/Q[0,:-1,:]    
    F[1,:,:] += -eta / h * du[1:-1,:]
    F[2,:,:] += -eta/ h * dv[1:-1,:]
    return F

def FY(Q,k,h,eta):
  
    Flux = np.zeros_like(Q)

    dQ = Q[:,:,1:]-Q[:,:,:-1]
    f = np.zeros_like(Q)
    f[0, :, :] = Q[2, :, :]
    f[2, :, :] = Q[2, :, :]**2 / Q[0, :, :] + 1/2 * (Q[0, :, :])**2
    f[1, :, :] = Q[1, :, :] * Q[2, :, :] / Q[0, :, :]
    favr = 0.5*(f[:,:,:-1] + f[:,:,1:])
    #print(np.max(favr[0,:,:]), np.min(favr[0,:,:]))

    # we'll just vectorize the calculations from fy
    usqrt = Q[1,:,:]/Q[0,:,:]
    vsqrt = Q[2,:,:]/Q[0,:,:]    
    nsqrt = np.sqrt(Q[0,:,:])
    uavr = (usqrt[:,:-1]*nsqrt[:,:-1] + usqrt[:,1:]*nsqrt[:,1:])/(nsqrt[:,:-1]+nsqrt[:,1:])
    vavr = (vsqrt[:,:-1]*nsqrt[:,:-1] + vsqrt[:,1:]*nsqrt[:,1:])/(nsqrt[:,:-1]+nsqrt[:,1:])
    cavr =np.sqrt((Q[0,:,:-1]+Q[0,:,1:]))

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

    eigs_diag = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))
    eigs_diag[0, 0, :, :] = eigs[0, :, :]
    eigs_diag[1, 1, :, :] = eigs[1, :, :]
    eigs_diag[2, 2, :, :] = eigs[2, :, :]
    # build an array of nu values for each cell
    nu = np.zeros((3, uavr.shape[0], uavr.shape[1]))
    nu = k / h * eigs
    
    # build an array of Aabs values for each cell
    #Aabs = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))
    Aabs = np.einsum('imkl,mjkl,jnkl->inkl', R_values, np.abs(eigs_diag), L_values)
    Aabs_lim = minmod(Aabs[:,1:,:], Aabs[:,:-1,:])

    alpha = np.einsum('ijkl,jkl->ikl', L_values, dQ)
    alpha_diag = np.zeros((3, 3, uavr.shape[0], uavr.shape[1]))
    alpha_diag[0, 0, :, :] = alpha[0, :, :]
    alpha_diag[1, 1, :, :] = alpha[1, :, :]
    alpha_diag[2, 2, :, :] = alpha[2, :, :]
    Ra = np.einsum("ij...,jk...->ik...", R_values, alpha_diag)
    sn = minmod(Ra[:,:,:,1:], Ra[:,:,:,:-1])
    S = 0.5*eigs*(np.sign(nu)-nu)
    high_res_plus = np.einsum("ijkl,jkl->ikl", sn[:,:,:,:-1], ((eigs[:,:,1:-1]>0)*S[:,:,1:-1]))
    high_res_minus = np.einsum("ijkl,jkl->ikl", sn[:,:,:,1:], ((eigs[:,:,1:-1]<0)*S[:,:,1:-1]))

    Aabs = np.einsum('ijkl,mjkl,jnkl->inkl', R_values, np.abs(eigs_diag), L_values)
    Flux = favr[:,:,1:-1]-0.5 * np.einsum('ijkl,jkl->ikl', Aabs[:,:,:,1:-1], dQ[:,:,1:-1]) + high_res_plus + high_res_minus

    du = Q[1,:,1:]/Q[0,:,1:]-Q[1,:,:-1]/Q[0,:,:-1]
    dv = Q[2,:,1:]/Q[0,:,1:]-Q[2,:,:-1]/Q[0,:,:-1]
    Flux[1,:,:] += -eta/h*du[:,1:-1]
    Flux[2,:,:] += -eta/h*dv[:,1:-1]

    

    return Flux

