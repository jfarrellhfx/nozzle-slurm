import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def minmod(a,b):
    return 1/2*(np.sign(a)+np.sign(b)) * np.minimum(np.abs(a),np.abs(b))

def make_plot(q,simtime,x,y,X,Y,mask):
    NY = q.shape[2]
    global fig, axes, ax1, ax2, ax3, ax4, cbar
    mask[mask==0] = np.nan
    u_mag = np.sqrt(q[1,:,:]**2 + q[2,:,:]**2)/q[0]

    # plotting initialization
    if simtime == 0:
        
        fig, axes = plt.subplots(2,2, figsize=(10,8))
        ax1,ax2 = axes[0]
        ax3,ax4 = axes[1]
        plt.show(block=False)
        

    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    
    # density plot
    ax1.plot(x, q[0,:,NY//2])
    ax1.set_title(' Local Density')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$n(x)$')


    # Mach number plot
    ux = q[1,:,:]/q[0,:,:]
    uy = q[2,:,:]/q[0,:,:]
    local_vs = np.sqrt(q[0,:,:])
    u_mag = np.sqrt(ux**2 + uy**2)

    #
    ax2.plot(x, ux[:,NY//2]/local_vs[:,NY//2])
    ax2.set_title('Local Mach number')
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$M(x)$')



    ux[mask == 1] = np.nan
    uy[mask == 1] = np.nan
    

    #stream = ax4.streamplot(X.T, Y.T, ux.T, uy.T, color=u_mag.T,linewidth=2,broken_streamlines=False,density =0.5)
    vcolor = ax4.pcolormesh(X.T, Y.T, u_mag.T, cmap='viridis')
    for art in ax4.get_children():
        if not isinstance(art, matplotlib.patches.FancyArrowPatch):
            continue
        art.remove()        # Method 1
    if simtime == 0:
        cbar = fig.colorbar(vcolor, ax=ax4, location='bottom', label='Velocity magnitude')
    else:
        cbar.update_normal(vcolor)
    pcm = ax4.pcolormesh(X.T, Y.T, mask.T, cmap='gray', zorder=1000)


    ax4.set_xlabel('$x$')
    ax4.set_ylabel('$y$')
    ax4.set_title('Flow Velocity')
    pcm = ax4.pcolormesh(X.T, Y.T, mask.T, cmap='gray', zorder=1000)


    dens=ax3.pcolormesh(x,y, q[0,:,:].T)
    pcm2 = ax3.pcolormesh(X.T, Y.T, mask.T, cmap='gray', zorder=1000)
    global cbar2
    if simtime == 0:
        cbar2 = fig.colorbar(dens, ax=ax3, location='bottom', label='Density')
    else:
        cbar2.update_normal(dens)
  


    ax3.set_title('Density')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')


    plt.tight_layout()

    fig.canvas.start_event_loop(0.001)
    fig.canvas.draw_idle()


