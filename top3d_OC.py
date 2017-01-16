from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt

# MAIN DRIVER
def main(nelx,nely,nelz,volfrac,penal,rmin,ft):
    print("Minimum compliance problem with OC")
    print("ndes: " + str(nelx) + " x " + str(nely)+ " x " + str(nelz))
    print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("Filter method: Sensitivity based")

    # Max and min stiffness
    Emin=1e-9
    Emax=1.0
    nu = 0.3

    # dofs
    ndof = 3*(nelx+1)*(nely+1)*(nelz+1)

    #USER - DEFINED LOAD DOFs
    kl = np.arange(nelz+1)
    loadnid = kl * (nelx + 1) * (nely + 1) + (nely + 1) * (nelx + 1)-1 # Node IDs
    loaddof = 3*loadnid+1 # DOFs

    #USER - DEFINED SUPPORT FIXED DOFs
    [jf,kf] = np.meshgrid(np.arange(nely+1),np.arange(nelz+1)) # Coordinates
    fixednid = (kf)*(nely+1)*(nelx+1)+jf # Node IDs
    fixeddof = np.array([3*fixednid,3*fixednid+1,3*fixednid+2]).flatten() # DOFs

    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(nely * nelx * nelz, dtype=float)
    xold = x.copy()
    xPhys = x.copy()
    g = 0  # must be initialized to use the NGuyen/Paulino OC approach

    # FE: Build the index vectors for the for coo matrix format.
    KE = lk(nu)
    edofMat = np.zeros((nelx * nely * nelz, 24), dtype=int)
    for elz in range(nelz):
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + (elx * nely) + elz * (nelx * nely)
                n1 = elz * (nelx + 1) * (nely + 1) + (nely + 1) * elx + ely
                n2 = elz * (nelx + 1) * (nely + 1) + (nely + 1) * (elx + 1) + ely
                n3 = (elz + 1) * (nelx + 1) * (nely + 1) + (nely + 1) * elx + ely
                n4 = (elz + 1) * (nelx + 1) * (nely + 1) + (nely + 1) * (elx + 1) + ely
                edofMat[el, :] = np.array(
                    [3 * n1 + 3, 3 * n1 + 4, 3 * n1 + 5, 3 * n2 + 3, 3 * n2 + 4, 3 * n2 + 5, \
                     3 * n2, 3 * n2 + 1, 3 * n2 + 2, 3 * n1, 3 * n1 + 1, 3 * n1 + 2, \
                     3 * n3 + 3, 3 * n3 + 4, 3 * n3 + 5, 3 * n4 + 3, 3 * n4 + 4, 3 * n4 + 5, \
                     3 * n4, 3 * n4 + 1, 3 * n4 + 2, 3 * n3, 3 * n3 + 1, 3 * n3 + 2])
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((24, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 24))).flatten()

    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    nfilter = nelx * nely * nelz * ((2 * (np.ceil(rmin) - 1) + 1) ** 3)
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for z in range(nelz):
        for i in range(nelx):
            for j in range(nely):
                row = i * nely + j + z * (nelx * nely)
                kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
                ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(rmin), nely))
                mm1 = int(np.maximum(z - (np.ceil(rmin) - 1), 0))
                mm2 = int(np.minimum(z + np.ceil(rmin), nelz))
                for m in range(mm1, mm2):
                    for k in range(kk1, kk2):
                        for l in range(ll1, ll2):
                            col = k * nely + l + m * (nelx * nely)
                            fac = rmin - np.sqrt((i - k) * (i - k) + (j - l) * (j - l) + (z - m) * (z - m))
                            iH[cc] = row
                            jH[cc] = col
                            sH[cc] = np.maximum(0.0, fac)
                            cc = cc + 1
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely * nelz, nelx * nely * nelz)).tocsc()
    Hs = H.sum(1)


    # BC's and support
    dofs = np.arange(3 * (nelx + 1) * (nely + 1) * (nelz + 1))
    free = np.setdiff1d(dofs, fixeddof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1),dtype=float)
    u = np.zeros((ndof, 1),dtype=float)
    # Set load
    f[loaddof, 0] = -1
    # Initialize plot and plot the initial design
    '''plt.ion() # Ensure that redrawing is possible
    fig, ax = plt.subplots()
    im = ax.imshow(-xPhys.reshape((nelx, nely, nelz)).T, cmap='gray', \
               interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    fig.show()'''
    # Set loop counter and gradient vectors
    loop = 0
    change = 1
    dv = np.ones(nely * nelx * nelz,dtype=float)
    dc = np.ones(nely * nelx * nelz,dtype=float)
    ce = np.ones(nely * nelx * nelz,dtype=float)
    while change > 0.01 and loop < 2000:
        loop = loop + 1
        # Setup and solve FE problem
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + (xPhys) ** penal * (Emax - Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = K[free, :][:, free]
        # Solve system
        u[free, 0] = spsolve(K, f[free, 0])
        # Objective and sensitivity
        ce[:] = (np.dot(u[edofMat].reshape(nelx * nely * nelz, 24), KE) * u[edofMat].reshape(nelx * nely * nelz, 24)).sum(1)
        obj = ((Emin + xPhys ** penal * (Emax - Emin)) * ce).sum()
        dc[:] = (-penal * xPhys ** (penal - 1) * (Emax - Emin)) * ce
        dv[:] = np.ones(nely * nelx * nelz)
        # Sensitivity filtering:
        if ft == 0:
            dc[:] = np.asarray((H * (x * dc))[np.newaxis].T / Hs)[:, 0] / np.maximum(0.001, x)
        elif ft == 1:
            dc[:] = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
            dv[:] = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]
        # Optimality criteria
        xold[:] = x
        (x[:], g) = oc(nelx, nely, nelz, x, volfrac, dc, dv, g)
        # Filter design variables
        if ft == 0:
            xPhys[:] = x
        elif ft == 1:
            xPhys[:] = np.asarray(H * x[np.newaxis].T / Hs)[:, 0]
        # Compute the change by the inf. norm
        change = np.linalg.norm(x.reshape(nelx * nely * nelz, 1) - xold.reshape(nelx * nely * nelz, 1), np.inf)
        # Plot to screen
        '''im.set_array(-xPhys.reshape((nelx, nely, nelz)).T)
        fig.canvas.draw()
        plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)'''
        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format( \
            loop, obj, (g + volfrac * nelx * nely * nelz) / (nelx * nely * nelz), change))
    '''plt.show()'''
    print xPhys

    #plot a cross section perpendicular to z direction
    x1=-xPhys[0:300]
    plt.ion()  # Ensure that redrawing is possible
    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(x1.reshape((nelx, nely)).T, cmap='gray', \
                   interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    fig1.show()
    plt.show()
    plt.pause(0.01)

    x2=-xPhys[300:600]
    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(x2.reshape((nelx, nely)).T, cmap='gray', \
                   interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    fig2.show()
    plt.show()
    plt.pause(0.01)

    x3=-xPhys[600:900]
    fig3, ax3 = plt.subplots()
    im3 = ax3.imshow(x3.reshape((nelx, nely)).T, cmap='gray', \
                   interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    fig3.show()
    plt.show()
    plt.pause(0.01)

    x4=-xPhys[900:1200]
    fig4, ax4 = plt.subplots()
    im4 = ax4.imshow(x4.reshape((nelx, nely)).T, cmap='gray', \
                   interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    fig3.show()
    plt.show()
    plt.pause(0.01)

    x5=-xPhys[1200:1500]
    fig5, ax5 = plt.subplots()
    im5 = ax5.imshow(x5.reshape((nelx, nely)).T, cmap='gray', \
                   interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    fig5.show()
    plt.show()
    plt.pause(0.01)

    raw_input("Press any key...")

def oc(nelx,nely,nelz,x,volfrac,dc,dv,g):
    l1=0
    l2=1e9
    move=0.2
    # reshape to perform vector operations
    xnew=np.zeros(nelx*nely*nelz,dtype=float)
    while (l2-l1)/(l1+l2)>1e-3:
        lmid=0.5*(l2+l1)
        xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
        gt=g+np.sum((dv*(xnew-x)))
        if gt>0 :
            l1=lmid
        else:
            l2=lmid
    return (xnew,gt)

def lk(nu):
    A = np.array([[32,6,-8,6,-6,4,3,-6,-10,3,-3,-3,-4,-8],[-48,0,0,-24,24,0,0,0,12,-12,0,12,12,12]])
    b = np.array([[1],[nu]])
    k = 1/float(144)*np.dot(A.T,b).flatten()

    K1 = np.array([[k[0],k[1],k[1],k[2],k[4],k[4]],
    [k[1],k[0],k[1],k[3],k[5],k[6]],
    [k[1],k[1],k[0],k[3],k[6],k[5]],
    [k[2],k[3],k[3],k[0],k[7],k[7]],
    [k[4],k[5],k[6],k[7],k[0],k[1]],
    [k[4],k[6],k[5],k[7],k[1],k[0]]])

    K2 = np.array([[k[8],k[7],k[11],k[5],k[3],k[6]],
    [k[7],k[8],k[11],k[4],k[2],k[4]],
    [k[9],k[9],k[12],k[6],k[3],k[5]],
    [k[5],k[4],k[10],k[8],k[1],k[9]],
    [k[3],k[2],k[4],k[1],k[8],k[11]],
    [k[10],k[3],k[5],k[11],k[9],k[12]]])

    K3 = np.array([[k[5],k[6],k[3],k[8],k[11],k[7]],
    [k[6],k[5],k[3],k[9],k[12],k[9]],
    [k[4],k[4],k[2],k[7],k[11],k[8]],
    [k[8],k[9],k[1],k[5],k[10],k[4]],
    [k[11],k[12],k[9],k[10],k[5],k[3]],
    [k[1],k[11],k[8],k[3],k[4],k[2]]])

    K4 = np.array([[k[13],k[10],k[10],k[12],k[9],k[9]],
    [k[10],k[13],k[10],k[11],k[8],k[7]],
    [k[10],k[10],k[13],k[11],k[7],k[8]],
    [k[12],k[11],k[11],k[13],k[6],k[6]],
    [k[9],k[8],k[7],k[6],k[13],k[10]],
    [k[9],k[7],k[8],k[6],k[10],k[13]]])

    K5 = np.array([[k[0],k[1],k[7],k[2],k[4],k[3]],
    [k[1],k[0],k[7],k[3],k[5],k[10]],
    [k[7],k[7],k[0],k[4],k[10],k[5]],
    [k[2],k[3],k[4],k[0],k[7],k[1]],
    [k[4],k[5],k[10],k[7],k[0],k[7]],
    [k[3],k[10],k[5],k[1],k[7],k[0]]])

    K6 = np.array([[k[13],k[10],k[6],k[12],k[9],k[11]],
    [k[10],k[13],k[6],k[11],k[8],k[1]],
    [k[6],k[6],k[13],k[9],k[1],k[8]],
    [k[12],k[11],k[9],k[13],k[6],k[10]],
    [k[9],k[8],k[1],k[6],k[13],k[6]],
    [k[11],k[1],k[8],k[10],k[6],k[13]]])

    KE1=np.hstack((K1,K2,K3,K4))
    KE2=np.hstack((K2.T,K5,K6,K3.T))
    KE3=np.hstack((K3.T,K6,K5.T,K2.T))
    KE4=np.hstack((K4,K3,K2,K1.T))
    KE = 1/float(((nu+1)*(1-2*nu)))*np.vstack((KE1,KE2,KE3,KE4))

    return(KE)


# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx=30
    nely=10
    nelz=5
    volfrac=0.5
    rmin=1.2
    penal=3.0
    ft=0 # ft==0 -> sens, ft==1 -> dens
    import sys
    if len(sys.argv)>1: nelx   =int(sys.argv[1])
    if len(sys.argv)>2: nely   =int(sys.argv[2])
    if len(sys.argv)>3: nelz   =int(sys.argv[3])
    if len(sys.argv)>4: volfrac=float(sys.argv[4])
    if len(sys.argv)>5: rmin   =float(sys.argv[5])
    if len(sys.argv)>6: penal  =float(sys.argv[6])
    if len(sys.argv)>7: ft     =int(sys.argv[7])
    main(nelx,nely,nelz,volfrac,penal,rmin,ft)


