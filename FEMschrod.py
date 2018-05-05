import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import gmsh
from mpl_toolkits.mplot3d import Axes3D
from trigauss import *
from matplotlib import rc
import math
rc('text', usetex=True)
mesh = gmsh.Mesh()

# Order of quadrature and mesh to use.
gauord = 7
mesh.read_msh('sqmesh.msh')

E = mesh.Elmts[9][1]
V = mesh.Verts[:,:2]

ne = E.shape[0]
nv = V.shape[0]
X = V[:,0]
Y = V[:,1]

def gradphi(funcnum,r,s):
    if funcnum == 0:
        grad = np.array([-3.0+4.0*r+4.0*s,-3.0+4.0*r+4.0*s])
    elif funcnum == 1:
        grad = np.array([-1.0+4.0*r,0.0])
    elif funcnum == 2:
        grad = np.array([0.0,4.0*s-1.0])        
    elif funcnum == 3:
        grad = np.array([4.0-8.0*r-4.0*s,-4.0*r])       
    elif funcnum == 4:
        grad = np.array([4.0*s,4.0*r])      
    elif funcnum == 5:
        grad = np.array([-4.0*s,4.0-4.0*r-8.0*s])
    return grad

def phi(funcnum,r,s):
    if funcnum == 0:
        val = (1.0-r-s)*(1.0-2.0*r-2.0*s)
    elif funcnum == 1:
        val = r * ( 2.0*r - 1.0)
    elif funcnum == 2:
        val = s * ( 2.0*s - 1.0)
    elif funcnum == 3:
        val = 4.0*r*(1.0-r-s)
    elif funcnum == 4:
        val = 4.0*r*s
    elif funcnum == 5:
        val = 4.0*s*(1.0-r-s)
    return val

def getJ(x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,r,s):
    j11 = (-3.0 + 4.0*r + 4.0*s)*x0 + (4.0*r - 1.0)*x1 - 4.0*(-1.0 + 2.0*r + s)*x3 + 4.0*s*(x4 - x5)
    j12 = (-3.0 + 4.0*r + 4.0*s)*x0 + (4.0*s - 1.0)*x2 + 4.0*r*(x4 - x3) - 4.0*x5*(2.0*s + r - 1.0)
    j21 = (-3.0 + 4.0*r + 4.0*s)*y0 + (4.0*r - 1.0)*y1 - 4.0*(-1.0 + 2.0*r + s)*y3 + 4.0*s*(y4 - y5)
    j22 = (-3.0 + 4.0*r + 4.0*s)*y0 + (4.0*s - 1.0)*y2 + 4.0*r*(y4 - y3) - 4.0*y5*(2.0*s + r - 1.0)
    return np.array([[j11,j12],[j21,j22]])

def getdbasis(r,s):
    return np.array([[-3.0+4.0*r+4.0*s,-1.0+4.0*r,0.0,4.0-8.0*r-4.0*s,4.0*s,-4.0*s],[-3.0+4.0*r+4.0*s,0.0,4.0*s-1.0,-4.0*r,4.0*r,4.0-4.0*r-8.0*s]])

# Initialize arrays
AA = np.zeros((ne, 36))
AA2 = np.zeros((ne, 36))
IA = np.zeros((ne, 36))
JA = np.zeros((ne, 36))

qx,qw = trigauss(gauord)

# Main loop
for ei in range(0,ne):
    Aelem = np.zeros((6,6))
    A2elem = np.zeros((6,6))
    K = E[ei,:]
    x0, y0 = X[K[0]], Y[K[0]]
    x1, y1 = X[K[1]], Y[K[1]]
    x2, y2 = X[K[2]], Y[K[2]]
    x3, y3 = X[K[3]], Y[K[3]]
    x4, y4 = X[K[4]], Y[K[4]]
    x5, y5 = X[K[5]], Y[K[5]]

    # estimate the integral using quadrature
    for qp in range(0,len(qw)):
        r = qx[qp,0]
        s = qx[qp,1]
        w = qw[qp]
        J = getJ(x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,r,s)
        invJ = la.inv(J.T)
        detJ = la.det(J)
        dbasis = getdbasis(r,s)
        dphi = invJ.dot(dbasis)
        Aelem += w * (dphi.T).dot(dphi) * detJ
        phis = np.array([[phi(i,r,s) for i in range(0,6)]])
        A2elem += w * detJ * (phis.T).dot(phis)
    AA2[ei,:] = A2elem.ravel()
    AA[ei, :] = Aelem.ravel()
    IA[ei, :] = np.repeat(K,6)
    JA[ei, :] = np.tile(K,6)


# Assembly
A = sparse.coo_matrix((AA.ravel(), (IA.ravel(), JA.ravel())),dtype=complex)
A = A.tocsr()
A = A.tocoo()
A2 = sparse.coo_matrix((AA2.ravel(), (IA.ravel(), JA.ravel())),dtype=complex)
A2 = A2.tocsr()
A2 = A2.tocoo()

# Boundary conditions
tol = 1e-6
Dflag = np.logical_or.reduce((abs(X) < tol,
                              abs(Y) < tol,
                              abs(X-1.0) < tol,
                              abs(Y-1.0) < tol))
# removing this causes solution to accumulate at corners
for k in range(0, len(A.data)):
    i = A.row[k]
    j = A.col[k]
    if Dflag[i] or Dflag[j]:
        if i == j:
            A.data[k] = complex(1.0,0)
        else:
            A.data[k] = complex(0.0,0)

# this doesn't seem to do much, not sure if this is needed
#for k in range(0, len(A2.data)):
#    i = A2.row[k]
#    j = A2.col[k]
#    if Dflag[i] or Dflag[j]:
#        if i == j:
#            A2.data[k] = complex(1.0,0)
#        else:
#            A2.data[k] = complex(0.0,0.0)

# Now solve (and correct from above)
#A = A.tocsr()
#print np.linalg.cond(A.todense())
#u = sla.spsolve(A, b)
#u = u + u0
hbar, m = 1.0, 1.0
lx, ly = 1.0, 1.0
def uex(x, y, t=0):
    nxs = [1, 2, 1]
    nys = [1, 2, 2]
    phis = []
    for nx, ny in zip(nxs, nys):
        En = (((hbar*np.pi)**2)/(2*m*lx*ly))*(nx**2 + ny**2)
        phit = np.exp(-1j*t*En/hbar)
        N = (np.sqrt(2)/lx)*(np.sqrt(2)/ly)
        phi = np.sin(nx*x*np.pi)*np.sin(ny*y*np.pi)*phit
        phis.append(phi)
    return sum(phis).astype(np.complex128)
    return (np.sin(x*np.pi)*np.sin(y*np.pi)*np.exp(-1j*t*((((hbar*np.pi)**2)/(2*m*lx*ly))*(1**2 + 1**2)))+np.sin(2*x*np.pi)*np.sin(2*y*np.pi)*np.exp(-1j*t*((((hbar*np.pi)**2)/(2*m*lx*ly))*(2**2 + 2**2))))/np.sqrt(2)

def ic(x, y):
    #return uex(x, y)
    return ( np.sin(x*np.pi)*np.sin(y*np.pi)+np.sin(2*x*np.pi)*np.sin(2*y*np.pi) ) / np.sqrt(2)
imagu = complex(0.0,1.0)
nt = 600
tf = 2.0
dt = tf / ( nt - 1.0 )
#A = np.array(A.todense())
#A2 = np.array(A2.todense())
#U = np.matmul(np.linalg.inv(A2-0.5*dt*dummy),A2+0.5*dt*dummy)
U = np.dot(sla.inv(A2+A*(hbar*imagu*(dt*1)/(4.0*m))),A2-A*(hbar*imagu*(dt*1)/(4.0*m))).tocoo()

# this becomes very unstable if done on its own
#for k in range(0, len(U.data)):
#    i = U.row[k]
#    j = U.col[k]
#    if Dflag[i] or Dflag[j]:
#        if i == j:
#            U.data[k] = 1.0
#        else:
#            U.data[k] = 0.0

#print A
#print A2
#print U
psi0 = uex(X,Y)
psi = np.array(psi0)
fig = plt.figure()
errors=np.zeros(nt)
pdens = np.real(psi * np.conj(psi))
plot_error = True
for i in range(0,nt):
    psit = uex(X, Y, i*dt)
    pdens_true = np.real(np.conj(psit) * psit)
    num = 2 if plot_error else 1
    if 1:
        triang = tri.Triangulation(X,Y)
        plt.clf()

        ax = fig.add_subplot(1,num,1)
        ax.set_title("Calculated $|\psi|^{2}$"+", t={0:.3f}/{1:.3f}".format(dt*i, tf))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        surf = ax.tripcolor(X, Y, pdens, triangles=E[:,:6], cmap=plt.cm.jet, linewidth=0.2)
        ax.tricontour(triang, pdens, colors='k',vmin=0,vmax=3)
        fig.colorbar(surf)
        fig.tight_layout()

        if plot_error:
            ax = fig.add_subplot(1,num,2)
            ax.set_title("True $|\psi|^{2}$"+", t={0:.3f}/{1:.3f}".format(dt*i, tf))
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            surf = ax.tripcolor(X, Y, pdens_true, triangles=E[:,:6], cmap=plt.cm.jet, linewidth=0.2)        
            ax.tricontour(triang, pdens_true, colors='k',vmin=0,vmax=3)
            fig.colorbar(surf)
            fig.tight_layout()
        plt.pause(0.0001)
    errors[i] = np.linalg.norm(pdens_true-pdens,np.inf)/np.linalg.norm(pdens_true,np.inf)
    psi = U.dot(psi)
    pdens = np.real(psi * np.conj(psi))
        #plt.savefig('Frames/Frame'+str(i).zfill(3)+'.png')
        #plt.close()
        #print 'Saved frame ',i

plt.plot(errors)
plt.show()
#fig = plt.figure()
#ax = plt.gca(projection='3d')
#surf = ax.plot_trisurf(X, Y, uex, triangles=E, cmap=plt.cm.jet, linewidth=0.2)
#fig.colorbar(surf)
#plt.title(r'$\tilde{u}\left(x,y\right)$')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

#   fig = plt.figure()
#   ax = plt.gca(projection='3d')
#   surf = ax.plot_trisurf(X, Y, u, triangles=E, cmap=plt.cm.jet, linewidth=0.2)
#   fig.colorbar(surf)
#   plt.title('Numerical')
#   plt.title(r'$u\left(x,y\right)$')
#   plt.xlabel('x')
#   plt.ylabel('y')
#   plt.show()

#fig = plt.figure()
#ax = plt.gca(projection='3d')
#surf = ax.plot_trisurf(X, Y, uex-u, triangles=E, cmap=plt.cm.jet, linewidth=0.2)
#fig.colorbar(surf)
#plt.title('Difference')
#plt.show()
