import numpy as np
import scipy.sparse as sparse
import scipy.linalg as la
import scipy.sparse.linalg as spla
import gmsh
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from trigauss import trigauss
from basis import *



def integral(f, n=3):
    # integral on the triangle (0,0), (0,1), (1,0)
    nodes, weights = trigauss(n)
    y = 0
    for i in range(nodes.shape[0]):
        y += weights[i]*f(nodes[i, 0], nodes[i, 1])
    return y





mesh = gmsh.Mesh()
mesh.read_msh('sqmesh.msh') # coarse

hbar, m = 1, 1
lx, ly = 1, 1

## Only works for quadratic triangle elements
# degree = 1 num_bases = 3
# degree = 2 num_bases = 6
num_bases = 3
ind = 2 if num_bases == 3 else 9
E = mesh.Elmts[2][1]
V = mesh.Verts[:,:2]

ne = E.shape[0] #
nv = V.shape[0]
X = V[:,0].astype(np.complex128)      # X shape: nv x 1
Y = V[:,1].astype(np.complex128)


def uex(x, y, t=0):
    nx, ny = 3, 4
    En = (((hbar*np.pi)**2)/(2*m*lx*ly))*(nx**2 + ny**2)
    phit = np.exp(-1j*t*En/hbar)
    N = (np.sqrt(2)/lx)*(np.sqrt(2)/ly)
    return np.sin(nx*x*np.pi)*np.sin(ny*y*np.pi)*phit




# Create the LHS and RHS
AA = np.zeros((ne, num_bases**2), dtype= np.complex128)

IA = np.zeros((ne, num_bases**2), dtype= np.complex128)
JA = np.zeros((ne, num_bases**2), dtype= np.complex128)

for t in range(ne):
    K = E[t]
    def Jacobian(r, s):
        dphi = np.zeros((2, num_bases), dtype= np.complex128)
        J = np.zeros((2,2), dtype= np.complex128)
        for i in range(1, num_bases+1):
            dphi[0, i-1] = dphi_dr(i, r, s)
            dphi[1, i-1] = dphi_ds(i, r, s)
        dx = np.dot(dphi, X[K])
        dy = np.dot(dphi, Y[K])
        J[0, :] = dx
        J[1, :] = dy
        return J


    def g(r, s):
        dphi = np.zeros((2, num_bases), dtype= np.complex128)
        J = Jacobian(r, s)
        invJ = la.inv(J.T)
        detJ = la.det(J)
        for i in range(1,num_bases+1):
            dphi[0, i-1] = dphi_dr(i, r, s)
            dphi[1, i-1] = dphi_ds(i, r, s)
        dlambda = np.dot(invJ, dphi)

        return detJ*np.dot(dlambda.T, dlambda)


    A_local = 1j*integral(g)*(hbar/(2*m))


    # assign to global matrices
    AA[t, :] = A_local.ravel()

    for i in range(num_bases):
        for j in range(num_bases):
            IA[t, i*num_bases+j] = K[i]
            JA[t, i*num_bases+j] = K[j]



A = sparse.coo_matrix((AA.ravel(), (IA.ravel(), JA.ravel())))
A = A.tocsr()
A = A.tocoo()

# First flag the locations of the boundary 
tol = 1e-12
Dflag = np.logical_or.reduce((#abs(1-  np.sqrt(X**2 + Y**2)),
                              #abs(1-2*np.sqrt(X**2 + Y**2))
                              abs(1-np.abs(X)) < tol,
                              abs(1-np.abs(Y)) < tol,
                              abs(X) < tol,
                              abs(Y) < tol
                             ))
ID = np.where(Dflag)[0]
"""
# plot the boundary points to make sure they are right
plt.figure()
uc = np.ones(X.shape[0])
uc[ID] = 0
plt.scatter(X, Y, c=uc)
plt.show()
"""

# Then mark the diagonal as 1.0 and the off diagonals as 0.0 for each boundary vertex
for k in range(len(A.data)):
    i = A.row[k]
    j = A.col[k]
    if Dflag[i] or Dflag[j]:
        if i == j:
            A.data[k] = 1.0
        else:
            A.data[k] = 0.0


# A is the Hamiltonian
A = A.tocsr()
plot = True


if plot:
    fig = plt.figure()
nt = 60
tf = 100.0
# dt may be incorrect
dt = tf/(nt+1.0)
v = -1
errors = {"EB":[], "EF":[]}

EBM = sparse.eye(A.shape[0]).tocsc() - dt*A.tocsc()
inv_A = spla.splu(EBM)
#plt.figure()
for method in ["EB", "EF"]:
    u = uex(X, Y)
    for i in range(nt):
        ux = uex(X, Y, i*nt)
        pdist      = np.real(np.conj(u)*u)
        pdist_true = np.real(np.conj(ux)*ux)
        er = la.norm((pdist_true - pdist).flatten(), np.inf)
        errors[method].append(er)

        if plot:
            plt.clf()
            ax = fig.add_subplot(1,1,1)
            ax.set_title("Probability Distribution "+ method +" t={0:.2f}/{1:.2f}".format(dt*i, tf))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            surf = ax.tripcolor(X, Y, pdist, triangles=E[:,:num_bases], cmap=plt.cm.jet, linewidth=0.2)
            #fig.colorbar(surf)
            plt.pause(.001)

        # go to next step
        if method == "EF":
            # Euler Forward O(dt)
            u = u - dt*A.dot(u)
        else:
            # Euler Backward O(dt) but stable
            u = inv_A.solve(u)





plt.figure()
plt.title("Error of Probability Distribution")
plt.yscale("log")
for method in errors:
    if len(errors[method]) == nt:
        plt.plot(range(nt), errors[method], label=method)
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.legend()
plt.show()




"""
# 3d plot of distribution
ax = plt.gca(projection='3d')
ax.set_title("Probability Distribution t={0:.2f}/{1:.2f}".format(dt*i, 1))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("u")
print(X.shape, Y.shape, E.shape, pdist.shape)
print(X.shape == Y.shape)
surf = ax.tripcolor(X, Y, pdist, triangles=E[:,:num_bases], cmap=plt.cm.jet, linewidth=0.2)
#surf = ax.plot_trisurf(X, Y, pdist, triangles=E[:,:3], cmap=plt.cm.jet, linewidth=0.2)
fig.colorbar(surf)
plt.pause(1.0)
"""
