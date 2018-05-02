import numpy as np
import scipy.sparse as sparse
import scipy.linalg as la
import scipy.sparse.linalg as spla
import gmsh
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from trigauss import trigauss
from basis import *



def integral(f, n=6):
    # integral on the triangle (0,0), (0,1), (1,0)
    nodes, weights = trigauss(n)
    y = 0
    for i in range(nodes.shape[0]):
        y += weights[i]*f(nodes[i, 0], nodes[i, 1])
    return y





mesh = gmsh.Mesh()
mesh.read_msh('sqmesh.msh') # coarse

hbar, m = 1.0, 1.0
lx, ly = 1.0, 1.0

## Only works for quadratic triangle elements
# degree = 1 num_bases = 3
# degree = 2 num_bases = 6
num_bases = 3
elem = "linear" if num_bases == 3 else "quadratic"
ind = 2 if num_bases == 3 else 9
E = mesh.Elmts[ind][1]
V = mesh.Verts[:,:2]

ne = E.shape[0] #
nv = V.shape[0]
X = V[:,0]      # X shape: nv x 1
Y = V[:,1]


def uex(x, y, t=0):
    nxs = [1, 2, 2]
    nys = [1, 1, 2]
    phis = []
    for nx, ny in zip(nxs, nys):
        En = (((hbar*np.pi)**2)/(2*m*lx*ly))*(nx**2 + ny**2)
        phit = np.exp(-1j*t*En/hbar)
        N = (np.sqrt(2)/lx)*(np.sqrt(2)/ly)
        phi = np.sin(nx*x*np.pi)*np.sin(ny*y*np.pi)*phit
        phis.append(phi)
    return sum(phis).astype(np.complex128)


# Create the LHS and RHS
AA = np.zeros((ne, num_bases**2), dtype= np.complex128)
BB = np.zeros((ne, num_bases**2), dtype= np.complex128)

IA = np.zeros((ne, num_bases**2))
JA = np.zeros((ne, num_bases**2))
IB = np.zeros((ne, num_bases**2))
JB = np.zeros((ne, num_bases**2))

for t in range(ne):
    K = E[t]

    def Jacobian(r, s):
        dphi = np.zeros((2, num_bases))
        J = np.zeros((2,2))
        for i in range(1, num_bases+1):
            dphi[0, i-1] = dphi_dr(i, r, s, elem=elem)
            dphi[1, i-1] = dphi_ds(i, r, s, elem=elem)
        dx = np.dot(dphi, X[K[:num_bases]])
        dy = np.dot(dphi, Y[K[:num_bases]])
        J[0, :] = dx
        J[1, :] = dy
        return J

    # matrix corresponding to the spatial derivatives
    def a(r, s):
        dphi = np.zeros((2, num_bases))
        J = Jacobian(r, s)
        invJ = la.inv(J.T)
        detJ = la.det(J)
        for i in range(1,num_bases+1):
            dphi[0, i-1] = dphi_dr(i, r, s, elem=elem)
            dphi[1, i-1] = dphi_ds(i, r, s, elem=elem)
        dlambda = np.dot(invJ, dphi)

        return detJ*np.dot(dlambda.T, dlambda)

    # matrix multiplying the derivative
    def b(r, s):
        phis = np.zeros((2, num_bases))
        J = Jacobian(r, s)
        invJ = la.inv(J.T)
        detJ = la.det(J)
        for i in range(1,num_bases+1):
            phis[0, i-1] = phi(i, r, s, elem=elem)
            phis[1, i-1] = phi(i, r, s, elem=elem)
        lambdas = phis#np.dot(invJ, phis)
        return np.dot(lambdas.T, lambdas)



    # this is right hand side of 
    # ih dt u = -h*h/2/m dxx u
    A_local = integral(a)
    B_local = integral(b)
    #A_local = np.dot(la.inv(B_local), A_local)

    #A_local = -2j*A_local*(hbar/(2.0*m))

    # assign to global matrices
    AA[t, :] = A_local.ravel()
    BB[t, :] = B_local.ravel()

    for i in range(num_bases):
        for j in range(num_bases):
            IA[t, i*num_bases+j] = K[i]
            JA[t, i*num_bases+j] = K[j]
            IB[t, i*num_bases+j] = K[i]
            JB[t, i*num_bases+j] = K[j]


A = sparse.coo_matrix((AA.ravel(), (IA.ravel(), JA.ravel())))
A = A.tocsr()
A = A.tocoo()

B = sparse.coo_matrix((BB.ravel(), (IB.ravel(), JB.ravel())))
B = B.tocsr()
B = B.tocoo()

# First flag the locations of the boundary 
tol = 1e-12
Dflag = np.logical_or.reduce((abs(1-np.abs(X)) < tol,
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
#A = A.tocsr()
#-2j*A_local*(hbar/(2.0*m))
A = -1j*(hbar/(2.0*m))*np.dot(spla.inv(B), A)
def convert(A):
    B = []
    for i in range(A.shape[0]):
        new_row = []
        for j in range(A.shape[1]):
            if abs(A[i,j]-1.0) < 1e-3:
                new_row.append(255)
            else:
                new_row.append(0)
        B.append(new_row)
    return np.array(B)

plot = True
"""
plt.figure()
plt.title("Real")
Ad = A.todense()


plt.imshow(convert(Ad.real))
plt.colorbar()

plt.figure()
plt.title("Imaginary")
plt.imshow(convert(Ad.imag))
plt.colorbar()

plt.show()
"""

if plot:
    fig = plt.figure()
nt = 100
tf = .01
# dt may be incorrect
dt = tf/(nt+1.0)
errors = {"EB":[], "EF":[]}

EBM = sparse.eye(A.shape[0]) - dt*A
inv_A = spla.splu(EBM)
#plt.figure()
for method in ["EF"]:
    u = uex(X, Y)
    print(u.shape)
    for i in range(nt):
        ux = uex(X, Y, i*nt)
        pdist      = np.real(np.conj(u)*u)
        pdist_true = np.real(np.conj(ux)*ux)
        er = la.norm((pdist_true - pdist).flatten(), np.inf)
        errors[method].append(er)

        if plot and (i%10==0):
            plt.clf()
            ax = fig.add_subplot(1,1,1)
            ax.set_title("Probability Distribution "+ method +" t={0:.3f}/{1:.3f}".format(dt*i, tf))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            surf = ax.tripcolor(X, Y, u.real, triangles=E[:,:num_bases], cmap=plt.cm.jet, linewidth=0.2)
            fig.colorbar(surf)
            plt.pause(.001)

        # go to next step
        if method == "EF":
            # Euler Forward O(dt)
            u = u - dt*A.dot(ux)
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
