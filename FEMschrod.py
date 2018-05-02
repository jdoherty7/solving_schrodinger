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
gauord = 5
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
BB = np.zeros((ne, 36))
IB = np.zeros((ne, 36))
JB = np.zeros((ne, 36))
bb = np.zeros((ne, 6))
ib = np.zeros((ne, 6))
jb = np.zeros((ne, 6))
qx,qw = trigauss(gauord)

# Main loop
for ei in range(0,ne):
	Aelem = np.zeros((6,6))
	A2elem = np.zeros((6,6))
	Belem = np.zeros(6)
	K = E[ei,:]
	x0, y0 = X[K[0]], Y[K[0]]
	x1, y1 = X[K[1]], Y[K[1]]
	x2, y2 = X[K[2]], Y[K[2]]
	x3, y3 = X[K[3]], Y[K[3]]
	x4, y4 = X[K[4]], Y[K[4]]
	x5, y5 = X[K[5]], Y[K[5]]
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
		A2elem += w * detJ * ((np.array([phi(i,r,s) for i in range(0,6)])).T).dot((np.array([phi(i,r,s) for i in range(0,6)])))
		Belem += w * detJ * np.array([phi(i,r,s) for i in range(0,6)])
	AA2[ei,:] = A2elem.ravel()
	AA[ei, :] = Aelem.ravel()
	IA[ei, :] = np.repeat(K,6)
	JA[ei, :] = np.tile(K,6)
	bb[ei, :] = Belem.ravel()
	ib[ei, :] = K
	jb[ei, :] = 0

# Assembly
A = sparse.coo_matrix((AA.ravel(), (IA.ravel(), JA.ravel())))
A = A.tocsr()
A = A.tocoo()
A2 = sparse.coo_matrix((AA2.ravel(), (IA.ravel(), JA.ravel())))
A2 = A2.tocsr()
A2 = A2.tocoo()
b = sparse.coo_matrix((bb.ravel(), (ib.ravel(), jb.ravel())))
b = b.tocsr()
b = np.array(b.todense()).ravel()

# Boundary conditions
tol = 1e-6
gflag = np.logical_or.reduce((abs(X) < tol,
			      abs(Y) < tol,
                              abs(X-1.0) < tol,
                              abs(Y-1.0) < tol))
Dflag = gflag
ID = np.where(Dflag)[0]
Ig = np.where(gflag)[0]
u0 = np.zeros((nv,))
u0[Ig] = 0
b = b - A * u0
b[ID] = 0.0

for k in range(0, len(A.data)):
    i = A.row[k]
    j = A.col[k]
    if Dflag[i] or Dflag[j]:
        if i == j:
            A.data[k] = 1.0
        else:
            A.data[k] = 0.0

# Now solve (and correct from above)
#A = A.tocsr()
#print np.linalg.cond(A.todense())
#u = sla.spsolve(A, b)
#u = u + u0

def ic(x, y):
    return np.sin(x*np.pi)*np.sin(y*np.pi)+np.sin(2*x*np.pi)*np.sin(2*y*np.pi)

imagu = complex(0.0,1.0)
nt = 200
tf = 5
dt = tf / ( nt - 1.0 )
A = np.array(A.todense())
A2 = np.array(A2.todense())
hbar = 1
m = 1
dummy = ( - imagu * hbar / ( 2 * m ) ) * A
U = np.matmul(np.linalg.inv(A2-0.5*dt*dummy),A2+0.5*dt*dummy)
print A
print A2
print U
psi0 = ic(X,Y)
psi = np.array(psi0)
for i in range(0,nt):
	psi = U.dot(psi)
	pdens = np.real(psi * np.conj(psi))
	if(1):
		triang = tri.Triangulation(X,Y)
		fig = plt.figure()
		surf = plt.tripcolor(X,Y,pdens, triangles=E[:,:3], cmap=plt.cm.viridis,linewidth=0.2,vmin=0,vmax=3)
		plt.tricontour(triang, pdens, colors='k',vmin=0,vmax=3)
		plt.xlabel('$x$')
		plt.ylabel('$y$')
		plt.title('$|\psi|^{2}$')
		fig.colorbar(surf)
		fig.tight_layout()
		plt.savefig('Frames/Frame'+str(i).zfill(3)+'.png')
		plt.cla()
	        plt.clf()
		plt.close()
		print 'Saved frame ',i

#fig = plt.figure()
#ax = plt.gca(projection='3d')
#surf = ax.plot_trisurf(X, Y, uex, triangles=E, cmap=plt.cm.jet, linewidth=0.2)
#fig.colorbar(surf)
#plt.title(r'$\tilde{u}\left(x,y\right)$')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

#	fig = plt.figure()
#	ax = plt.gca(projection='3d')
#	surf = ax.plot_trisurf(X, Y, u, triangles=E, cmap=plt.cm.jet, linewidth=0.2)
#	fig.colorbar(surf)
#	plt.title('Numerical')
#	plt.title(r'$u\left(x,y\right)$')
#	plt.xlabel('x')
#	plt.ylabel('y')
#	plt.show()

#fig = plt.figure()
#ax = plt.gca(projection='3d')
#surf = ax.plot_trisurf(X, Y, uex-u, triangles=E, cmap=plt.cm.jet, linewidth=0.2)
#fig.colorbar(surf)
#plt.title('Difference')
#plt.show()
