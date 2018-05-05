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
import os

#os.system("scp jjdoher2@stout.cs.illinois.edu:~/solving_schrodinger/figs/final-sol-arr.npy .")
rc('text', usetex=True)
mesh = gmsh.Mesh()

# Order of quadrature and mesh to use.
gauord = 7
mesh.read_msh('pseudoslit.msh')

E = mesh.Elmts[9][1]
V = mesh.Verts[:,:2]

ne = E.shape[0]
nv = V.shape[0]
X = V[:,0]
Y = V[:,1]


large_pdens = np.load("final-sol-arr-low-res.npy")

(u, v) = large_pdens.shape

nt = u-1
tf = 2.0
dt = tf / ( nt - 1.0 )
fig = plt.figure()
for i in range(nt+1):
    plt.clf()
    triang = tri.Triangulation(X,Y)
    surf = plt.tripcolor(X,Y,large_pdens[i], triangles=E[:,:3], cmap=plt.cm.viridis,linewidth=0.2,vmin=0,vmax=1)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('$|\psi|^{2}$ at t='+str(i*dt))
    fig.colorbar(surf)
    fig.tight_layout()
    plt.pause(.000000001)
