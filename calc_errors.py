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


hbar, m = 1.0, 1.0
lx, ly = 1.0, 1.0
def uex(x, y, t=0):
    nxs = [1, 2]#, 1]
    nys = [1, 2]#, 2]
    phis = []
    for nx, ny in zip(nxs, nys):
        En = (((hbar*np.pi)**2)/(2*m*lx*ly))*(nx**2 + ny**2)
        phit = np.exp(-1j*t*En/hbar)
        N = (np.sqrt(2)/lx)*(np.sqrt(2)/ly)*(1./4)
        phi = N*np.sin(nx*x*np.pi)*np.sin(ny*y*np.pi)*phit
        phis.append(phi)
    return sum(phis).astype(np.complex128)


def ic(x, y):
    return uex(x, y)
    #return np.sin(x*np.pi)*np.sin(y*np.pi)+np.sin(2*x*np.pi)*np.sin(2*y*np.pi)


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



def solve(dx, dt, tf):
    mesh = gmsh.Mesh()

    # Order of quadrature and mesh to use.
    gauord = 5
    # 0 = .25
    # 1 = .125
    # 2 = .0625
    # 3 = .03125
    # 4 = .015625
    names = {.25:     "sqmesh0.msh",
             .125:    "sqmesh1.msh",
             .0625:   "sqmesh2.msh",
             .03125:  "sqmesh3.msh",
             .015625: "sqmesh4.msh"}
    mesh.read_msh(names[dx])

    E = mesh.Elmts[9][1]
    V = mesh.Verts[:,:2]

    ne = E.shape[0]
    nv = V.shape[0]
    X = V[:,0]
    Y = V[:,1]


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
            phis = np.array([[phi(i,r,s) for i in range(0,6)]])#.reshape(-1, 1)
            A2elem += w * detJ * (phis.T).dot(phis)

        AA2[ei,:] = A2elem.ravel()
        AA[ei, :] = Aelem.ravel()
        IA[ei, :] = np.repeat(K,6)
        JA[ei, :] = np.tile(K,6)


    # Assembly
    A = sparse.coo_matrix((AA.ravel(), (IA.ravel(), JA.ravel())))
    A = A.tocsr()
    A = A.tocoo()
    A2 = sparse.coo_matrix((AA2.ravel(), (IA.ravel(), JA.ravel())))
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
                A.data[k] = 1.0
            else:
                A.data[k] = 0.0
    """
    # this doesn't seem to do much, not sure if this is needed
    for k in range(0, len(A2.data)):
        i = A2.row[k]
        j = A2.col[k]
        if Dflag[i] or Dflag[j]:
            if i == j:
                A2.data[k] = 1.0
            else:
                A2.data[k] = 0.0
    """
    # Now solve (and correct from above)
    #A = A.tocsr()
    #print np.linalg.cond(A.todense())
    #u = sla.spsolve(A, b)
    #u = u + u0

    imagu = complex(0.0,1.0)
    nt = (tf/dt) + 1.0
    #print("nt: ",nt, int(nt))
    nt = int(nt)
    #dt = tf / ( nt - 1.0 )
    

    dummy = ( - imagu * hbar / ( 2.0 * m ) ) * A
    print("Done Building. Inverting")
    U = np.dot(sla.inv(A2-0.5*dt*dummy),A2+0.5*dt*dummy).tocoo()
    print("Done inverting Matrix. Propogating")
    psi0 = ic(X,Y)
    psi = np.array(psi0)
    pdens = np.real(psi * np.conj(psi))
    for i in range(0,nt+1):
        psi = U.dot(psi)
        pdens = np.real(psi * np.conj(psi))

    return X, Y, psi



"""
for num in range(2,5):
    name = "sqmesh"+str(num)+".msh"
    #name = "sqmesh.msh"

    print(name)
    mesh = gmsh.Mesh()
    mesh.read_msh(name)

    E = mesh.Elmts[9][1]
    V = mesh.Verts[:,:2]

    ne = E.shape[0]
    nv = V.shape[0]
    X = V[:,0]
    Y = V[:,1]

    def dist(r1, r2):
        return la.norm(r1-r2, 2)
    
    max_dist = 0
    min_dist = np.inf
    max_dists = []
    min_dists = []
    avg_dists = []
    for triangle in E[:, :3]:
        r1, r2, r3 = np.array([X[triangle], Y[triangle]]).T
        d0 = dist(r1, r2)
        d1 = dist(r1, r3)
        d2 = dist(r2, r3)
        maxd = max(d0, max(d1, d2))
        avgd = (d0 + d1 + d2)/3
        mind = min(d0, min(d1, d2))
        if mind < min_dist:
            min_dist = mind
        if max_dist < maxd:
            max_dist = maxd
    dx = min_dist

    print("num: ", num)
    print("min_dist: ", min_dist)
    print("max_dist: ", max_dist)
"""
    #print("dx: ",dx)

# approximate max_dist
# ""= .0625
# 0 = .25
# 1 = 
# 2 = .0625
# 3 = .03125
# 4 = .015625

tf = 1./np.pi
dxs = [.25]#, .0625, .03125]#, .015625]
dts = [tf*.5, tf*.1, tf*.05, tf*.01, tf*.005, tf*.001, tf*.0005, tf*.0001]
#dts = np.linspace(.01*tf, .1*tf, 4)[::-1]

dx_small, dt_small = dxs[-1], dts[-1]
errors_dx = []
errors_dt = []

#dt_small = tf*.0001
#dx_small = .03125


for dt in dts:
    print("solve: dt=",dt, dx_small)
    #X, Y, psi_calc = solve(dx_small, dt, tf)
    X, Y, psi_calc = solve(dx_small, dt, tf)
    pdens_calc = np.real(np.conj(psi_calc) * psi_calc)


    psi_true = uex(X, Y, tf)
    pdens_true = np.real(np.conj(psi_true) * psi_true)

    er = pdens_true - pdens_calc
    print("error: ",la.norm(er.flatten(), np.inf))
    print()
    # calculated infinity norm error
    errors_dt.append(la.norm(er.flatten(), np.inf))

print("dts, and errors_dt")
print(dts)
print(errors_dt)
"""

for dx in dxs:
    print("solve: dx=",dx, dt_small)
    X, Y, psi_calc = solve(dx, dt_small, tf)
    pdens_calc = np.real(np.conj(psi_calc) * psi_calc)


    psi_true = uex(X, Y, tf)
    pdens_true = np.real(np.conj(psi_true) * psi_true)

    er = pdens_true - pdens_calc
    print("er: ",la.norm(er.flatten(), np.inf))
    print()
    # calculated infinity norm error
    errors_dx.append(la.norm(er.flatten(), np.inf))


print("dx, errors_dx")
print(dxs)
print(errors_dx)
print("dxsmall, dtsmall")
print(dx_small, dt_small)


print("dts, and errors_dt")
print(dts)
print(errors_dt)
"""
"""
# errors for
dx_small = .03125
dt_small = .001
dts = [.5, .1, .05, .01, .005, .001, .0005, .0001]
dxs = [.25, .0625, .03125]
errors_dt = [0.11951913322551189,
             0.25502990412114501, 
             0.72675657194899546,
             0.22535568712856358,
             0.025475375714562487, 
             0.028496573076750353,
             0.0157386231969,
             0.00347921170951]

errors_dx = [0.44050776566376482, 
             0.030163971485563934, 
             0.028496573076750353]
"""

dxsmall, dtsmall = 0.0625, 0.003183098861837907
#dts, and errors_dt
"""dts = [0.03183098861837907, 0.015915494309189534, 0.003183098861837907, 0.0015915494309189536, 0.0003183098861837907, 0.00015915494309189535, 3.183098861837907e-05]
errors_dt = [0.037757947902033462, 0.00081088018233699266, 0.00093156711277153637, 0.00039315324002164864, 0.00017881441578604385, 0.00014682287407852268, 0.00013711099203805865]
"""
dxs = [0.25, 0.0625, 0.03125]
errors_dx = [0.0082592283015007206, 0.00093156711277153637, 0.00089900212711191418]

"""
dts = [0.03183098861837907, 0.015915494309189534, 0.003183098861837907, 0.0015915494309189536, 0.0003183098861837907, 0.00015915494309189535, 3.183098861837907e-05]
errors_dt = [0.15138477247485493, 0.0027862048134688067, 0.0035960085084334459, 0.0011064474533526436, 7.8985092501149268e-05, 6.4180386871282025e-05, 7.8332072611875248e-05]
"""
dt_small = 3.183098861837907e-05
dxs = [0.25, 0.0625, 0.03125]
errors_dx = [0.034183663150276766, 0.00054844396820330488, 7.8332072611875248e-05]
"""
dx_small, dt_small = 0.03125, 3.183098861837907e-05

# with dx=.0625
dts1 = [ 0.0031831,   0.0127324,   0.02228169,  0.03183099]
errors_dt1 =[0.0035960085084334459, 0.0079789111381609601, 0.046029500363681475, 0.15138477247485493]
"""
print("dx")
for i in range(len(dxs)-1):
    A = np.log(dxs[i])/np.log(dxs[i+1])
    B = np.log(errors_dx[i])/np.log(errors_dx[i+1])
    print(A/B)
print("dt")
for i in range(len(dts)-1):
    A = np.log(dts[i])/np.log(dts[i+1])
    B = np.log(errors_dt[i])/np.log(errors_dt[i+1])
    print(A/B)


fig = plt.figure()
#plt.title("Error in Infinite Square Well Problem")
ax = fig.add_subplot(1,2,1)
ax.set_title("Error vs $dt$, $dx$={0:.4f}, $tf$={1:.3f}".format(dx_small, tf))
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("$dt$")
ax.set_ylabel("$\|error\|_{\infty}$")
#ax.plot(dts1, errors_dt1)
ax.plot(dts, errors_dt)


ax = fig.add_subplot(1,2,2)
ax.set_title("Error vs $dx$, $dt$={0:.6f}, $tf$={1:.3f}".format(dt_small, tf))
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("$dx$")
ax.set_ylabel("$\|error\|_{\infty}$")
ax.plot(dxs, errors_dx)
plt.show()
#plt.savefig("all_errtfpi.png")



