import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pylab import *
from math import pi,cos
# Run parameters
nx = 20
ny = 20
nt = 500
lx = 1.0
ly = 1.0
tf = 1
dx = lx / ( nx - 1.0 )
dy = ly / ( ny - 1.0 )
dt = tf / ( nt - 1.0 )
x = np.linspace(0+dx,1-dx,nx)
y = np.linspace(0+dy,1-dy,ny)
X, Y = np.meshgrid(x, y)
m = 1
hbar = 1

# x, y \in [0, 1]^2

def uex(x, y, t=0):
    nx, ny = 3,2
    a = 1
    m = 1
    hbar = 1
    En = (((hbar*np.pi)**2)/(2*m*a*a))*(nx**2 + ny**2)
    phit = np.exp(-1j*t*En/hbar)
    return np.sin(nx*x*np.pi)*np.sin(ny*y*np.pi)*phit
    #elif nx == 2:
    #    return 0.5 * ( sin(pi*x)*sin(pi*y) + sin(2*pi*x)*sin(2*pi*y) )*phit





# Define the potential and the initial wavefunction
v = np.empty((nx,ny))
psi = np.empty((nx,ny),dtype=np.complex128)
for i in range(0,nx):
    for j in range(0,ny):
        v[i,j] = 0.0
        psi[i,j] = uex(x[i], y[j])


# 1d arrays
psi1d = psi.reshape((nx*ny),order='F')
v1d = v.reshape((nx*ny),order='F')

# Set up propagator
ix = np.eye(nx)
iy = np.eye(ny)
ex = np.ones(nx)
ey = np.ones(ny)
ident = np.identity(nx*ny,dtype=complex)
imag = complex(0.0,1.0)

Ax = (sp.spdiags([ex,-2*ex,ex],[-1,0,1],nx,nx,format='csr')/(dx*dx))
Ay = (sp.spdiags([ey,-2*ey,ey],[-1,0,1],ny,ny,format='csr')/(dy*dy))
A2D = (sp.kron(iy,Ax)+sp.kron(Ay,ix)).todense()

lhs = ident - ( ( imag * hbar * dt ) / ( 4.0 * m ) ) + ( ( imag * dt ) / ( 2.0 * hbar) ) * (ident.dot(v1d))
rhs = ident + ( ( imag * hbar * dt ) / ( 4.0 * m ) ) - ( ( imag * dt ) / ( 2.0 * hbar) ) * (ident.dot(v1d))

u = np.matmul(np.linalg.inv(lhs),rhs)

# Apply succesively
for i in range(0,nt):
    psi1d = u.dot(psi1d)
    # need the transpose here
    psi = psi1d.reshape((nx,ny),order='F').T
    pdens = np.real(psi * np.conj(psi))
    #print pdens
    ux = uex(X, Y, i*dt)
    error = la.norm((pdens - (np.conj(ux) * ux)).flatten(), np.inf)
    #print error
    plt.cla()
    plt.clf()
    im = plt.imshow(pdens,cmap=cm.RdBu,origin='lower',interpolation='bilinear',extent=[0,1,0,1],vmin=0,vmax=1)
    cb = plt.colorbar(im)
    plt.pause(0.001)
