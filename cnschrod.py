import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pylab import *
from math import pi,cos
# Run parameters
nx = 35
ny = 35
nt = 5000
lx = 1.0
ly = 1.0
tf = 1
dx = lx / ( nx - 1.0 )
dy = ly / ( ny - 1.0 )
dt = tf / ( nt - 1.0 )
x = np.linspace(0+dx,1-dx,nx)
y = np.linspace(0+dy,1-dy,ny)
m = 1
hbar = 1

# Define the potential and the initial wavefunction
v = np.empty((nx,ny))
psi = np.empty((nx,ny),dtype=complex)
for i in range(0,nx):
	for j in range(0,ny):
		v[i,j] = 0.0
		psi[i,j] = 0.5 * ( sin(pi*x[i])*sin(pi*y[j]) + sin(2*pi*x[i])*sin(2*pi*y[j]) )

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
	psi = psi1d.reshape((nx,ny),order='F')
	pdens = np.real(psi * np.conj(psi))
	print pdens
	plt.cla()
        plt.clf()
	im = plt.imshow(pdens,cmap=cm.RdBu,origin='lower',interpolation='bilinear',extent=[0,1,0,1],vmin=0,vmax=1)
	cb = plt.colorbar(im)
	plt.pause(0.001)
