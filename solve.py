
# coding: utf-8

# Unsteady advection-diffusion equation
# ===
# 
# ## Case 1
# 
# $$ u_t + c\cdot\nabla u = \nabla\cdot\nu \nabla u $$
# on the domain $\left[-1,1\right]^2$. 
# 
# Starting from weak form,
# $$
#   v^T B u_t + \nu v^T A u = - v^T B (c_x D_x u+c_y D_y u)
# $$

import numpy as np; la =np.linalg
import scipy as sp
import matplotlib.pyplot as plt


# ### SG setup

exec(open('semsetup.py').read())

def bdfext_setup():    # BDF/EXT3
    al = np.zeros((3,3))
    bt = np.zeros((3,4))
    al[0,0] = 1.
    al[1,0] = 2.
    al[1,1] = -1.
    al[2,0] = 3.
    al[2,1] = -3.
    al[2,2] = 1.
    bt[0,0] = 1.
    bt[0,1] = -1.
    bt[1,0] = 3./2.
    bt[1,1] = -2.
    bt[1,2] = 1./2.
    bt[2,0] = 11./6.
    bt[2,1] = -3.
    bt[2,2] = 3./2.
    bt[2,3] = -1./3.
    return al, bt


import scipy.linalg as sl
import time
from mpl_toolkits.mplot3d import axes3d

def uex(x, y, t=0):
    # for a particle in a box, infinite potential
    n = 1
    a = 2
    m = 1
    hbar = 1
    phit = np.exp(-1j*t*(hbar*(n*np.pi)**2)/(2*m*a*a))
    return np.cos(x*np.pi/2)*np.cos(y*np.pi/2)*phit

def uinit(x,y):
    # Using an initial condition of only in the first energy level
    u = np.cos(np.pi*x/2)*np.cos(np.pi*y/2)
    ud = 0.                        # u = ud on Dirichlet, 0 bc V = inf
    bctyp = 3                      # 3: DDDD
    return u, ud, bctyp

# p is the discretization in space
# nt is the number of timesteps
# T is the final time to solve up to
# if verbose=True will create a real time graph of the solution
# as well as print the maximum error or the real and imaginary components
# at each time step
def advdif(p,T,nt,nplt, verbose=True):
    hbar = 1
    m = 1
    nu   = (1j * hbar)/(2.0*m)
    ah,bh,ch,dh,z,w = semhat(p)

    n1 = p + 1
    x = z
    h = z[1]-z[0]

    [X,Y] = np.meshgrid(x,x)


    X = X.T
    Y = Y.T
    u0, ud, bctyp = uinit(X,Y)
    # BC
    I1 = sp.sparse.eye(n1).tocsr()
    if(bctyp==1): # DDNN
        Rx = I1[1:-1,:].toarray() # Dirichlet left and right, Neumann top/bottom
        Ry = I1.toarray()
    elif(bctyp==2): # DNNN
        Rx = I1[1:,:].toarray() # Dirichlet left, Neumann everywhere
        Ry = I1.toarray()
    elif(bctyp==3): # DDDD
        Rx = I1[1:-1,:].toarray()
        Ry = I1[1:-1,:].toarray()
    elif(bctyp==4): # DNDN  -- D left/bottom
        Rx = I1[1:,:].toarray()
        Ry = I1[1:,:].toarray()
    else:
        Rx = I1.toarray() # Neumann everywhere?
        Ry = I1.toarray()
    Ryt = Ry.T

    # FastDiagM setup
    Ax = Rx.dot(ah).dot(Rx.T)
    Bx = Rx.dot(bh).dot(Rx.T)
    Ay = Ry.dot(ah).dot(Ry.T)
    By = Ry.dot(bh).dot(Ry.T)
    wy,vy = sl.eigh(Ay,By) # Equation in direction y
    wx,vx = sl.eigh(Ax,Bx) # Equation in direction x
    vyt = vy.T
    vxt = vx.T
    ry = wy.shape[0]
    rx = wx.shape[0]
    sy = np.ones(ry)
    sx = np.ones(rx)


    Dh1 = wy + wx[:,np.newaxis]   # reshaped lambda x I + I x lambda
    Dh2 = sy + 0.*sx[:,np.newaxis] # reshaped I x I

    dt = T/float(nt)
    dti = 1./dt
    al, bt = bdfext_setup()
    ndt = int(nt/nplt)
    if(ndt==0):
        ndt = 1
    ons = ud * np.ones(u0.shape)
    ub = ons - Rx.T.dot(Rx.dot(ons).dot(Ryt)).dot(Ry) # u=ud on D bndry
    u = Rx.T.dot(Rx.dot(u0).dot(Ry.T)).dot(Ry) + ub
    f1 = np.zeros(Rx.dot(u0.reshape((n1,n1))).dot(Ryt).shape)
    f2 = f1.copy()
    f3 = f2.copy()
    fb1 = f1.copy()
    fb2 = f1.copy()
    fb3 = f1.copy()

    t = 0.

    def myplot(u, t, fig=None, initial=False):
        # Booleans to Graph the Error and Graph the Probability Distribution
        err = 1
        graph_prob=True

        if err:
            u0 = u
            ux = uex(X, Y, t)
            error_real = np.abs(np.abs(ux.real) - np.abs(u.real))
            error_imag = np.abs(np.abs(ux.imag) - np.abs(u.imag))
            #u.real = error_real
            #u.imag = error_imag
            u = error_real + error_imag*1j
        if graph_prob:
            r, c = 2, 4
        else:
            r, c = 2,2
        if initial or fig is None:
            # Plot initial field
            fig = plt.figure(figsize=(12,12))
        else:
            plt.clf()
        ax1 = fig.add_subplot(r,c,1)
        surf = ax1.contourf(X,Y,u.real)
        fig.colorbar(surf)
        ax1.set_title('Real t=%f'%t)
        ax = fig.add_subplot(r,c,2,projection='3d')
        wframe = ax.plot_wireframe(X, Y, u.real)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('u')

        # plot complex
        ax1 = fig.add_subplot(r,c,3)
        surf = ax1.contourf(X,Y,u.imag)
        fig.colorbar(surf)
        ax1.set_title('Imaginary t=%f'%t)
        ax = fig.add_subplot(r,c,4,projection='3d')
        wframe = ax.plot_wireframe(X, Y, u.imag)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('u')

        if graph_prob:
            if err:
                prob = np.conj(ux)*ux - np.conj(u0)*u0
            else:
                prob = np.conj(u)*u
            #print(np.max(prob))
            # plot probability distribution
            ax1 = fig.add_subplot(r,c,5)
            surf = ax1.contourf(X,Y,prob)
            fig.colorbar(surf)
            ax1.set_title('Probability Distribution t=%f'%t)
            ax = fig.add_subplot(r,c,6,projection='3d')
            wframe = ax.plot_wireframe(X, Y, prob)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('u')

        plt.pause(0.06)
        return fig


    er, es = [], []
    ux = uex(X, Y, t)
    error_real = la.norm(ux.real.flatten() - u.real.flatten(), np.inf)
    error_imag = la.norm(ux.imag.flatten() - u.imag.flatten(), np.inf)
    if verbose:
        fig = myplot(u, t, initial=True)
        print()
        print("Time=",t)
        print("Total Probability: ", np.sum(np.conj(u)*u)*(h**2))
        print("Error (real, imag): (", error_real, ",", error_imag, ")")
    er.append(error_real)
    es.append(error_imag)
    for i in range(nt): # nt
        if(i<=2):
            ali = al[i,:]
            bti = bt[i,:]


        # Mass matrix, backward diff
        fb1 = Rx.dot(bh.dot(u).dot(bh.T)).dot(Ryt)
        # RHS, Everything
        f = - (dti)*( bti[1]*fb1 + bti[2]*fb2 + bti[3]*fb3 )\
                + ali[0]*f1 + ali[1]*f2 + ali[2]*f3\
                - nu * Rx.dot(ah.dot(ub).dot(bh)+bh.dot(ub).dot(ah.T)).dot(Ryt)
        # Save old states
        fb3 = fb2.copy()
        fb2 = fb1.copy()
        f3 = f2.copy()
        f2 = f1.copy()
        # Set up FDM solve
        h1 = nu
        h2 = bti[0] * dti
        Dl = h1 * Dh1 + h2 * Dh2
        # # FDM solve
        ug = vx.dot(np.divide(vxt.dot(f).dot(vy),Dl)).dot(vyt)
        u  = Rx.T.dot(ug).dot(Ry) + ub
        t  = float(i+1)*dt
        
        if((i+1)%ndt==0 or i==nt-1):
            ux = uex(X, Y, t)
            error_real = la.norm(ux.real.flatten() - u.real.flatten(), np.inf)
            error_imag = la.norm(ux.imag.flatten() - u.imag.flatten(), np.inf)
            if verbose:
                fig = myplot(u, t, fig)
                print()
                print("Time=",t)
                print("Total Probability: ", np.sum(np.conj(u)*u)*(h**2))
                print("Error (real, imag): (", error_real, ",", error_imag, ")")
                #print('t=%f, umin=%g, umax=%g'%(t,np.amin(u),np.amax(u)))
            er.append(error_real)
            es.append(error_imag)
    
    print(p, nt, er[-1], es[-1])
    plt.figure()
    plt.title("Error "+str(p)+" "+str(nt))
    plt.plot(er, label="Real Error")
    plt.plot(es, label="Imaginary Error")
    plt.legend()
    plt.savefig("figs/Error "+str(p)+" "+str(nt)+".png")

    succ = 0
    return succ


# nu=1 is a steady state, stops at 1e-2
# need p to be odd in order to be stable
# p=27 and nt = 2000 works
# if p / nt is too large it blows up again
# however, its not stable for longer time periods
p    = 25
T    = .1
nt   = 90
nplt = 30
succ = advdif(p,T,nt,nplt)
"""
for p in [10, 11, 26, 27, 44, 45]:
    for nt in np.power(10, np.array([1,2,3,4])):
        succ = advdif(p,T,nt,nplt, verbose=False)

"""