"""
@author: Diana Nitzschke
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
hbar = 1.05e-34
mass = 9.11e-31 # mass electron
wz=110*2*np.pi*1000 # freuquency in Hz
z0=np.sqrt(hbar/(wz*mass))
nz=2**9
dt=1*10**(-8) #dt in s
tfinal=2*np.pi/wz 
time_steps=int(np.round(tfinal/dt))
t=0
zmax=6.6*z0  #zmax in m
zmin=-6.6*z0 #zmin in m
dz=(zmax-zmin)/(nz-1)
n=np.linspace(-nz/2,nz/2-1,nz)
z=n*dz
psi0=np.exp(-((z)/z0)**2/2)
norm0=np.sum(psi0*np.conjugate(psi0))*dz
psi0=psi0/np.sqrt(norm0)
psi1=np.exp(-((z)/z0)**2/2)*2*z/z0
norm1=np.sum(psi1*np.conjugate(psi1))*dz
psi1=psi1/np.sqrt(norm1)
absalpha=2.5
psico=np.exp(-((z-z0*np.sqrt(2)*absalpha)/z0)**2/2)
normco=np.sum(psico*np.conjugate(psico))*dz
psico=psico/np.sqrt(normco)
#psi=(1/(np.sqrt(2)))*(psi1+psi0)
psi=psico
Vz=0.5*wz**2*z**2*mass
a=(1j*dt/hbar)/((4.0*mass*dz**2)/(hbar**2))
b=np.empty(nz, dtype=complex)
c=np.empty(nz, dtype=complex)
for i in range(0,nz):
    b[i]=1.0+0.5*dt/hbar*1j*(hbar**2/((dz**2*mass))+Vz[i])
    c[i]=1.0-0.5*dt/hbar*1j*(hbar**2/((dz**2*mass))+Vz[i])
A=np.empty([nz, nz], dtype=complex)
B=np.empty([nz, nz], dtype=complex)
for i in range(0,nz):
    for j in range(0,nz):
        if   (j==i):
            A[i][j]=b[i]
            B[i][j]=c[i]
        elif (j==i+1):
            A[i][j]=-a
            B[i][j]=a
        elif (j==i-1):
            A[i][j]=-a
            B[i][j]=a
        else:
            A[i][j]=0.0
            B[i][j]=0.0 
