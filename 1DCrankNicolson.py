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
