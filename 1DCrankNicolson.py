"""
The code below was written by @author: https://github.com/DianaNtz and is an 
implementation of the 1D Crank Nicolson method. It solves in particular the 
Schrödinger equation for the quantum harmonic oscillator.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
filenames = []
#some initial values
hbar = 1.05e-34
mass = 9.11e-31 #mass electron
wz=110*2*np.pi*1000 #freuquency in Hz
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
psi=(1/(np.sqrt(2)))*(psi1+psi0)
#psi=psico
Vz=0.5*wz**2*z**2*mass
a=(1j*dt/hbar)/((4.0*mass*dz**2)/(hbar**2))
b=np.empty(nz, dtype=complex)
c=np.empty(nz, dtype=complex)
for i in range(0,nz):
    b[i]=1.0+0.5*dt/hbar*1j*(hbar**2/((dz**2*mass))+Vz[i])
    c[i]=1.0-0.5*dt/hbar*1j*(hbar**2/((dz**2*mass))+Vz[i])
#setting up matrices A and B
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
invA=np.linalg.inv(A)
#starting loop for time iteration
for i in range(0,time_steps+1):
    psis=(1/(np.sqrt(2)))*(psi0*np.exp(-1j*wz*t/2)+psi1*np.exp(-1j*3*wz*t/2))
    rohco=np.exp(-((z-z0*np.sqrt(2)*absalpha*np.cos(wz*t))/z0)**2)/(z0*np.sqrt(np.pi)) 
    ka=B.dot(psi)
    psi=invA.dot(ka)
    #creating gif animation with imageio
    if(i%10==0): 
          print(i)
          ax1 = plt.subplots(1, sharex=True, figsize=(10,5))          
          #plt.plot(z*10**3,rohco/10**(3),
          #color='black',linestyle='-',linewidth=3.0,label="$|\psi_{α} (z, t)|^2$")
          plt.plot(z*10**3,np.real(psis*np.conjugate(psis))/10**(3),
          color='black',linestyle='-',linewidth=3.0,label="$|\psi_{s} (z, t)|^2$")
          plt.plot(z*10**3,np.real(psi*np.conjugate(psi))/10**(3),
          color='deepskyblue',linestyle='-.',linewidth=3.0,label = "$|\psi (z, t)|^2$")
          plt.xlabel("Position in [mm]",fontsize=16) 
          plt.ylabel(r'Probability density [1/mm]',fontsize=16)
          plt.ylim([0,70])
          plt.xlim([zmin*10**3,zmax*10**3]) 
          plt.xticks(fontsize= 16)
          plt.xticks([-0.08,-0.04,0,0.04,0.08])
          plt.yticks(fontsize= 16) 
          plt.text(0.04, 56.5,
          "t=".__add__(str(round(t*10**(6),1))).__add__(" $\mu$s"),fontsize=19 )
          plt.legend(loc=2,fontsize=19,handlelength=3,frameon=False) 
          filename ='bla{0:.0f}.png'.format(i/10)
          filenames.append(filename)    
          plt.savefig(filename,dpi=250)
          plt.close()
    t=t+dt
with imageio.get_writer('superpositionstate.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)       
for filename in set(filenames):
    os.remove(filename)