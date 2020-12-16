
""" 1 : Ising MMC | Version: 2020-12-15 """

"""

This example is intended to show the behaviour of an Ising system under a
run over n-steps using MMC algorithm.

"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 1: Load libraries (for 2D use Ising2D)

import numpy as np
import matplotlib.pyplot as plt
from lib.ising import Ising

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2: System Initial Conditions

Lx = 32 # Number of sites in x-dimension
Ly = 32 # Number of sites in y-dimensions [Ly=1]
Lz = 32 # Number of sites in z-dimensions [Lz=1] 
Ps = .5 # Spin-up probability in each site when instantiating system [Ps=.5]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 3: Boundary and Markov conditions

T = 2.0 # Fixed temperatura (2D Critical: kTc = 2J/log(1+sqrt(2)))
n_steps = int(1e2) # Number of n-steps to run over MMC
n_stats = round(n_steps/3) # Number of n-steps to subtract for statistics

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 4: Load Ising class and a system with initial probability Ps

MyIsing = Ising(Lx, Ly, Lz)

S = MyIsing.setS(Ps)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5: View grid for 2D (only) systems

plt.pcolor(S[:,:,0])
plt.clim(-1,1)
plt.axis('square')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 6: Execute MMC over n-steps and return values of interest

S, E, M = MyIsing.runOverMMC(S, T, n_steps)

E_mean, E_std, M_mean, M_std, M_abs_mean, M_abs_std = MyIsing.isingStats(E, M, n_stats)

C = MyIsing.specificHeat(T,E_std)
X = MyIsing.susceptibility(T,M_abs_std)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 7: Energy per site as a function of the n-step

plt.plot(E)
plt.plot(np.cumsum(E)/range(1,n_steps+1),'r--')
plt.axvline(n_stats,ls='--',color='k')
plt.title(rf'$\langle e \rangle = {round(E_mean,4)}$')
plt.xlabel('Number of Step')
plt.ylabel('Energy per site')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 8: Magnetization per site as a function of the n-step

plt.plot(M)
plt.plot(np.cumsum(M)/range(1,n_steps+1),'r--')
plt.axvline(n_stats,ls='--',color='k')
plt.title(rf'$\langle m \rangle = {round(M_mean,4)}$')
plt.xlabel('Number of Step')
plt.ylabel('Magnetization per site');
plt.ylim(top=1.1,bottom=-1.1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 9: Absolute magnetization per site as a function of the n-step

plt.plot(abs(M))
plt.plot(np.cumsum(abs(M))/range(1,n_steps+1),'r--')
plt.axvline(n_stats,ls='--',color='k')
plt.title(rf'$\langle m \rangle = {round(M_abs_mean,4)}$')
plt.xlabel('Number of Step')
plt.ylabel('Absolute Magnetization per site');
plt.ylim(top=1.1,bottom=-0.1)