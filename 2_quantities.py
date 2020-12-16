
""" 2 : Ising quantities | Version: 2020-12-15 """

"""

This example is intended to show how to obtain several quantities of an
Ising system over a range of temperatures each one with n-steps using MMC.

"""

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 1: Load libraries (for 2D use Ising2D)

import numpy as np
import matplotlib.pyplot as plt
from lib.ising import Ising
import time

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2: System Initial Conditions

Lx = 32 # Number of sites in x-dimension
Ly = 32 # Number of sites in y-dimensions [Ly=1]
Lz = 32 # Number of sites in z-dimensions [Lz=1] 
Ps = .5 # Spin-up probability in each site when instantiating system [Ps=.5]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 3: Boundary and Markov conditions

n_steps = int(1e3) # Number of n-steps to run over MMC
n_stats = round(0) # Number of n-steps to subtract for statistics

T_lower = 0.1 # Minimum temperature sampling
T_upper = 4.5 # Maximum temperature sampling
T_critical = 2/np.log(1+np.sqrt(2)) # 2D Critical: kTc = 2J/log(1+sqrt(2))
T_critical_interval = np.sqrt(T_critical)/2

T_steps_lower = int(1e1) # Number of T steps in the lower range
T_steps_critical = int(1e2) # Number of T steps in the critical range
T_steps_upper = int(1e1) # Number of T steps in the upper range

# OPTIONAL: If you want to generate a normalized distribution around T_critical
# from scipy.stats import norm
# T_steps = int(1e2+20)
# T_critical = 2/np.log(1+np.sqrt(2))
# T_scale = np.sqrt(T_critical/2)
# T = np.asarray([Ti for Ti in norm.ppf(np.linspace(0, 1, T_steps)[1:-1],T_critical,T_scale) if Ti >= 0])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 4: Initialize Ising class and estimate quantities over temperature

init_time = time.time()

MyIsing = Ising(Lx, Ly, Lz)

S = MyIsing.setS(Ps)

T = MyIsing.rangeTemperature(T_lower, T_upper, T_critical, T_critical_interval, T_steps_lower, T_steps_critical, T_steps_upper)

E_mean, E_std, M_mean, M_std, M_abs_mean, M_abs_std = MyIsing.runOverTemperature(S, T, n_steps, n_stats)

C = MyIsing.specificHeat(T, E_std)
X = MyIsing.susceptibility(T, M_std)

print(time.time()-init_time)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5: Save results in 'data' folder

f_date = 2020_12_15
f_name = 'ising_mean_Lx'+str(Lx)+'_Ly'+str(Ly)+'_Lz'+str(Lz)+'_n'+str(n_steps)+'_id'+str(f_date)
np.save('data/'+f_name+'.npy', [T, E_mean, E_std, M_mean, M_std, M_abs_mean, M_abs_std, C, X])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 6: Load results from 'data' folder

f_name = 'ising_mean_Lx16384_Ly16384_n10_id20201215_Ps_method'
[T, E_mean, E_std, M_mean, M_std, M_abs_mean, M_abs_std, C, X] = np.load('data/'+f_name+'.npy')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 7: Mean energy per site as a function of temperature

plt.plot(T, E_mean, '.k')
plt.ylabel('Mean Energy per site')
plt.xlabel('Temperature')
plt.axvline(T_critical, ls='--', color='#888')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 8: Mean magnetization per site as a function of temperature

plt.plot(T, M_mean, '.k')
plt.ylabel('Mean Magnetization per site')
plt.xlabel('Temperature')
plt.axvline(T_critical, ls='--', color='#888')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 9: Mean absolute magnetization per site as a function of temperature

plt.plot(T, M_abs_mean, '.k')
plt.ylabel('Mean Absolute Magnetization per site')
plt.xlabel('Temperature')
plt.axvline(T_critical, ls='--', color='#888')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 10: Specific heat as a function of temperature

plt.plot(T, C, '.k')
plt.ylabel('Specific Heat')
plt.xlabel('Temperature')
plt.axvline(T_critical, ls='--', color='#888')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 11: Magnetic susceptibility as a function of temperature

plt.plot(T, X, '.k')
plt.ylabel('Magnetic Susceptibility')
plt.xlabel('Temperature')
plt.axvline(T_critical, ls='--', color='#888')