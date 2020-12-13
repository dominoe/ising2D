
""" 02_ising2d_measures | Version: 2020-12-12 """

#%%%%%%%%%%%%%%%%%%%%
# 1: Cargar librerías

import numpy as np
import matplotlib.pyplot as plt
from lib.ising2d import Ising2D

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2: Condiciones iniciales del sistema

Lx = 16384 # Número de sitios en la dimensión x
Ly = 16384 # Número de sitios en la dimensión y
Ps = .5 # Probabilidad inicial en la cadena de que un sitio sea spin-up

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 3: Condiciones de contorno y de Markov

n_steps = int(1e1) # Número de pasos en la cadena de Markov
n_stats = round(n_steps/2) # Cantidad de n-pasos a remover para estadística

T_lower = 0.1 # Minimum temperature sampling
T_upper = 4.5 # Maximum temperature sampling
T_critical = 2.269 # Critical Temperature 2D : kT_c/J=-2/\log(\sqrt{2}-1)\simeq 2.269185314213022
T_critical_interval = np.sqrt(T_critical)/2

T_steps_lower = int(1e1) # Number of T steps in the lower range
T_steps_critical = int(1e2) # Number of T steps in the critical range
T_steps_upper = int(1e1) # Number of T steps in the upper range

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 4: Cargar la clase y estimar magnitudes medias

MyIsing = Ising2D(Lx,Ly)

T = MyIsing.rangeTemperature(T_lower, T_upper, T_critical, T_critical_interval, T_steps_lower, T_steps_critical, T_steps_upper)

[E_mean, E_std, M_mean, M_std, M_abs_mean, M_abs_std] = MyIsing.runOverTemperature(Ps, T, n_steps, n_stats)

C = MyIsing.specificHeat(T, E_std)
X = MyIsing.susceptibility(T, M_std)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5: Guardar los datos en la carpeta data

unique_id = 2020_12_08
f_name = 'ising_mean_Lx'+str(Lx)+'_Ly'+str(Ly)+'_n'+str(n_steps)+'_id'+str(unique_id)
np.save('data/'+f_name+'.npy', [T, E_mean, E_std, M_mean, M_std, M_abs_mean, M_abs_std, C, X])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 6: Cargar los datos de la carpeta data

f_name = 'ising_mean_Lx48_Ly48_n100000_id6'
[T, E_mean, E_std, M_mean, M_std, M_abs_mean, M_abs_std, C, X] = np.load('data/'+f_name+'.npy')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 7: Energía media por sitio como función de la temperatura

plt.plot(T, E_mean, '.k')
plt.ylabel('Energía media por sitio')
plt.xlabel('Temperatura [kb*K]')
plt.axvline(T_critical, ls='--', color='#888')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 8: Magnetización media por sitio como función de la temperatura

plt.plot(T, M_mean, '.k')
plt.ylabel('Magnetización media por sitio')
plt.xlabel('Temperatura [kb*K]')
plt.axvline(T_critical, ls='--', color='#888')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 9: Magnetización absoluta media por sitio como función de la temperatura

plt.plot(T, M_abs_mean, '.k')
plt.ylabel('Magnetización absoluta media por sitio')
plt.xlabel('Temperatura [kb*K]')
plt.axvline(T_critical, ls='--', color='#888')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 10: Calor específico como función de la temperatura

plt.plot(T, C, '.k')
plt.ylabel('Calor específico')
plt.xlabel('Temperatura [kb*K]')
plt.axvline(T_critical, ls='--', color='#888')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 11: Susceptibilidad magnética como función de la temperatura

plt.plot(T, X, '.k')
plt.ylabel('Susceptibilidad magnética')
plt.xlabel('Temperatura [kb*K]')
plt.axvline(T_critical, ls='--', color='#888')