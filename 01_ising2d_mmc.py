
""" 01_ising2d_mmc | Version: 2020-12-12 """

#%%%%%%%%%%%%%%%%%%%%
# 1: Cargar librerías

import numpy as np
import matplotlib.pyplot as plt
from lib.ising2d import Ising2D

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2: Condiciones iniciales del sistema

Lx = 32 # Número de sitios en la dimensión x
Ly = 32 # Número de sitios en la dimensión y
Ps = .5 # Probabilidad inicial en la cadena de que un sitio sea spin-up

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 3: Condiciones de contorno y de Markov

T = 2.0 # Temperatura del sistema (Crítica Onsager es 2.269185314213022)
n_steps = int(1e3) # Número de pasos en la cadena de Markov
n_stats = round(n_steps/3) # Cantidad de n-pasos para estadística

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 4: Cargar la clase de Ising

MyIsing = Ising2D(Lx,Ly)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5: Verificar una configuración inicial posible del sistema

S = MyIsing.setS(Ps)
plt.pcolor(S)
plt.clim(-1,1)
plt.axis('square')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 6: Ejecutar MC en la cadena y obtener magnitudes y estimaciones

E, M = MyIsing.runOverMarkov(Ps, T, n_steps)

[E_mean, E_std, M_mean, M_std, M_abs_mean, M_abs_std] = MyIsing.isingStats(E, M, n_stats)

C = MyIsing.specificHeat(T,E_std)
X = MyIsing.susceptibility(T,M_abs_std)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 7: Energía por sitio como función del n-paso de Markov

plt.plot(E)
plt.plot(np.cumsum(E)/range(1,n_steps+1),'r--')
plt.axvline(n_stats,ls='--',color='k')
plt.title(rf'$\langle e \rangle = {round(E_mean,4)}$')
plt.xlabel('Número de paso')
plt.ylabel('Energía por sitio')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 8: Magnetización por sitio como función del n-paso de Markov

plt.plot(M)
plt.plot(np.cumsum(M)/range(1,n_steps+1),'r--')
plt.axvline(n_stats,ls='--',color='k')
plt.title(rf'$\langle m \rangle = {round(M_mean,4)}$')
plt.xlabel('Número de paso')
plt.ylabel('Magnetización por sitio');
plt.ylim(top=1.1,bottom=-1.1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 9: Magnetización absoluta por sitio como función del n-paso de Markov

plt.plot(abs(M))
plt.plot(np.cumsum(abs(M))/range(1,n_steps+1),'r--')
plt.axvline(n_stats,ls='--',color='k')
plt.title(rf'$\langle m \rangle = {round(M_abs_mean,4)}$')
plt.xlabel('Número de paso')
plt.ylabel('Magnetización por sitio');
plt.ylim(top=1.1,bottom=-0.1)