
""" G9E4 | Version: 2020-12-07 """

""" Análisis de datos | Exponentes críticos """

#%%%%%%%%%%%%%%%%%%%%
# 1: Cargar librerías

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import special

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2: Cargar todas las simulaciones desde 2x2 hasta 16384x16384

load_size = 14

f_name = ['']*load_size
Lx = [None]*load_size
Ly = [None]*load_size
n_stat = [None]*load_size
T = [None]*load_size
E_mean = [None]*load_size
E_std = [None]*load_size
E_mean = [None]*load_size
M_mean = [None]*load_size
M_std = [None]*load_size
M_abs_mean = [None]*load_size
M_abs_std = [None]*load_size
C = [None]*load_size
X = [None]*load_size
beta = [None]*load_size

Lx[0] =      2; Ly[0] =      2; n_stat[0] =  1e7-round(1e7/3); f_name[0] =  'ising_mean_Lx2_Ly2_n10000000_id20201208';
Lx[1] =      4; Ly[1] =      4; n_stat[1] =  1e7-round(1e7/3); f_name[1] =  'ising_mean_Lx4_Ly4_n10000000_id20201208';
Lx[2] =      8; Ly[2] =      8; n_stat[2] =  1e6-round(1e6/3); f_name[2] =  'ising_mean_Lx8_Ly8_n1000000_id20201208';
Lx[3] =     16; Ly[3] =     16; n_stat[3] =  1e6-round(1e6/3); f_name[3] =  'ising_mean_Lx16_Ly16_n1000000_id20201208';
Lx[4] =     32; Ly[4] =     32; n_stat[4] =  1e6-round(1e6/3); f_name[4] =  'ising_mean_Lx32_Ly32_n1000000_id20201208';
Lx[5] =     64; Ly[5] =     64; n_stat[5] =  1e6-round(1e6/3); f_name[5] =  'ising_mean_Lx64_Ly64_n1000000_id20201208';
Lx[6] =    128; Ly[6] =    128; n_stat[6] =  1e6-round(1e6/3); f_name[6] =  'ising_mean_Lx128_Ly128_n1000000_id20201208';
Lx[7] =    256; Ly[7] =    256; n_stat[7] =  1e5-round(1e5/3); f_name[7] =  'ising_mean_Lx256_Ly256_n100000_id20201208';
Lx[8] =    512; Ly[8] =    512; n_stat[8] =  1e5-round(1e5/3); f_name[8] =  'ising_mean_Lx512_Ly512_n100000_id20201208';
Lx[9] =   1024; Ly[9] =   1024; n_stat[9] =  1e5-round(1e4/3); f_name[9] =  'ising_mean_Lx1024_Ly1024_n10000_id20201208';
Lx[10] =  2048; Ly[10] =  2048; n_stat[10] = 1e3-round(1e3/2); f_name[10] = 'ising_mean_Lx2048_Ly2048_n1000_id20201208';
Lx[11] =  4096; Ly[11] =  4096; n_stat[11] = 1e3-round(1e3/2); f_name[11] = 'ising_mean_Lx4096_Ly4096_n1000_id20201208';
Lx[12] =  8192; Ly[12] =  8192; n_stat[12] = 1e2-round(1e2/2); f_name[12] = 'ising_mean_Lx8192_Ly8192_n100_id20201208';
Lx[13] = 16384; Ly[13] = 16384; n_stat[13] = 1e1-round(1e1/2); f_name[13] = 'ising_mean_Lx16384_Ly16384_n10_id20201208';

for n in range(load_size):
    [T[n], E_mean[n], E_std[n], M_mean[n], M_std[n], M_abs_mean[n], M_abs_std[n], C[n], X[n]] = np.load('data/'+f_name[n]+'.npy')
    beta[n] = 1/T[n]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 3: Definir cantidades de interés

beta_onsager = np.log(1+np.sqrt(2))/2 # = 0.4406867935097714...
T_onsager = 1/beta_onsager

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 4: Energía media por sitio como función de la temperatura

for n in range(1,load_size-5):
    plt.plot(beta[n], E_mean[n], label="{0}x{1}".format(Lx[n], Ly[n]))

plt.legend()
plt.ylabel('Energía media por sitio')
plt.xlabel('Beta')
plt.axvline(beta_onsager, ls='--', color='#888')
plt.xlim(left=0.2,right=.7)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5: Magnetización media por sitio como función de la temperatura

for n in range(1,load_size):
    plt.plot(T[n], M_mean[n], '.', label="{0}x{1}".format(Lx[n], Ly[n]))

plt.legend()
plt.ylabel('Magnetización media por sitio')
plt.xlabel('Temperatura [kb*K]')
plt.axvline(T_onsager, ls='--', color='#888')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 6: Magnetización absoluta media por sitio como función de la temperatura

for n in range(1,load_size):
    plt.plot(T[n], M_abs_mean[n], label="{0}x{1}".format(Lx[n], Ly[n]))

plt.legend()
plt.ylabel('Magnetización absoluta media por sitio')
plt.xlabel('Temperatura [kb*K]')
plt.axvline(T_onsager, ls='--', color='#888')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 7A: Calor específico como función de la temperatura

for n in range(1,7):
    plt.plot(beta[n], C[n], label="{0}x{1}".format(Lx[n], Ly[n]))

plt.legend()
plt.ylabel('Calor específico')
plt.xlabel('Beta')
plt.axvline(beta_onsager, ls='--', color='#888')
plt.xlim(left=0.2,right=.7)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 7B: Calor específico como función de la temperatura

for n in range(1,load_size):
    plt.plot(beta[n], np.log(C[n]), label="{0}x{1}".format(Lx[n], Ly[n]))

plt.legend()
plt.ylabel('Calor específico [log_e]')
plt.xlabel('Beta')
plt.axvline(beta_onsager, ls='--', color='#888')
plt.xlim(left=0.2,right=.7)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 8A: Susceptibilidad magnética como función de la temperatura

for n in range(1,6):
    plt.plot(beta[n], X[n], label="{0}x{1}".format(Lx[n], Ly[n]))

plt.legend()
plt.ylabel('Susceptibilidad magnética')
plt.xlabel('Beta')
plt.axvline(beta_onsager, ls='--', color='#888')
plt.xlim(left=0.2,right=.7)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 8B: Susceptibilidad magnética como función de la temperatura

for n in range(2,load_size-5):
    plt.plot(beta[n], np.log(X[n]), label="{0}x{1}".format(Lx[n], Ly[n]))

plt.legend()
plt.ylabel('Susceptibilidad magnética [log_e]')
plt.xlabel('Beta')
plt.axvline(beta_onsager, ls='--', color='#888')
plt.xlim(left=0.2,right=.7)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 3: Definición de funciones para análisis de datos

# Ref: The Analytical Expressions for a Finite-Size 2D Ising Model
#      M.Yu. Malsagov, I.M. Karandashev and B.V. Kryzhanovsky,

beta_onsager = np.log(1+np.sqrt(2))/2 # = 0.4406867935097714...
T_onsager = 1/beta_onsager

def beta_critical(N):
    return (0.4406867935097714)*(1+5/4/np.sqrt(N))

def energy_critical(N):
    return -np.sqrt(2)*(1-1/(2*np.sqrt(N)))

def specific_heat_critical(N):
    return (4/np.pi)*(beta_critical(N)**2)*(np.log(N)-1.7808)

def delta_1(N):
    return 5/4/np.sqrt(N)

def delta_2(N):
    return np.pi**2/N

def zeta(beta,J,delta_1):
    return 2*beta*J/(1+delta_1)

def kappa(z,delta_2):
    return 2*np.sinh(z)/(1+delta_2)/np.cosh(z)**2

def rho(z,delta_2):
    return (1-np.sinh(z)**2)**2 /( (1+delta_2)**2 * np.cosh(z)**4 - 4*np.sinh(z)**2 )

def alpha_1(rho,delta_2):
    return rho*(1+delta_2)**2
    
def alpha_2(rho):
    return 2*rho-1

def energy_analytic(beta,J,N):
    x = delta_1(N)
    y = delta_2(N)
    z = zeta(beta,J,x)
    K1 = special.ellipk(kappa(z,y)) # full elliptic integral first type
    return -1/(1+x)*(2*np.tanh(z)+(np.sinh(z)**2-1)/np.sinh(z)/np.cosh(z)*(2*K1/np.pi-1))
    
def heat_capacity_analytic(beta,J,N):
    x = delta_1(N)
    y = delta_2(N)
    z = zeta(beta,J,x)
    p = rho(z,y)
    a1 = alpha_1(p,y)
    a2 = alpha_2(p)
    K1 = special.ellipk(kappa(z,y)) # full elliptic integral first type
    K2 = special.ellipe(kappa(z,y)) # full elliptic integral second type
    return z**2/np.pi/np.tanh(z)**2 * ( a1*(K1-K2) - (1-np.tanh(z)**2)*(np.pi/2 + (2*a2*np.tanh(z)**2-1)*K1 ) )

def heat_capacity_approximate(beta, J, N, beta_critical):
    return (4*beta_critical**2*J**2/np.pi)*(3*np.log(2)-np.pi/2-np.log(4*J**2*(beta-beta_critical)**2+np.pi**2/N))

def magnetic_susceptibility(T, Tc, alpha):
  return np.abs(1-T/Tc)**(-alpha)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Intentos de fiteo análitico con datos... no dan

q = 7

J = 1
N = Lx[q]*Ly[q]
beta = 1/T[q]
#beta_expected = 0.4462 ### se usó el de 100x100

heat_capacity_simulated =  C[q]

beta_space = np.linspace(0.01, 2, 100000)
temp_space = np.linspace(0.01, 4.5, 10000)

plt.plot(beta, heat_capacity_simulated, '.k')
plt.plot(beta_space, heat_capacity_analytic(beta_space,J,10000))
plt.xlim(left=0,right=.8)


#i_min = 45
#i_max = 100

#T_sim = T[i_min:i_max]
#X_sim = X[i_min:i_max]



# popt, pcov = curve_fit(chi, T_sim, X_sim, p0 = (2.269, 1.0))

# Tc = popt[0]
# dTc = np.sqrt(pcov[0,0])
# alpha = popt[1]
# dalpha = np.sqrt(pcov[1,1])

# print(Tc)
# print(dTc)

# print(alpha)
# print(dalpha)

# plt.plot(T_sim, X_sim, '.k')
# plt.plot(T_exp, chi(T_exp, *popt))
# plt.ylim(top=50, bottom=-0.1)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Estimación del beta crítico como función del L

L_space = np.linspace(4, 1000, 100000)
N_space = L_space*L_space

i_critical_arr = np.argmax(C, axis=1)
beta_critical_arr = 1/T[0][i_critical_arr]
specific_heat_critical_arr = C[0][i_critical_arr]

plt.plot(Lx, beta_critical_arr, '.')
plt.plot(L_space, beta_critical(N_space))
plt.xlim(left=0,right=1050)


print(beta_critical_arr)

#plt.plot(Lx, specific_heat_critical_arr)
#plt.plot(beta[1], energy_analytic(beta[1], 1, 32))
#plt.xlim(left=0,right=1050)

