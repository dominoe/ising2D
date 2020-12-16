
""" 3 : Ising Analysis | Version: 2020-12-15 """

"""

This example is created in order to analize many different systems at once.

"""

#%%%%%%%%%%%%%%%%%%
# 1: Load libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 2: Load simulations, using load_size and relevant info

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
Ln = [None]*load_size

Lx[0] =      2; Ly[0] =      2; n_stat[0] =  1e7-round(1e7/3); f_name[0] =  '2D_ising_mean_Lx2_Ly2_n10000000_id20201208';
Lx[1] =      4; Ly[1] =      4; n_stat[1] =  1e7-round(1e7/3); f_name[1] =  '2D_ising_mean_Lx4_Ly4_n10000000_id20201208';
Lx[2] =      8; Ly[2] =      8; n_stat[2] =  1e6-round(1e6/3); f_name[2] =  '2D_ising_mean_Lx8_Ly8_n1000000_id20201208';
Lx[3] =     16; Ly[3] =     16; n_stat[3] =  1e6-round(1e6/3); f_name[3] =  '2D_ising_mean_Lx16_Ly16_n1000000_id20201208';
Lx[4] =     32; Ly[4] =     32; n_stat[4] =  1e6-round(1e6/3); f_name[4] =  '2D_ising_mean_Lx32_Ly32_n1000000_id20201208';
Lx[5] =     64; Ly[5] =     64; n_stat[5] =  1e6-round(1e6/3); f_name[5] =  '2D_ising_mean_Lx64_Ly64_n1000000_id20201208';
Lx[6] =    128; Ly[6] =    128; n_stat[6] =  1e6-round(1e6/3); f_name[6] =  '2D_ising_mean_Lx128_Ly128_n1000000_id20201208';
Lx[7] =    256; Ly[7] =    256; n_stat[7] =  1e5-round(1e5/3); f_name[7] =  '2D_ising_mean_Lx256_Ly256_n100000_id20201208';
Lx[8] =    512; Ly[8] =    512; n_stat[8] =  1e5-round(1e5/3); f_name[8] =  '2D_ising_mean_Lx512_Ly512_n100000_id20201208';
Lx[9] =   1024; Ly[9] =   1024; n_stat[9] =  1e5-round(1e4/3); f_name[9] =  '2D_ising_mean_Lx1024_Ly1024_n10000_id20201208';
Lx[10] =  2048; Ly[10] =  2048; n_stat[10] = 1e3-round(1e3/2); f_name[10] = '2D_ising_mean_Lx2048_Ly2048_n1000_id20201208';
Lx[11] =  4096; Ly[11] =  4096; n_stat[11] = 1e3-round(1e3/2); f_name[11] = '2D_ising_mean_Lx4096_Ly4096_n1000_id20201208';
Lx[12] =  8192; Ly[12] =  8192; n_stat[12] = 1e2-round(1e2/2); f_name[12] = '2D_ising_mean_Lx8192_Ly8192_n100_id20201208';
Lx[13] = 16384; Ly[13] = 16384; n_stat[13] = 1e1-round(1e1/2); f_name[13] = '2D_ising_mean_Lx16384_Ly16384_n10_id20201208';

for n in range(load_size):
    [T[n], E_mean[n], E_std[n], M_mean[n], M_std[n], M_abs_mean[n], M_abs_std[n], C[n], X[n]] = np.load('data/'+f_name[n]+'.npy')
    beta[n] = 1/T[n]
    Ln[n] = Lx[n]*Ly[n]

color = plt.cm.jet(np.linspace(0.1,0.9,load_size))

beta_space = np.linspace(0.01, 2.00, 100000)
temp_space = np.linspace(0.01, 4.50, 100000)
L_space = np.linspace(4, 1000, 100000)
N_space = L_space*L_space

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 3: Define quantities of interest

beta_onsager = np.log(1+np.sqrt(2))/2 # = 0.4406867935097714...
T_onsager = 1/beta_onsager

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 4: Mean energy per site as a function of temperature 

for n in range(1,load_size):
    plt.plot(T[n], E_mean[n], label="{0}x{1}".format(Lx[n], Ly[n]), color = color[n])

plt.legend()
plt.ylabel('Mean Energy per site')
plt.xlabel('Temperature')
plt.axvline(T_onsager, ls='--', color='#888')
#plt.xlim(left=0.2,right=.7)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 5: Mean magnetization per site as a function of temperature

for n in range(1,load_size):
    plt.plot(T[n], M_mean[n], '.', label="{0}x{1}".format(Lx[n], Ly[n]), color = color[n])

plt.legend()
plt.ylabel('Mean Magnetization per site')
plt.xlabel('Temperature')
plt.axvline(T_onsager, ls='--', color='#888')
#plt.xlim(left=0.2,right=.7)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 6: Mean absolute magnetization per site as a function of temperature

for n in range(1,load_size):
    plt.plot(beta[n], M_abs_mean[n], label="{0}x{1}".format(Lx[n], Ly[n]), color = color[n])

plt.legend()
plt.ylabel('Mean Absolute Magnetization per site')
plt.xlabel('Temperature')
plt.axvline(T_onsager, ls='--', color='#888')
plt.xlim(left=0.2,right=.7)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 7A: Specific heat as a function of beta

for n in range(1,load_size):
    plt.plot(beta[n], C[n], label="{0}x{1}".format(Lx[n], Ly[n]), color = color[n])

plt.legend()
plt.ylabel('Specific Heat')
plt.xlabel('Beta')
plt.axvline(beta_onsager, ls='--', color='#888')
plt.xlim(left=0.2,right=.7)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 7B: Log Specific heat as a function of beta

for n in range(1,load_size):
    plt.plot(beta[n], np.log(C[n]), label="{0}x{1}".format(Lx[n], Ly[n]), color = color[n])

plt.legend()
plt.ylabel('Specific Heat [log]')
plt.xlabel('Beta')
plt.axvline(beta_onsager, ls='--', color='#888')
plt.xlim(left=0.2,right=.7)
plt.ylim(bottom=-4,top=18)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 8A:  Magnetic susceptibility as a function of beta

for n in range(1,load_size):
    plt.plot(beta[n], X[n], label="{0}x{1}".format(Lx[n], Ly[n]), color = color[n])

plt.legend()
plt.ylabel('Magnetic Susceptibility')
plt.xlabel('Beta')
plt.axvline(beta_onsager, ls='--', color='#888')
plt.xlim(left=0.2,right=.7)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 8B: Log Magnetic susceptibility as a function of beta

for n in range(1,load_size-2):
    plt.plot(beta[n], np.log(X[n]), label="{0}x{1}".format(Lx[n], Ly[n]), color = color[n])

plt.legend()
plt.ylabel('Magnetic Susceptibility [log]')
plt.xlabel('Beta')
plt.axvline(beta_onsager, ls='--', color='#888')
plt.xlim(left=0.2,right=.7)
plt.ylim(bottom=-4,top=12)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 9: Tc estimations as the sampled maximum from C(T)

i_critical_max = np.argmax(C, axis=1)
T_critical_max = [None]*load_size
beta_critical_max = [None]*load_size

for n in range(load_size):
    T_critical_max[n] = T[n][i_critical_max[n]]
    beta_critical_max[n] = 1/T_critical_max[n]


for n in range(1,9):
    plt.plot(beta[n], C[n], label="{0}x{1}".format(Lx[n], Ly[n]), color = color[n+3])
    plt.axvline(beta_critical_max[n], ls='--', color = color[n+3])

plt.legend()
plt.ylabel('Specific Heat')
plt.xlabel('Beta')
plt.axvline(beta_onsager, ls='--', color='#888')
plt.xlim(left=0.40,right=.45)
plt.ylim(bottom=0,top=12)

print(np.round(T_critical_max,3))
print(np.round(beta_critical_max,3))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 10: Beta Critical in terms of L. Run after previous block.

plt.plot(Lx, beta_critical_max, '--', marker='o')
plt.axhline(beta_onsager, ls='--', color='#888')
plt.xlim(left=-10,right=1050)
plt.ylim(bottom=0.36,top=0.45)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 11: Magnetization fitting for T<Tc to estimate beta exponent

temperature_for_fit = [None]*load_size
magnetization_for_fit = [None]*load_size
log_magnetization_for_fit = [None]*load_size
beta_exponent_fit = [None]*load_size
constant_exponent_fit = [None]*load_size

for n in range(0,load_size):
    i_min = i_critical_max[n]-2
    i_max = i_critical_max[n]
    temperature_for_fit[n] = T_critical_max[n]-T[n][i_min:i_max]
    magnetization_for_fit[n] = M_abs_mean[n][i_min:i_max]
    temperature_for_fit[n] = np.log(temperature_for_fit[n])
    log_magnetization_for_fit[n] = np.log(magnetization_for_fit[n])
    beta_exponent_fit[n],constant_exponent_fit[n] = np.polyfit(temperature_for_fit[n], log_magnetization_for_fit[n], 1)
    plt.plot(temperature_for_fit[n], log_magnetization_for_fit[n], label="{0}x{1}".format(Lx[n], Ly[n]), color = color[n])

plt.legend()
plt.ylabel('ln(|m|)')
plt.xlabel('ln(Tc-T)')

print(np.round(beta_exponent_fit,3))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 12: Susceptibiliy fitting for T<Tc to estimate gamma exponent

temperature_for_fit = [None]*load_size
magnetization_for_fit = [None]*load_size
log_magnetization_for_fit = [None]*load_size
gamma_exponent_fit = [None]*load_size
constant_exponent_fit = [None]*load_size

for n in range(0,load_size):
    i_min = i_critical_max[n]-2
    i_max = i_critical_max[n]
    temperature_for_fit[n] = T_critical_max[n]-T[n][i_min:i_max]
    magnetization_for_fit[n] = X[n][i_min:i_max]
    temperature_for_fit[n] = np.log(temperature_for_fit[n])
    log_magnetization_for_fit[n] = np.log(magnetization_for_fit[n])
    gamma_exponent_fit[n],constant_exponent_fit[n] = np.polyfit(temperature_for_fit[n], log_magnetization_for_fit[n], 1)
    gamma_exponent_fit[n] = -gamma_exponent_fit[n]
    plt.plot(temperature_for_fit[n], log_magnetization_for_fit[n], label="{0}x{1}".format(Lx[n], Ly[n]), color = color[n])

plt.legend()
plt.ylabel(r'$\ln{|\chi|}$')
plt.xlabel(r'$\ln{(T_c-T)}$')

print(np.round(gamma_exponent_fit,3))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Ref:  The Analytical Expressions for a Finite-Size 2D Ising Model
#       M.Yu. Malsagov, I.M. Karandashev and B.V. Kryzhanovsky,
# Notes: Do not use the following, may contain errors.

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

def magnetization_fit(T, Tc, beta_exp):
  return np.abs(Tc-T)**(-beta_exp)