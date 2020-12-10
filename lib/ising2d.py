
""" Version: 2020-12-07 """

"""

Lx, Ly : Número de sitios en la dimensión x,y
Ln : Número total de sitios [Lx]x[Ly]
h : Intensidad del campo magnético (interacción campo-spin)
i_next,...,j_prev : Posición de cierto sitio
x_next,...,y_prev : Vector centrado en (0,0) que da las posiciones de los sitios a derecha,...,abajo
S[i,j] : Matriz de spins
mag : Interacción spin campo.
energia : Interaccion entre spins
beta : = 1/kT

"""


#%%############################################
# LIBRARIES

import numpy as np
from numba import int32, float32
from numba.experimental import jitclass

#%%############################################
# ISING CLASS

@jitclass([('Lx',int32),
           ('Ly',int32),
           ('Ln',int32),
           ('H',float32),
           ('x_next',int32[:]),
           ('x_prev',int32[:]),
           ('y_next',int32[:]),
           ('y_prev',int32[:]) ])

class Ising2D:

    def __init__(self, Lx, Ly = None, H = .0):
        if Ly is None:
            Ly = Lx
        self.Lx = Lx
        self.Ly = Ly
        self.Ln = Lx*Ly
        self.H = H
        self.x_next = np.roll(np.arange(0, Lx, dtype=np.int32), 1)
        self.x_prev = np.roll(np.arange(0, Lx, dtype=np.int32),-1)
        self.y_next = np.roll(np.arange(0, Ly, dtype=np.int32), 1)
        self.y_prev = np.roll(np.arange(0, Ly, dtype=np.int32),-1)

    def setS(self, Ps):
        return np.where(np.random.rand(self.Lx,self.Ly)<Ps,1,-1)

    def isingEnergy(self, S):

        Lx = self.Lx
        Ly = self.Ly
        H = self.H

        E = .0

        for i in range(0, Lx):
            i_next = self.x_next[i]
            i_prev = self.x_prev[i]
            for j in range (0, Ly):
                j_next = self.y_next[j]
                j_prev = self.y_prev[j]
                E += -S[i,j]*( H + (S[i,j_next] + S[i,j_prev] + S[i_next,j] + S[i_prev,j])/2. )

        return E

    def isingMagnetization(self, S):
        return np.sum(S)

    def specificHeat(self, T, E_std):
        return (1./T**2)*(E_std**2)*(self.Ln) # E : Energía media por sitio

    def susceptibility(self, T, M_abs_std):
        return (1./T)*(M_abs_std**2)*(self.Ln) # M : Mag. media abs. por sitio

    def isingStepMC(self,S,T):

        Lx = self.Lx
        Ly = self.Ly
        H = self.H
        x_next = self.x_next
        x_prev = self.x_prev
        y_next = self.y_next
        y_prev = self.y_prev
        
        beta = 1./T
        dE = .0
        dM = .0

        for i in range(0, Lx):
            i_next = x_next[i]
            i_prev = x_prev[i]
            for j in range(0, Ly):
                j_next = y_next[j]
                j_prev = y_prev[j]
                spin_new = -S[i,j]
                energy_new = (-2.)*spin_new*( H + S[i_next,j] + S[i_prev,j] + S[i,j_next] + S[i,j_prev] )
                if( (energy_new < 0) or (np.random.rand() < np.exp(-beta*energy_new)) ):
                    S[i,j] = spin_new
                    dE += energy_new
                    dM += 2*spin_new

        return S, dE, dM

    def runOverMarkov(self, Ps, T, n_steps):
        
        E = np.zeros(n_steps)
        M = np.zeros(n_steps)
        
        S = self.setS(Ps)
        E[0] = self.isingEnergy(S)
        M[0] = self.isingMagnetization(S)

        for n in range(1, n_steps):
            S, dE, dM = self.isingStepMC(S, T)
            E[n] = E[n-1] + dE
            M[n] = M[n-1] + dM

        return E/self.Ln, M/self.Ln
    
    def runOverTemperature(self, Ps, T, n_steps, n_stats):

        T_steps = T.size
        E_mean = np.zeros(T_steps)
        M_mean = np.zeros(T_steps)
        M_abs_mean = np.zeros(T_steps)
        E_std = np.zeros(T_steps)
        M_std = np.zeros(T_steps)
        M_abs_std = np.zeros(T_steps)

        for i in range(T_steps-1,-1,-1): # Se recorre el array al revés
            E, M = self.runOverMarkov(Ps, T[i], n_steps)
            [E_mean[i], E_std[i], M_mean[i], M_std[i], M_abs_mean[i], M_abs_std[i]] = self.isingStats(E, M, n_stats)
            Ps = 1-(M_mean[i]+1)/2 # Opcional: Probabilidad de un spin-up con switcheo [1-P_up]
            #print(i,"/",T_steps)
            print(np.round(100-100*i/T_steps),"%")
            
        return [E_mean, E_std, M_mean, M_std, M_abs_mean, M_abs_std]

    def isingStats(self, E, M, n_stats):
        E_mean = np.mean(E[:n_stats])
        M_mean = np.mean(M[:n_stats])
        M_abs_mean = np.mean(np.abs(M[:n_stats]))
        E_std = np.std(E[:n_stats])
        M_std = np.std(M[:n_stats])
        M_abs_std = np.std(np.abs(M[:n_stats]))
        return [E_mean, E_std, M_mean, M_std, M_abs_mean, M_abs_std]

    def rangeTemperature(self, T_lower, T_upper, T_critical, T_critical_interval, T_steps_lower, T_steps_critical, T_steps_upper):
        return np.concatenate((
            np.linspace(T_lower, T_critical-T_critical_interval, T_steps_lower),
            np.linspace(T_critical-T_critical_interval, T_critical+T_critical_interval, T_steps_critical),
            np.linspace(T_critical+T_critical_interval, T_upper, T_steps_upper) ))