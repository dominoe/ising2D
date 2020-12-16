
""" Ising2D Class | Last update: 2020-12-15 """

"""

This class is intented to resolve Ising in arbitrary 2D systems
using the Metropolis Monte Carlo (MMC) algorithm.

An alternative Ising Class is provided if you wish to calculate 3D systems.
You may expect a performance loss of about -16% for 2D systems in the other.

Examples for the alternative Ising Class are provided in the parent folder.

Since @jit-class compiler is experimental, some thing may break up in
future updates from numba. Either way, C-compilation is a must.

"""

import numpy as np
from numba import int32, float32
from numba.experimental import jitclass

spec = [ ('Lx', int32), ('Ly', int32), ('N', int32), ('H', float32), ('SE', float32[:,:]) ]

@jitclass(spec)
class Ising2D:

    def __init__(self, Lx, Ly = 1, H = .0):
        self.Lx, self.Ly = Lx, Ly
        self.H  = H
        self.N  = Lx*Ly
        self.SE = np.zeros((3,13), dtype=np.float32)
        self.SE[+1,+0] = -2*(H+0)
        self.SE[+1,+1] = -2*(H+1)
        self.SE[+1,+2] = -2*(H+2)
        self.SE[+1,+3] = -2*(H+3)
        self.SE[+1,+4] = -2*(H+4)
        self.SE[+1,+5] = -2*(H+5)
        self.SE[+1,+6] = -2*(H+6)
        self.SE[+1,-1] = -2*(H-1)
        self.SE[+1,-2] = -2*(H-2)
        self.SE[+1,-3] = -2*(H-3)
        self.SE[+1,-4] = -2*(H-4)
        self.SE[+1,-5] = -2*(H-5)
        self.SE[+1,-6] = -2*(H-6)
        self.SE[-1,+0] = +2*(H+0)
        self.SE[-1,+1] = +2*(H+1)
        self.SE[-1,+2] = +2*(H+2)
        self.SE[-1,+3] = +2*(H+3)
        self.SE[-1,+4] = +2*(H+4)
        self.SE[-1,+5] = +2*(H+5)
        self.SE[-1,+6] = +2*(H+6)
        self.SE[-1,-1] = +2*(H-1)
        self.SE[-1,-2] = +2*(H-2)
        self.SE[-1,-3] = +2*(H-3)
        self.SE[-1,-4] = +2*(H-4)
        self.SE[-1,-5] = +2*(H-5)
        self.SE[-1,-6] = +2*(H-6)

    def setS(self, Ps):
        return np.where(np.random.rand(self.Lx,self.Ly)<Ps,1,-1)

    def isingMagnetization(self, S):
        return np.sum(S)

    def isingEnergy(self, S):
        
        Lx, Ly = self.Lx, self.Ly
        i_max, j_max = Lx-1, Ly-1
        SE = self.SE
        E = .0

        for i in range(0, Lx):
            i_prev = i-1 if i!=0 else i_max
            i_next = i+1 if i!=i_max else 0
            for j in range (0, Ly):
                j_prev = j-1 if j!=0 else j_max
                j_next = j+1 if j!=j_max else 0
                E += SE[S[i,j], (S[i,j_next] + S[i,j_prev] + S[i_next,j] + S[i_prev,j])//2 ]

        return E/2

    def isingMMC(self, S, T):

        Lx, Ly = self.Lx, self.Ly
        i_max, j_max = Lx-1, Ly-1
        SE = self.SE
        PE = np.exp(-SE/T)
        RD = np.random.rand(Lx,Ly)
        dE, dM = 0, 0

        for i in range(0, Lx):
            i_prev = i-1 if i!=0 else i_max
            i_next = i+1 if i!=i_max else 0
            for j in range(0, Ly):
                j_prev = j-1 if j!=0 else j_max
                j_next = j+1 if j!=j_max else 0
                spin_inv = -S[i,j]
                spin_sum = S[i_next,j] + S[i_prev,j] + S[i,j_next] + S[i,j_prev]
                energy_new = SE[spin_inv,spin_sum]
                energy_pro = PE[spin_inv,spin_sum]
                if( (energy_new < 0) or (RD[i,j] < energy_pro) ):
                    S[i,j], dE, dM = spin_inv, dE+energy_new, dM+2*spin_inv

        return S, dE, dM

    def runOverMMC(self, S, T, n_steps):
        
        E = np.zeros(n_steps)
        M = np.zeros(n_steps)
        
        E[0] = self.isingEnergy(S)
        M[0] = self.isingMagnetization(S)

        for n in range(1, n_steps):
            S, dE, dM = self.isingMMC(S, T)
            E[n] = E[n-1] + dE
            M[n] = M[n-1] + dM

        return S, E/self.N, M/self.N
    
    def runOverTemperature(self, S, T, n_steps, n_stats):

        T_steps = T.size
        E_mean = np.zeros(T_steps)
        M_mean = np.zeros(T_steps)
        M_abs_mean = np.zeros(T_steps)
        E_std = np.zeros(T_steps)
        M_std = np.zeros(T_steps)
        M_abs_std = np.zeros(T_steps)

        S, _, _ = self.runOverMMC(S, T[-1], n_steps)
        for i in range(T_steps-1,-1,-1):
            S, E, M = self.runOverMMC(S, T[i], n_steps)
            E_mean[i], E_std[i], M_mean[i], M_std[i], M_abs_mean[i], M_abs_std[i] = self.isingStats(E, M, n_stats)
            S = self.setS(1-(M_mean[i]+1)/2) # Estimate probability spin-up, flip probability and set new S
            print(np.round(100-100*i/T_steps),"%")

        return E_mean, E_std, M_mean, M_std, M_abs_mean, M_abs_std

    def isingStats(self, E, M, n_stats):
        E_mean = np.mean(E[n_stats:])
        M_mean = np.mean(M[n_stats:])
        M_abs_mean = np.mean(np.abs(M[n_stats:]))
        E_std = np.std(E[n_stats:])
        M_std = np.std(M[n_stats:])
        M_abs_std = np.std(np.abs(M[n_stats:]))
        return E_mean, E_std, M_mean, M_std, M_abs_mean, M_abs_std

    def specificHeat(self, T, E_std):
        return (1./T**2)*(E_std**2)*(self.N) # E : Mean energy per site

    def susceptibility(self, T, M_abs_std):
        return (1./T)*(M_abs_std**2)*(self.N) # M : Abs. Mean Magn. per site

    def rangeTemperature(self, T_lower, T_upper, T_critical, T_critical_interval, T_steps_lower, T_steps_critical, T_steps_upper):
        return np.concatenate((
            np.linspace(T_lower, T_critical-T_critical_interval, T_steps_lower),
            np.linspace(T_critical-T_critical_interval, T_critical+T_critical_interval, T_steps_critical),
            np.linspace(T_critical+T_critical_interval, T_upper, T_steps_upper) ))