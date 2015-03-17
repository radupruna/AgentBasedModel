__author__ = 'Radu'
import math
class SwitchingMechanism:
    nf=[]
    nc=[]

    def __init__(self,n_f,n_c):
        self.nf[0]=n_f
        self.nc[0]=n_c

    def getAttractLvl(self,p_f,p_t):
        alpha_0 = -0.15
        alpha_n = 1.35
        alpha_p = 11.40
        a_t = alpha_0 + alpha_n*(self.nf[-1]-self.nc[-1]) + alpha_p * (p_t - p_f)**2
        return a_t

    def updateFractions(self,a_t):
        b = 1
        n_f = 1 / (1 + math.exp( -b * a_t))
        n_c = 1 - n_f
        self.nf.append(n_f)
        self.nc.append(n_c)

    def getNf(self):
        return self.nf

    def getNc(self):
        return self.nc