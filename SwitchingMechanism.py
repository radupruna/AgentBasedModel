__author__ = 'Radu'
import Fundamentalist
import Chartist
    class SwitchingMechanism:
        fundamentalist = Fundamentalist(0)
        nf=[]
        nc=[]
        def __init__(self,n_f,n_c):
            self.nf[0]=n_f
            self.nc[0]=n_c

        def computeAttractiveness(self):
            alpha_0 = -0.15
            alpha_n = 1.35
            alpha_p = 11.40
            a_t = alpha_0 + alpha_n*(self.nf[-1]-self.nc[-1]) + alpha_p * (self.fundamentalist.getPf()

        def updateFractions(self,a_t):
            pass