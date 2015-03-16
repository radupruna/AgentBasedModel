__author__ = 'Radu'
import numpy
import MarketMaker
class Chartist:
    chi = 2.35
    sigma_c = 1.91

    mm = MarketMaker(0,0.1)
    p_t = mm.p_t

    def __int__(self,p0,p1):
        self.p_t[0]=p0
        self.p_t[1]=p1

    def getDemand(self):
        epsilon_c = numpy.random.normal(0,self.sigma_c)
        demand = self.chi * (self.p_t[-2] - self.p_t[-1]) + epsilon_c
        return demand


