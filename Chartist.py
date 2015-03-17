__author__ = 'Radu'
import numpy

class Chartist:
    chi = 2.35
    sigma_c = 1.91

    def __int__(self,p_0,p_1):
        pass

    def getDemand(self,pt):
        epsilon_c_t = numpy.random.normal(0,self.sigma_c)
        demand = self.chi * (pt[-2] - pt[-1]) + epsilon_c_t
        return demand

    def setP_0(self,p_0):
        self.p_0 = p_0
    def getP_0(self):
        return self.p_0
    def setP_1(self,p_1):
        self.p_1=p_1
    def getP_1(self):
        return self.p_1