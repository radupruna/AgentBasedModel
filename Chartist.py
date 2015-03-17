__author__ = 'Radu'
import numpy

class Chartist:
    chi = 2.35
    sigma_c = 1.91

    def __int__(self,p_0,p_11):
        self.p_0=p_0
        self.p_1=p_1

    def setP_0(self,p_0):
        self.p_0 = p_0
    def getP0(self):
        return self.p0

    def setP_1(self,p_1):
        self.p_1=p_1
    def getP1(self):
        return self.p1

    def getDemand(self,pt):
        epsilon_c_t = numpy.random.normal(0,self.sigma_c)
        demand = self.chi * (p[-2] - p[-1]) + epsilon_c_t
        return demand