__author__ = 'Radu'
import numpy

class Chartist:
    chi = 2.35
    sigma_c = 1.91

    def __int__(self,p0,p1):
        self.p0=p0
        self.p1=p1

    def getP0(self):
        return self.p0

    def getP1(self):
        return self.p1

    def getDemand(self,p):
        epsilon_c_t = numpy.random.normal(0,self.sigma_c)
        demand = self.chi * (p[-2] - p[-1]) + epsilon_c_t
        return demand