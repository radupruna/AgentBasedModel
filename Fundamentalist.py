__author__ = 'Radu'
import numpy

class Fundamentalist:
    phi = 0.18
    sigma_f = 0.79
    def __init__(self, p_f):
        self.p_f=p_f

    def getPf(self):
        return self.p_f

    def setPf(self,p_f):
        self.p_f=p_f

    def getDemand(self,p):
        epsilon_f_t = numpy.random.normal(0,self.sigma_f)
        demand = self.phi * (self.getPf() - p[-1]) + epsilon_f_t
        return demand