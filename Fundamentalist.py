__author__ = 'Radu'
import numpy

class Fundamentalist(object):
    phi = 0.18
    sigma_f = 0.79
    def __init__(self, p_f):
        self.p_f=p_f

    def getPf(self):
        return self.p_f

    def setPf(self,p_f):
        self.p_f=p_f

    def getDemand(self,p_t):
        epsilon_f = numpy.random.normal(0,self.sigma_f)
        demand = self.phi * (self.getPf() - p_t) * epsilon_f
        return demand