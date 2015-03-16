__author__ = 'Radu'
import numpy

class Fundamentalist(object):
    phi = 0.18
    sigma_f = 0.79
    def __init__(self, p_f):
        self.p_f=p_f

    def getPf(self):
        return self.p_f

    def setPf(self,p):
        self.p_f=p

    def getDemand(self,p_t):

        demand = self.phi * ( self.getPf() - p_t)

    print (numpy.random.normal(0,sigma_f))
