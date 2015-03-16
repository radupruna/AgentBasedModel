__author__ = 'Radu'
import numpy

class Chartist(object):
    chi = 0.18
    sigma_c = 0.79

    def getDemand(self,p_0,p_t):
        epsilon_c = numpy.random.normal(0,self.sigma_c)
        demand = self.chi * (p_t - p_0) * epsilon_c
        return demand