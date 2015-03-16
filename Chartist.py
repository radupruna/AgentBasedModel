__author__ = 'Radu'
import numpy

class Chartist(object):
    chi = 2.35
    sigma_c = 1.91


    def getDemand(self,p_0,p_1):
        epsilon_c = numpy.random.normal(0,self.sigma_c)
        demand = self.chi * (p_1 - p_0) * epsilon_c
        return demand