__author__ = 'Radu'
import numpy

class Fundamentalist:
    phi = 0.18
    sigma_f = 0.79
    p_f = 0
    epsilon_f = 0

    def __init__(self, p_f):
        self.p_f=p_f

    def getP_f(self):
        return self.p_f

    def setP_f(self,p_f):
        self.p_f=p_f

    def getDemand(self,pt):
        self.epsilon_f = numpy.random.normal(0,self.sigma_f)
        demand = self.phi * (self.getP_f() - pt[-1]) + self.epsilon_f
        return demand
