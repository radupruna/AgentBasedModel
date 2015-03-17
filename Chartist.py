__author__ = 'Radu'
import numpy

class Chartist:
    chi = 2.35
    sigma_c = 1.91
    epsilon_c = 0

    def getDemand(self,pt):
        self.epsilon_c = numpy.random.normal(0,self.sigma_c)
        demand = self.chi * (pt[-1] - pt[-2]) + self.epsilon_c
        return demand

    def setP_0(self,p_0):
        self.p_0 = p_0
    def getP_0(self):
        return self.p_0
    def setP_1(self,p_1):
        self.p_1=p_1
    def getP_1(self):
        return self.p_1
