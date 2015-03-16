__author__ = 'Radu'

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

        demand = self.phi * ( self.getPf() - p_t) * epsilon_t


