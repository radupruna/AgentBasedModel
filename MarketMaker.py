__author__ = 'Radu'
from Fundamentalist import Fundamentalist
from Chartist import Chartist
import math

class MarketMaker:
    pf = 0
    pt=[]
    nf=[]
    nc=[]
    df = []
    dc = []
    fund = Fundamentalist
    chart= Chartist

    def __init__(self,pf,p_0,p_1,nf_0, nc_0):
        self.pt.append(p_0)
        self.pt.append(p_1)
        self.nf.append(nf_0)
        self.nc.append(nc_0)
        self.fund = Fundamentalist(pf)
        self.chart = Chartist()

    def setDemands(self):
        self.df.append( self.fund.getDemand(self.pt) )
        self.dc.append( self.chart.getDemand(self.pt) )

    def getDemandC(self):
        return self.dc[-1]
    def getDemandF(self):
        return self.df[-1]

    def getAttractLvl(self,p_f,p_t):
        alpha_0 = -0.15
        alpha_n = 1.35
        alpha_p = 11.40
        a_t = alpha_0 + alpha_n*(self.nf[-1]-self.nc[-1]) + alpha_p * (p_t - p_f)**2
        return a_t

    def updateFractions(self,a_t):
        b = 1
        n_f = 1 / (1 + math.exp( -b * a_t))
        n_c = 1 - n_f
        self.nf.append(n_f)
        self.nc.append(n_c)

    def getNf(self):
        return self.nf

    def getNc(self):
        return self.nc


    def updatePrice(self):
        nu = 0.01
        price = self.pt[-1] + nu * (self.dc[-1] * self.nc[-1] + self.df[-1] * self.nf[-1]) +


MM= MarketMaker(0,0.12,14.12,0.7,0.3)
for i in range(5000):
    MM.setDemands()
    MM.updateFractions(MM.getAttractLvl(MM.pf,MM.pt[-2]))

print (MM.pt)
print ("Demand F: ", MM.getDemandF())
print ("Demand C: ",MM.getDemandC())
print ("NC: ", MM.getNc())
print ("NF: ",MM.getNf())