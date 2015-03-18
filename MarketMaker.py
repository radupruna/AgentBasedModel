__author__ = 'Radu'
from Fundamentalist import Fundamentalist
from Chartist import Chartist
import numpy
import matplotlib.pyplot as plt

class MarketMaker:
    pf = 0
    pt=[]
    nf=[]
    nc=[]
    xt=[]
    df = []
    dc = []
    attract=[]
    noise=[]
    fund = Fundamentalist
    chart= Chartist

    def __init__(self,pf,p_0,p_1,nf_0, nc_0):
        self.pt.append(p_0)
        self.pt.append(p_1)
        self.nf.append(nf_0)
        self.nc.append(nc_0)
        self.xt.append(nf_0-nc_0)
        self.fund = Fundamentalist(pf)
        self.chart = Chartist()

    def updateDemands(self):
        self.df.append( self.fund.getDemand(self.pt) )
        self.dc.append( self.chart.getDemand(self.pt) )

    def getDemandC(self):
        return self.dc

    def getDemandF(self):
        return self.df

    def getAttractLvl(self,p_f,x_t,p_t):
        alpha_0 = -0.15
        alpha_n = 1.35
        alpha_p = 11.40
        a_t = alpha_0 + alpha_n*x_t + alpha_p *(p_t - p_f)
        return a_t

    def updateFractions(self,a_t):
        b = 1.00
        n_f = 1 /(1+numpy.exp(-b*a_t))
        n_c = 1 - n_f
        self.nf.append(n_f)
        self.nc.append(n_c)
        self.xt.append(n_f-n_c)

    def getNf(self):
        return self.nf

    def getNc(self):
        return self.nc

    def updatePrice(self):
        nu = 0.01
        self.updateDemands()
        a = self.getAttractLvl(self.pf,self.xt[-1],self.pt[-1])
        self.attract.append(a)
        self.updateFractions(a)
        price = self.pt[-1] + nu*(self.dc[-1]*self.nc[-1] + self.df[-1]*self.nf[-1])
        self.pt.append(price)

MM= MarketMaker(0,0.151234,0.146115,0.7,0.3)

for i in range(6000):
    MM.updatePrice()

plt.figure(1)
plt.plot(MM.dc)
plt.title("Demand C")

plt.figure(2)
plt.plot(MM.df)
plt.title("Demand F")

plt.figure(3)
plt.plot(MM.pt)
plt.title("Price")

plt.figure(4)
plt.plot(MM.xt)
plt.title("Majority Index")

plt.figure(5)
plt.plot(MM.nf)
plt.title("Nf")

plt.figure(6)
plt.plot(MM.attract)
plt.title("Attractiveness Index")

plt.show()
