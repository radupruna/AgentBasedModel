__author__ = 'Radu'
from Fundamentalist import Fundamentalist
from Chartist import Chartist
from SwitchingMechanism import SwitchingMechanism

class MarketMaker:
    p_f = 0
    pt=[]
    nf=[]
    nc=[]
    df = []
    dc = []

    def __init__(self,p_f,p_0,p_1,nf_0, nc_0):
        self.pt.append(p_0)
        self.pt.append(p_1)
        self.nf.append(nf_0)
        self.nc.append(nc_0)

    fund = Fundamentalist(p_f)
    chart = Chartist()

    def ret(self):
        return self.fund.getP_f()

    def setDemands(self):
        p=self.pt
        self.df.append( self.fund.getDemand(p) )
        self.dc.append( self.chart.getDemand(p) )

    def getDemandC(self):
        return self.dc[-1]
    def getDemandF(self):
        return self.df[-1]


MM= MarketMaker(0,0.3,0.4,0.7,0.3)
print (MM.ret)
