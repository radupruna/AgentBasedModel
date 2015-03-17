__author__ = 'Radu'
from Fundamentalist import Fundamentalist
from Chartist import Chartist
from SwitchingMechanism import SwitchingMechanism

class MarketMaker:
    pf = 0
    pt=[]
    nf=[]
    nc=[]
    df = []
    dc = []

    fund = Fundamentalist
    chart= Chartist
    sm = SwitchingMechanism

    def __init__(self,pf,p_0,p_1,nf_0, nc_0):
        self.pt.append(p_0)
        self.pt.append(p_1)
        self.nf.append(nf_0)
        self.nc.append(nc_0)
        fund = Fundamentalist(pf)
        chart = Chartist()
        sm = SwitchingMechanism(nf_0,nc_0)
        self.nf=sm.nf
        self.nc=sm.nc

    def setDemands(self):
        self.df.append(self.fund.getDemand(self.pt))
        self.dc.append(self.chart.getDemand(self.pt))

    def getDemandC(self):
        return self.dc[-1]
    def getDemandF(self):
        return self.df[-1]


MM= MarketMaker(0,0.3,0.4,0.7,0.3)
MM.setDemands()
print (MM.getDemandC())