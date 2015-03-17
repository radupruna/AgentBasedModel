__author__ = 'Radu'
import Fundamentalist
import Chartist
import SwitchingMechanism

class MarketMaker:
    pt = []
    fund = Fundamentalist
    chart = Chartist
    sm = SwitchingMechanism
    nf=sm.nf
    nc=sm.nc

    def __init__(self,pf,p0,p1,n_f, n_c):
        self.pf = pf
        self.pt[0] = p0
        self.pt[1] = p1
        self.nf[0] = n_f
        self.nc[0] =n_c
        fund = Fundamentalist(pf)
        chart = Chartist(p0, p1)
        sm = SwitchingMechanism(n_f,n_c)

