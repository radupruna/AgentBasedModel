__author__ = 'Radu'
import numpy
import matplotlib.pyplot as plt


class Chartist:
    chi = 1.50
    sigma_c = 2.147
    epsilon_c = 0

    def getDemand(self, pt):
        self.epsilon_c = numpy.random.normal(0, self.sigma_c)
        demand = self.chi * (pt[-1] - pt[-2]) + self.epsilon_c
        return demand


class Fundamentalist:
    phi = 0.12
    sigma_f = 0.708
    p_f = 0
    epsilon_f = 0

    def getDemand(self, pt):
        self.epsilon_f = numpy.random.normal(0, self.sigma_f)
        demand = self.phi * (self.p_f - pt[-1]) + self.epsilon_f
        return demand


class MarketMaker:
    pf = 0  # Fundamental price
    pt = []  # Price series
    nf = []  # Market Fraction of Fundamentalists
    nc = []  # Market Fraction of Technical Analysts
    xt = []  # Majority index
    df = []  # Demands of Fundamentalists
    dc = []  # Demands of Chartists
    attract = []  # Attractiveness Levels
    rt = []  # Returns

    fund = Fundamentalist
    chart = Chartist

    def __init__(self, pf, p_0, p_1, nf_0, nc_0):
        self.pt.append(p_0)
        self.pt.append(p_1)
        self.nf.append(nf_0)
        self.nc.append(nc_0)
        self.xt.append(nf_0 - nc_0)
        self.rt.append(100 * (p_1 - p_0))

        self.fund = Fundamentalist()
        self.chart = Chartist()

    def updateDemands(self):
        self.df.append(self.fund.getDemand(self.pt))
        self.dc.append(self.chart.getDemand(self.pt))

    def getAttractLvl(self, p_f, x_t, p_t):
        alpha_0 = -0.336
        alpha_n = 1.839
        alpha_p = 19.671
        a_t = alpha_0 + alpha_n * x_t + alpha_p * (p_t - p_f)
        return a_t

    def updateFractions(self, a_t):
        b = 1.00
        n_f = 1 / (1 + numpy.exp(-b * a_t))
        n_c = 1 - n_f
        self.nf.append(n_f)
        self.nc.append(n_c)
        self.xt.append(n_f - n_c)

    def getNf(self):
        return self.nf

    def getNc(self):
        return self.nc

    def updatePrice(self):
        mu = 0.01
        self.updateDemands()
        a = self.getAttractLvl(self.pf, self.xt[-1], self.pt[-1])
        self.attract.append(a)
        self.updateFractions(a)
        price = self.pt[-1] + mu * (self.dc[-1] * self.nc[-1] + self.df[-1] * self.nf[-1])
        self.pt.append(price)
        self.rt.append(100 * (self.pt[-1] - self.pt[-2]))


MM = MarketMaker(0, 0.5, 0.5, 0.5, 0.5)