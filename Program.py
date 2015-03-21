__author__ = 'Radu'
import numpy
import matplotlib.pyplot as plt
import scipy.stats as sts

class Chartist:
    chi = 1.50
    sigma_c = 2.147
    epsilon_c = 0

    def demand(self, pt):
        self.epsilon_c = numpy.random.normal(0, self.sigma_c)
        demand = self.chi * (pt[-1] - pt[-2]) + self.epsilon_c
        return demand


class Fundamentalist:
    phi = 0.12
    sigma_f = 0.708
    p_f = 0
    epsilon_f = 0

    def demand(self, pt):
        self.epsilon_f = numpy.random.normal(0, self.sigma_f)
        demand = self.phi * (self.p_f - pt[-1]) + self.epsilon_f
        return demand


class MarketMaker:
    pf = 0  # Fundamental price
    price_t = []  # Price series
    nf = []  # Market Fraction of Fundamentalists
    nc = []  # Market Fraction of Technical Analysts
    x_t = []  # Majority index
    df = []  # Demands of Fundamentalists
    dc = []  # Demands of Chartists
    attract = []  # Attractiveness Levels
    return_t = []  # Returns

    fund = Fundamentalist
    chart = Chartist

    def __init__(self, p_0, p_1, nf_0, nc_0):
        self.price_t.append(p_0)
        self.price_t.append(p_1)
        self.nf.append(nf_0)
        self.nc.append(nc_0)
        self.x_t.append(nf_0 - nc_0)
        self.return_t.append(100 * (p_1 - p_0))

        self.fund = Fundamentalist()
        self.chart = Chartist()

    def update_demands(self):
        self.df.append(self.fund.demand(self.price_t))
        self.dc.append(self.chart.demand(self.price_t))

    @staticmethod
    def get_attractiveness(p_f, x_t, p_t):
        alpha_0 = -0.336
        alpha_n = 1.839
        alpha_p = 19.671
        a_t = alpha_0 + alpha_n * x_t + alpha_p * (p_t - p_f) ** 2
        return a_t

    def update_fractions(self, a_t):
        b = 1.00
        n_f = 1 / (1 + numpy.exp(-b * a_t))
        n_c = 1 - n_f
        self.nf.append(n_f)
        self.nc.append(n_c)
        self.x_t.append(n_f - n_c)

    def update_price(self):
        mu = 0.01
        self.update_demands()
        a = self.get_attractiveness(self.pf, self.x_t[-1], self.price_t[-1])
        self.attract.append(a)
        self.update_fractions(a)
        price = self.price_t[-1] + mu * (self.dc[-1] * self.nc[-1] + self.df[-1] * self.nf[-1])
        self.price_t.append(price)
        self.return_t.append(100 * (self.price_t[-1] - self.price_t[-2]))


MM = MarketMaker(0, 0, 0.5, 0.5)

for i in range(3000):
    MM.update_price()

print ('Kurtosis: ',sts.kurtosis(MM.price_t, fisher=False))
print ('Skewness: ',sts.skew(MM.price_t))

plt.figure()
plt.plot(MM.price_t)
plt.axhline(0, color='black', ls='dotted', lw=1)
plt.ylabel('log price')
plt.xlabel('time')
plt.title("Price Series")

plt.figure()
plt.plot(MM.x_t)
plt.axhline(0, color='black', ls='dotted', lw=1)
plt.ylabel('majority index')
plt.xlabel('time')
plt.title("Majority Index")

plt.figure()
plt.plot(MM.return_t)
plt.title("Returns")
plt.ylabel('return')
plt.xlabel('time')

plt.figure()
plt.subplot(2,1,1)
plt.plot(MM.price_t)
plt.axhline(0, color='black', ls='dotted', lw=1)
plt.title('log price')
plt.subplot(2,1,2)
plt.plot(MM.x_t)
plt.axhline(0, color='black', ls='dotted', lw=1)
plt.title('majority index')

print('mean price= ', numpy.mean(MM.price_t))
print('mean return= ',numpy.mean(MM.return_t))
print('mean majority index= ',numpy.mean(MM.x_t))
plt.show()


