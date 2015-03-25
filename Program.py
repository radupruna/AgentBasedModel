__author__ = 'Radu'
import numpy
import matplotlib.pyplot as plt
import scipy.stats as sts
from scipy.optimize import curve_fit
import time
import powerlaw.powerlaw as powerlaw
from statsmodels.tsa import stattools as tsast


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

    def __init__(self, p_0, p_1, nf_0, nc_0):
        self.pf = 0  # Fundamental price
        self.price_t = []  # Price series
        self.nf = []  # Market Fraction of Fundamentalists
        self.nc = []  # Market Fraction of Technical Analysts
        self.x_t = []  # Majority index
        self.df = []  # Demands of Fundamentalists
        self.dc = []  # Demands of Chartists
        self.attract = []  # Attractiveness Levels
        self.return_t = []  # Returns

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


def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    tau = [numpy.sqrt(numpy.std(numpy.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = numpy.polyfit(numpy.log(lags), numpy.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


def hill(pt):
    """Returns the Hill Tail index of the price series vector ts"""
    a = sorted(pt)
    n = len(a) - 1
    h = []
    for k in range(1, 500):
        s = 0
        for j in range(2, k):
            s = s + (numpy.log(a[n - j + 1]) - numpy.log(a[n - k]))
        h.append(s / k)
    return h


def dist_compare():
    """Pairwise comparisons between power_law, exponential, lognormal, truncated_power_law, stretched_exponential distributions"""
    xmins = []
    alphas = []
    sigmas = []
    Ds = []

    R_pw_exps = []
    p_pw_exps = []
    R_pw_logs = []
    p_pw_1ogs = []
    R_pw_trs = []
    p_pw_trs = []
    R_pw_srs = []
    p_pw_srs = []

    R_exp_logs = []
    p_exp_logs = []
    R_exp_trs = []
    p_exp_trs = []
    R_exp_srs = []
    p_exp_srs = []

    R_log_trs = []
    p_log_trs = []
    R_log_srs = []
    p_log_srs = []

    R_tr_srs = []
    p_tr_srs = []


    for j in range(501):
        MM = MarketMaker(0, 0, 0.5, 0.5)
        print('Iteration: ',j)
        for i in range(5999):
            MM.update_price()

        # Calculate absolute returns
        abs_returns = [abs(x) for x in MM.return_t]

        # Fit a power law distribution to absolute returns
        fit = powerlaw.Fit(abs_returns)
        # Calculating best minimal value for power law fit

        xmin = fit.xmin
        alpha = fit.power_law.alpha
        sigma = fit.power_law.sigma
        D = fit.power_law.D

        xmins.append(xmin)
        alphas.append(alpha)
        sigmas.append(sigma)
        Ds.append(D)

        R_pw_exp, p_pw_exp = fit.distribution_compare('power_law', 'exponential')
        R_pw_log, p_pw_1og = fit.distribution_compare('power_law', 'lognormal')
        R_pw_tr, p_pw_tr = fit.distribution_compare('power_law', 'truncated_power_law')
        R_pw_sr, p_pw_sr = fit.distribution_compare('power_law', 'stretched_exponential')

        R_pw_exps.append(float("{0:.4f}".format(R_pw_exp)))
        p_pw_exps.append(float("{0:.4f}".format(p_pw_exp)))
        R_pw_logs.append(float("{0:.4f}".format(R_pw_log)))
        p_pw_1ogs.append(float("{0:.4f}".format(p_pw_1og)))
        R_pw_trs.append(float("{0:.4f}".format(R_pw_tr)))
        p_pw_trs.append(float("{0:.4f}".format(p_pw_tr)))
        R_pw_srs.append(float("{0:.4f}".format(R_pw_sr)))
        p_pw_srs.append(float("{0:.4f}".format(p_pw_sr)))

        R_exp_log, p_exp_log = fit.distribution_compare('exponential', 'lognormal')
        R_exp_tr, p_exp_tr = fit.distribution_compare('exponential', 'truncated_power_law')
        R_exp_sr, p_exp_sr = fit.distribution_compare('exponential', 'stretched_exponential')

        R_exp_logs.append(float("{0:.4f}".format(R_exp_log)))
        p_exp_logs.append(float("{0:.4f}".format(p_exp_log)))
        R_exp_trs.append(float("{0:.4f}".format(R_exp_tr)))

        p_exp_trs.append(float("{0:.4f}".format(p_exp_tr)))
        R_exp_srs.append(float("{0:.4f}".format(R_exp_sr)))
        p_exp_srs.append(float("{0:.4f}".format(p_exp_sr)))

        R_log_tr, p_log_tr = fit.distribution_compare('lognormal', 'truncated_power_law')
        R_log_sr, p_log_sr = fit.distribution_compare('lognormal', 'stretched_exponential')

        R_log_trs.append(float("{0:.4f}".format(R_log_tr)))
        p_log_trs.append(float("{0:.4f}".format(p_log_tr)))
        R_log_srs.append(float("{0:.4f}".format(R_log_sr)))
        p_log_srs.append(float("{0:.4f}".format(p_log_sr)))

        R_tr_sr, p_tr_sr = fit.distribution_compare('truncated_power_law', 'stretched_exponential')

        R_tr_srs.append(float("{0:.4f}".format(R_tr_sr)))
        p_tr_srs.append(float("{0:.4f}".format(p_tr_sr)))

    print('xmin: ', numpy.median(xmins))
    print('alpha: ', numpy.median(alphas))
    print('sigma: ', numpy.median(sigmas))
    print('D: ', numpy.median(Ds))

    R_pw_exp = numpy.median(R_pw_exps)
    index = R_pw_exps.index(R_pw_exp)
    p_pw_exp = p_pw_exps[index]
    p_pw_exp1 = numpy.median(p_pw_exps)


    R_pw_log = numpy.median(R_pw_logs)
    index = R_pw_logs.index(R_pw_log)
    p_pw_1og = p_pw_1ogs[index]
    p_pw_1og1 = numpy.median(p_pw_1ogs)

    R_pw_tr = numpy.median(R_pw_trs)
    index = R_pw_trs.index(R_pw_tr)
    p_pw_tr = p_pw_trs[index]
    p_pw_tr1 = numpy.median(p_pw_trs)

    R_pw_sr = numpy.median(R_pw_srs)
    index = R_pw_srs.index(R_pw_sr)
    p_pw_sr = p_pw_srs[index]
    p_pw_sr1 = numpy.median(p_pw_srs)

    R_exp_log = numpy.median(R_exp_logs)
    index = R_exp_logs.index(R_exp_log)
    p_exp_log = p_exp_logs[index]
    p_exp_log1 = numpy.median(p_exp_logs)

    R_exp_tr = numpy.median(R_exp_trs)
    index = R_exp_trs.index(R_exp_tr)
    p_exp_tr = p_exp_trs[index]
    p_exp_tr1 = numpy.median(p_exp_trs)

    R_exp_sr = numpy.median(R_exp_srs)
    index = R_exp_srs.index(R_exp_sr)
    p_exp_sr = p_exp_srs[index]
    p_exp_sr1 = numpy.median(p_exp_srs)

    R_log_tr = numpy.median(R_log_trs)
    index = R_log_trs.index(R_log_tr)
    p_log_tr = p_log_trs[index]
    p_log_tr1 = numpy.median(p_log_trs)

    R_log_sr = numpy.median(R_log_srs)
    index = R_log_srs.index(R_log_sr)
    p_log_sr = p_log_srs[index]
    p_log_sr1 = numpy.median(p_log_srs)

    R_tr_sr = numpy.median(R_tr_srs)
    index = R_tr_srs.index(R_tr_sr)
    p_tr_sr = p_tr_srs[index]
    p_tr_sr1 = numpy.median(p_tr_srs)

    print('R_pw_exp: ', R_pw_exp)
    print('p_pw_exp: ', p_pw_exp)
    print('p_pw_exp1: ', p_pw_exp1)
    print('R_pw_log: ', R_pw_log)
    print('p_pw_1og: ', p_pw_1og)
    print('p_pw_1og1: ', p_pw_1og1)
    print('R_pw_tr: ', R_pw_tr)
    print('p_pw_tr: ', p_pw_tr)
    print('p_pw_tr1: ', p_pw_tr1)
    print('R_pw_sr: ', R_pw_sr)
    print('p_pw_sr: ', p_pw_sr)
    print('p_pw_sr1: ', p_pw_sr1)

    print('R_exp_log: ', R_exp_log)
    print('p_exp_log: ', p_exp_log)
    print('p_exp_log1: ', p_exp_log1)
    print('R_exp_tr: ', R_exp_tr)
    print('p_exp_tr: ', p_exp_tr)
    print('p_exp_tr1: ', p_exp_tr1)
    print('R_exp_sr: ', R_exp_sr)
    print('p_exp_sr: ', p_exp_sr)
    print('p_exp_sr1: ', p_exp_sr1)

    print('R_log_tr: ', R_log_tr)
    print('p_log_tr: ', p_log_tr)
    print('p_log_tr1: ', p_log_tr1)
    print('R_log_sr: ', R_log_sr)
    print('p_log_sr: ', p_log_sr)
    print('p_log_sr1: ', p_log_sr1)

    print('R_tr_sr: ', R_tr_sr)
    print('p_tr_sr: ', p_tr_sr)
    print('p_tr_sr1: ', p_tr_sr1)

def kurt_skew():
    kurt=[]
    skewness=[]

    for j in range(10000):
        MM=MarketMaker(0, 0, 0.5, 0.5)
        print('iteration', j)
        for i in range(5999):
            MM.update_price()
        kurt.append(sts.kurtosis(MM.return_t))
        skewness.append(sts.skew(MM.return_t))
    print('Kurtosis: ', float("{0:.4f}".format(numpy.median(kurt))))
    print('Skewness: ', float("{0:.4f}".format(numpy.median(skewness))))

def power_fitting_time():
    # Abs Returns as a function of time assumes return = a* time**b
    # pars = Optimal values for the parameters so that the sum of the squared error of f(xdata, *pars) - ydata is minimized
    # covar = the estimated covariance of pars. The diagonals provide the variance of the parameter estimate.
    # err = compute one standard deviation errors (scaled) on the parameters u.
    def powerlaw(x,a,b):
        return a*(x**b)
    MM=MarketMaker(0, 0, 0.5, 0.5)
    for i in range(5999):
        MM.update_price()
    t=numpy.linspace(0,5999,6000)

    abs_returns = [abs(x) for x in MM.return_t]
    pars,covar =  curve_fit(powerlaw,t,abs_returns)
    err = numpy.sqrt(numpy.diag(covar))

    print('pars: ', pars)
    print('covar: ', covar)
    print('error:', err)

def autocorrelation_function():
    MM=MarketMaker(0, 0, 0.5, 0.5)
    for i in range(5*5999):
        MM.update_price()
    abs_returns = [abs(x) for x in MM.return_t]
    a = tsast.acf(abs_returns,nlags=99)
    b= tsast.acf(MM.return_t,nlags=99)


    plt.figure()
    plt.ylim(-0.1,0.3)
    plt.plot(a,label='abs returns')
    plt.plot(b, label='raw returns')
    plt.title('Autocorrelation function')
    plt.xlabel('lags')
    plt.ylabel('autocorrelation')

t0 = time.time()
autocorrelation_function()
t1 = time.time()
total = t1 - t0
print('time: ', total)
plt.legend()
plt.show()
# MM = MarketMaker(0, 0, 0.5, 0.5)
#
# for i in range(5998):
#     MM.update_price()
# abs_returns = [abs(x) for x in MM.return_t]
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(MM.return_t)
# plt.subplot(2, 1, 2)
# plt.plot(abs_returns)
# plt.show()
# # Fit a power law distribution
# fit=powerlaw.Fit(MM.return_t)
# # Calculating best minimal value for power law fit
# print('xmin: ',fit.xmin)
# print('fixed?: ',fit.fixed_xmin)
# print('alpha: ',fit.power_law.alpha)
# print('sigma: ',fit.power_law.sigma)
# print('D: ',fit.power_law.D)
# R_exp, p_exp = fit.distribution_compare('power_law', 'exponential')
# print ('R_exp, p_exp',R_exp,p_exp)
# R_log, p_1og = fit.distribution_compare('power_law', 'lognormal')
# print ('R_log, p_1og',R_log,p_1og)
# R_tr, p_tr = fit.distribution_compare('power_law', 'truncated_power_law')
# print ('R_tr, p_tr',R_tr,p_tr)
# R_sr, p_sr = fit.distribution_compare('power_law', 'stretched_exponential')
# print ('R_sr, p_sr',R_sr,p_sr)
#
#
# print('Price Kurtosis: ', sts.kurtosis(MM.price_t))
# print('Price Skewness: ', sts.skew(MM.price_t))
#
# print('Returns Kurtosis: ', sts.kurtosis(MM.return_t))
# print('Returns Skewness: ', sts.skew(MM.return_t))
#
# plt.figure()
# plt.plot(MM.price_t)
# plt.axhline(0, color='black', ls='dotted', lw=1)
# plt.ylabel('log price')
# plt.xlabel('time')
# plt.title("Price Series")
#
# plt.figure()
# plt.plot(MM.x_t)
# plt.axhline(0, color='black', ls='dotted', lw=1)
# plt.ylabel('majority index')
# plt.xlabel('time')
# plt.title("Majority Index")
#
# plt.figure()
# plt.plot(MM.return_t)
# plt.title("Returns")
# plt.ylabel('return')
# plt.xlabel('time')
#
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(MM.price_t)
# plt.axhline(0, color='black', ls='dotted', lw=1)
# plt.title('log price')
# plt.subplot(2, 1, 2)
# plt.plot(MM.x_t)
# plt.axhline(0, color='black', ls='dotted', lw=1)
# plt.title('majority index')
# plt.show()

