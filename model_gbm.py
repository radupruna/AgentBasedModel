__author__ = 'Radu'
import numpy
import matplotlib.pyplot as plt
import scipy.stats as sts
import scipy
from scipy.optimize import curve_fit
import powerlaw.powerlaw as powerlaw
import matplotlib.mlab as mlab
from networkx.utils import powerlaw_sequence
from statsmodels.tsa import stattools as tsast


class Chartist:
    chi = 1.50
    sigma_c = 2.147
    epsilon_c = 0

    def __init__(self):
        self.dc = []
        self.ut = []

    def update_demand(self, pt):
        self.epsilon_c = numpy.random.normal(0, self.sigma_c)
        demand = self.chi * (pt[-1] - pt[-2]) + self.epsilon_c
        self.dc.append(demand)
        return demand

    def update_utility(self, pt, demand):
        self.ut.append((pt[-1] - pt[-2]) * demand[-1])


""" Simple Moving Averages MA(n)
"""


class SMA:
    def __init__(self, pt, n):
        self.averages = []
        self.n = n
        self.pt = pt

    def update_averages(self):
        if (len(self.pt) >= self.n):
            self.averages.append(numpy.average(self.pt[(len(self.pt) - self.n): len(self.pt)]))


""" Exponential Moving Averages EMA(n)
"""


class EMA:
    def __init__(self, pt, n):
        self.averages = []
        self.n = n
        self.pt = pt
        self.alpha = 2 / (n + 1)

    def update_averages(self):
        if (len(self.pt) == self.n):
            FirstEMA = numpy.average(self.pt[0: len(self.pt)])
            self.averages.append(FirstEMA)
        elif (len(self.pt) > self.n):
            EMA_today = (self.pt[-1] - self.averages[-1] ) * self.alpha + self.averages[-1]
            self.averages.append(EMA_today)


class Fundamentalist:
    phi = 0.12
    sigma_f = 0.708
    epsilon_f = 0

    def __init__(self):
        self.df = []
        self.ut = []

    def update_demand(self, pf, pt):
        self.epsilon_f = numpy.random.normal(0, self.sigma_f)
        demand = self.phi * (pf[-1] - pt[-1]) + self.epsilon_f
        return demand

    def update_utility(self, pt, demand):
        self.ut.append((pt[-1] - pt[-2]) * demand[-1])


class MarketMaker:
    def __init__(self, p_0, p_1, nf_0, nc_0):
        self.pf = []
        self.price_t = []  # Price series
        self.nf = []  # Market Fraction of Fundamentalists
        self.nc = []  # Market Fraction of Technical Analysts
        self.x_t = []  # Majority index
        self.df = []  # Demands of Fundamentalists
        self.dc = []  # Demands of Chartists
        self.volume = []
        self.attract = []  # Attractiveness Levels
        self.return_t = []  # Returns
        self.volatility = []

        pf_0=1
        self.pf.append(pf_0)
        self.price_t.append(p_0)
        self.price_t.append(p_1)
        self.nf.append(nf_0)
        self.nc.append(nc_0)
        self.x_t.append(nf_0 - nc_0)

        self.fund = Fundamentalist()
        self.chart = Chartist()

        # self.SMA200 = SMA(self.price_t, 200)
        # self.EMA12 = SMA(self.price_t, 12)
        # self.EMA26 = EMA(self.price_t, 26)

    def update_demands(self):
        self.df.append(self.fund.update_demand(self.pf, self.price_t))
        self.dc.append(self.chart.update_demand(self.price_t))

    def get_attractiveness(self, p_f, x_t, p_t):
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
        a = self.get_attractiveness(self.pf[-1], self.x_t[-1], self.price_t[-1])
        self.attract.append(a)
        self.update_fractions(a)

        m = 0.015  #drift
        sig = 0.075 #volatility
        fund_price = generate_next_gbm(self.pf[-1], m, sig)
        self.pf.append(fund_price)

        price = self.price_t[-1] + mu * (self.dc[-1] * self.nc[-1] + self.df[-1] * self.nf[-1])
        self.price_t.append(price)

        # self.SMA200.update_averages()
        # self.EMA12.update_averages()
        # self.EMA26.update_averages()

        self.volume.append(abs(self.dc[-1] * self.nc[-1]) + abs(self.df[-1] * self.nf[-1]))
        self.return_t.append(100 * (self.price_t[-1] - self.price_t[-2]))
        self.volatility.append((abs(self.return_t[-1])))
        self.fund.update_utility(self.price_t, self.df)
        self.chart.update_utility(self.price_t, self.dc)


"""Geometric Brownian Motion prices
    delta is the duration in days between each generated price, and sigma and mu are annualised values - i.e. delta = 1.0 generates daily prices.
    \mu ('the percentage drift') and  \sigma ('the percentage volatility')
"""


def generate_next_gbm(prevSt, mu, sigma):
    delta = 1
    DAYS_PER_YEAR = 252.0
    t = delta / DAYS_PER_YEAR
    W = numpy.random.normal(0, 1) * numpy.sqrt(t)
    Y = (mu - 0.5 * sigma ** 2) * t + sigma * W
    price = prevSt * numpy.exp(Y)
    return price


""" Fit a powerlaw distribution p(x) = x^-alpha to simulated data and S&P 500 data.
    Plot multiple local minima of Kolmogorov-Smirnov distance D across xmin.
    @return xmin, alpha, sigma, KS statistic D.

"""


def powerlaw_fit():
    xmins = []
    alphas = []
    sigmas = []
    Ds = []

    # with open('abs_returns.txt') as g:
    #     s_p_abs = g.readlines()
    # s_p_abs_returns = [float(x) for x in s_p_abs if float(x) > 0]
    # s_p_squared_returns = [x ** 2 for x in s_p_abs_returns]
    # d = tsast.acf(s_p_squared_returns, nlags=99)
    # fitSP = powerlaw.Fit(d)
    # print('xmin S&P: ', fitSP.xmin)
    # print('alpha S&P: ', fitSP.power_law.alpha)
    # print('sigma S&P: ', fitSP.power_law.sigma)
    # print('D S&P: ', fitSP.power_law.D)
    #
    # # Example of multiple local minima of Kolmogorov-Smirnov distance D across xmin
    # plt.figure()
    # plt.plot(fitSP.xmins, fitSP.Ds, 'b', label='D')
    # plt.plot(fitSP.xmins, fitSP.sigmas, 'g--', label='sigma')
    # plt.plot(fitSP.xmins, fitSP.sigmas / fitSP.alphas, 'r--', label='sigma/alpha')
    # plt.xlabel('xmin')
    # plt.title('S&P 500 data')
    # plt.legend(loc=2)
    # plt.ylim(0, 0.6)

    for j in range(1):
        MM = MarketMaker(1, 1, 0.5, 0.5)
        print('Iteration: ', j)
        for i in range(6000):
            MM.update_price()

        # Calculate absolute returns
        abs_returns = [abs(x) for x in MM.return_t if abs(x) > 0]

        # Fit a power law distribution to absolute returns
        fit = powerlaw.Fit(MM.volume)
        # Calculating best minimal value for power law fit
        xmin = fit.xmin
        alpha = fit.power_law.alpha
        sigma = fit.power_law.sigma
        D = fit.power_law.D

        xmins.append(xmin)
        alphas.append(alpha)
        sigmas.append(sigma)
        Ds.append(D)

        # Example of multiple local minima of Kolmogorov-Smirnov distance D across xmin
        # plt.figure()
        # plt.plot(fit.xmins, fit.Ds, 'b', label='D')
        # plt.plot(fit.xmins, fit.sigmas, 'g--', label='sigma')
        # plt.plot(fit.xmins, fit.sigmas / fit.alphas, 'r--', label='sigma/alpha')
        # plt.xlabel('xmin')
        # plt.title('Simulated data')
        # plt.legend(loc=2)
        # plt.ylim(0, 0.6)

    print('xmin: ', numpy.median(xmins))
    print('alpha: ', numpy.median(alphas))
    print('sigma: ', numpy.median(sigmas))
    print('D: ', numpy.median(Ds))


""" Pairwise comparisons between data and power_law, exponential, lognormal, truncated_power_law, stretched_exponential distributions
    @return R the loglikelihood ratio between the two candidate distributions.
    R>0 if the data is more likely in the first distribution, R<0 if the data is more likely in the second distribution.
    @return p the significance value for that direction
"""


def distibution_compare():
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

    for j in range(1001):
        MM = MarketMaker(0, 0, 0.5, 0.5)
        print('Iteration: ', j)
        for i in range(5999):
            MM.update_price()

        # Calculate absolute returns
        abs_returns = [abs(x) for x in MM.return_t]

        # Fit a power law distribution to absolute returns
        fit = powerlaw.Fit(abs_returns)
        # Calculating best minimal value for power law fit

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


""" Kurtosis and Skewness
"""


def kurt_skew():
    kurt = []
    skewness = []

    with open('raw_returns.txt') as g:
        s_p_raw = g.readlines()
    s_p_raw_returns = [float(x) for x in s_p_raw]
    print('S&P Kurtosis:', sts.kurtosis(s_p_raw_returns))
    print('S&P Skewness:', sts.skew(s_p_raw_returns))

    for j in range(500):
        MM = MarketMaker(1, 1, 0.5, 0.5)
        print('iteration', j)
        for i in range(6000):
            MM.update_price()
        kurt.append(sts.kurtosis(MM.return_t))
        skewness.append(sts.skew(MM.return_t))
    print('Kurtosis: ', float("{0:.4f}".format(numpy.median(kurt))))
    print('Skewness: ', float("{0:.4f}".format(numpy.median(skewness))))


""" LONG RANGE dependency
    Autocorrelation decay a function of time assumes return = time**(-a)
    curve_fit(f, xdata, ydata)
    Assumes ydata = f(xdata, *params) + eps

    pars= Optimal values for the parameters so that the sum of the squared error of f(xdata, *popt) - ydata is minimized

    @return pars = Optimal values for the parameters so that the sum of the squared error of f(xdata, *pars) - ydata is minimized
    @return covar = the estimated covariance of pars. The diagonals provide the variance of the parameter estimate.
    @return err = compute one standard deviation errors (scaled) on the parameters u.
"""


def power_fitting_time():
    def powerlaw(x, a):
        return (x**(-a))
    abs_par=[]
    abs_err=[]
    sq_par=[]
    sq_err=[]
    t = numpy.linspace(1, 100, 100)

    with open('raw_returns.txt') as f:
        s_p_raw = f.readlines()
    with open('abs_returns.txt') as g:
        s_p_abs = g.readlines()
    s_p_raw_returns = [float(x) for x in s_p_raw]
    s_p_abs_returns = [float(x) for x in s_p_abs]
    s_p_squared_returns = [x ** 2 for x in s_p_raw_returns]

    d = tsast.acf(s_p_abs_returns, nlags=99)
    sp_abs_pars, sp_abs_covar = curve_fit(powerlaw, t, d, p0=0.3)
    sp_abs_err = numpy.sqrt(numpy.diag(sp_abs_covar))
    print(sp_abs_pars,sp_abs_err)

    c = tsast.acf(s_p_squared_returns, nlags=99)
    sp_sq_pars, sp_sq_covar = curve_fit(powerlaw, t, c, p0=0.3)
    sp_sq_err = numpy.sqrt(numpy.diag(sp_sq_covar))
    print(sp_sq_pars,sp_sq_err)

    for j in range(11):
        print('iteration ',j)
        MM = MarketMaker(1, 1, 0.5, 0.5)
        for i in range(5999):
            MM.update_price()

        abs_returns = [abs(x) for x in MM.return_t]
        squared_returns = [x ** 2 for x in MM.return_t]

        a = tsast.acf(abs_returns, nlags=99)
        pars, covar = curve_fit(powerlaw, t, a, p0=0.3)
        err = numpy.sqrt(numpy.diag(covar))

        abs_par.append(float("{0:.4f}".format(pars[0])))
        abs_err.append(float("{0:.4f}".format(err[0])))

        b = tsast.acf(squared_returns, nlags=99)
        pars1, covar1 = curve_fit(powerlaw, t, b, p0=0.3)
        err1 = numpy.sqrt(numpy.diag(covar1))
        sq_par.append(float("{0:.4f}".format(pars1[0])))
        sq_err.append(float("{0:.4f}".format(err1[0])))

    abs_p = numpy.median(abs_par)
    ind = abs_par.index(abs_p)
    abs_e = abs_err[ind]
    print('abs param:' , abs_p)
    print('abs err: ',abs_e)

    sq_p = numpy.median(sq_par)
    ind = sq_par.index(sq_p)
    sq_e = sq_err[ind]
    print('sq param:' , sq_p)
    print('sq err: ',sq_e)


""" Returns the exponent of a fitted PowerLaw distribution using Maximum Likelihood Method
    This estimator is equivalent to the Hill estimator+
    @param returns
    @param xmin
    @return Hill Tail index
"""


def hill_index():
    alphas = []
    alphas95 = []
    alphas1 = []
    for j in range(11):
        MM = MarketMaker(1, 1, 0.5, 0.5)
        for i in range(8500 - 1):
            MM.update_price()
        abs_returns = [abs(x) for x in MM.return_t]

        rs = sorted(abs_returns)
        xmin = 1.2658
        rs1 = [x for x in rs if x > xmin]
        n = len(rs1)
        sum = 0
        for i in range(n):
            sum = sum + numpy.log(rs1[i] / xmin)
        alpha = 1 + n * sum ** -1
        alphas.append(alpha)

        rs95 = rs[8075:]
        n95 = len(rs95)
        sum95 = 0
        xmin95 = rs[8075]
        for j in range(n95):
            sum95 = sum95 + numpy.log(rs95[j] / xmin95)
        alpha95 = 1 + n95 * sum95 ** -1
        alphas95.append(alpha95)

        with open('abs_returns.txt') as g:
            s_p_abs = g.readlines()
        s_p_abs_returns = [float(x) for x in s_p_abs]
        returns = sorted(s_p_abs_returns)
        slice = len(rs1) / len(rs)
        nsp = len(returns) - int(slice * len(returns))
        print(nsp)
        returns1 = returns[nsp:]
        sum1 = 0
        xmin1 = returns[nsp]
        for j in range(len(returns1)):
            sum1 = sum1 + numpy.log(returns1[j] / xmin1)
        alpha1 = 1 + len(returns1) * sum1 ** -1
        alphas1.append(alpha1)

    print('Hill index S&P', numpy.median(alphas1))
    print('Hill Index: ', numpy.median(alphas))
    print('Hill Index upper 5%: ', numpy.median(alphas95))



""" Autocorrelation function plot for volume , raw,abs returns, S&P raw,abs returns
"""


def autocorrelations():
    MM = MarketMaker(1, 1, 0.5, 0.5)
    for i in range(6000):
        MM.update_price()
    abs_returns = [abs(x) for x in MM.return_t]
    squared_returns = [x ** 2 for x in MM.return_t]

    a = tsast.acf(abs_returns, nlags=99)
    b = tsast.acf(MM.return_t, nlags=99)

    with open('raw_returns.txt') as f:
        s_p_raw = f.readlines()
    with open('abs_returns.txt') as g:
        s_p_abs = g.readlines()
    s_p_raw_returns = [float(x) for x in s_p_raw]
    s_p_abs_returns = [float(x) for x in s_p_abs]
    s_p_squared_returns = [x ** 2 for x in s_p_raw_returns]

    c = tsast.acf(s_p_raw_returns, nlags=99)
    d = tsast.acf(s_p_abs_returns, nlags=99)

    fitAbs = powerlaw.Fit(a)
    alpha = fitAbs.power_law.alpha
    print('alpha abs: ', fitAbs.power_law.alpha)
    pl_sequence = powerlaw_sequence(100, exponent=alpha)
    pl_sequence.sort(reverse=True)
    pl_sequence = [x / 10 for x in pl_sequence]

    plt.figure()
    beta = 1 / alpha
    plt.plot(a, 'r', label='abs returns')
    plt.plot(pl_sequence, label='shifted power law beta = %f' % beta)
    plt.plot(d, 'r--', label='S&P abs returns')
    plt.title('Absolute returns autocorrelation decay')
    plt.xlabel('lags')
    plt.ylabel('autocorrelation')
    plt.legend()

    plt.figure()
    plt.ylim(-0.1, 0.3)
    plt.plot(a, 'r', label='abs returns')
    plt.plot(d, 'r--', label='S&P abs returns')
    plt.plot(b, 'b', label='raw returns')
    plt.plot(c, 'b--', label='S&P raw returns')
    plt.axhline(0, 0, 1, color='black', ls='dotted', lw=1)
    plt.grid(True)
    plt.title('Returns autocorrelation function')
    plt.xlabel('lags')
    plt.ylabel('autocorrelation')
    plt.legend()

    v = tsast.acf(MM.volume, nlags=10)
    plt.figure()
    plt.plot(v, label='volume')
    plt.axhline(0, 0, 1, color='black', ls='dotted', lw=1)
    plt.grid(True)
    plt.xlabel('lags')
    plt.ylabel('autocorrelation')
    plt.title('Volume autocorrelation function')
    plt.legend()

    sr = tsast.acf(squared_returns, nlags=100)
    spsr = tsast.acf(s_p_squared_returns, nlags=100)
    plt.figure()
    plt.ylim(-0.1, 0.4)
    plt.plot(sr, 'r', label='squared returns')
    plt.plot(spsr, 'r--', label='S&P squared returns')
    plt.axhline(0, 0, 1, color='black', ls='dotted', lw=1)
    plt.grid(True)
    plt.title('Squared returns autocorrelation function')
    plt.xlabel('lags')
    plt.ylabel('autocorrelation')
    plt.legend()

    fitSq = powerlaw.Fit(sr)
    alphaSq = fitAbs.power_law.alpha
    print('alpha squared: ', fitSq.power_law.alpha)
    pl_sequence_sq = powerlaw_sequence(100, exponent=alphaSq)
    pl_sequence_sq.sort(reverse=True)
    pl_sequence_sq = [x / 10 for x in pl_sequence_sq]

    plt.figure()
    betaSq = 1 / alphaSq
    plt.plot(sr, 'r', label='squared returns')
    plt.plot(spsr, 'r--', label='S&P squared returns')
    plt.plot(pl_sequence_sq, label='shifted power law beta = %f' % betaSq)
    plt.title('Squared returns autocorrelation decay')
    plt.xlabel('lags')
    plt.ylabel('autocorrelation')
    plt.legend()
    plt.show()

""" Anderson-Darling Test
    Works for exponential, normal, logistic, extreme 1 and Gumbel distributions
    If A2 is larger than these critical values then for the corresponding significance level,
    the null hypothesis that the data come from the chosen distribution can be rejected.
    @param returns
    @param distribution
    @return A2 : Anderson-Darling test statistic
    @return critical : the critical values for this distribution
    @return: The significance levels for the corresponding critical values in percents.
"""


def anderson_test(distribution):
    MM = MarketMaker(0, 0, 0.5, 0.5)
    for i in range(5998):
        MM.update_price()
    abs_returns = [abs(x) for x in MM.return_t]
    print('Anderson Test data vs ', distribution, ': ', sts.anderson(abs_returns, distribution))


""" Kolmogorov-Smirnov Test
    Returns the D test value and p-value
    @param returns
    @param distribution
    @return D: KS test statistic, either D, D+ or D-.
    @return p-value : One-tailed or two-tailed p-value.
"""


def kstst(distribution):
    abs_expon_D = []
    abs_expon_p = []
    for j in range(101):
        MM = MarketMaker(0, 0, 0.5, 0.5)
        print('Iteration', j)
        for i in range(5999):
            MM.update_price()
        abs_returns = [abs(x) for x in MM.return_t]
        [D1, p1] = sts.kstest(abs_returns, 'distribution')
        abs_expon_D.append(D1)
        abs_expon_p.append(p1)
    print('K-S test data vs ', distribution, ': ', numpy.median(abs_expon_D),
          abs_expon_p[abs_expon_D.index(numpy.median(abs_expon_D))])


""" Returns the Hurst Exponent used to measure long-memory of time series X
"""


def hurst(X):
    """ Compute the Hurst exponent of X. If the output H=0.5,the behavior
    of the time-series is similar to random walk. If H<0.5, the time-series
    cover less "distance" than a random walk and vice verse.
    """
    N = len(X)
    T = numpy.array([float(i) for i in range(1, N + 1)])
    Y = numpy.cumsum(X)
    Ave_T = Y / T

    S_T = numpy.zeros((N))
    R_T = numpy.zeros((N))
    for i in range(N):
        S_T[i] = numpy.std(X[:i + 1])
        X_T = Y - T * Ave_T[i]
        R_T[i] = max(X_T[:i + 1]) - min(X_T[:i + 1])

    R_S = R_T / S_T
    R_S = numpy.log(R_S)
    n = numpy.log(T).reshape(N, 1)
    H = numpy.linalg.lstsq(n[1:], R_S[1:])[0]
    return H[0]


""" Calculate quantiles for a probability plot, and optionally show the plot.
    Assessing how closely two data sets agree, plots the two cumulative distribution functions against each other.
    Generates a probability plot of sample data against the quantiles of a specified theoretical distribution
    Calculates a best-fit line for the data and plots the results
    The distributions are equal if and only if the plot falls on this line – any deviation indicates a difference between the distributions.
"""


def pp_plot(distribution):
    MM = MarketMaker(1, 1, 0.5, 0.5)
    for i in range(5999):
        MM.update_price()

    abs_returns = [abs(x) for x in MM.return_t if abs(x) > 0]

    plt.figure()
    sts.probplot(abs_returns, dist=distribution, plot=plt)
    plt.title('P-P plot abs returns vs ' + distribution + ' distribution')
    plt.figure()
    sts.probplot(MM.return_t, dist=distribution, plot=plt)
    plt.title('P-P plot raw returns vs ' + distribution + ' distribution')

    with open('raw_returns.txt') as g:
        s_p_raw = g.readlines()
    s_p_raw_returns = [100 * float(x) for x in s_p_raw]
    plt.figure()
    sts.probplot(s_p_raw_returns, dist=distribution, plot=plt)
    plt.title('P-P plot S&P raw returns vs ' + distribution + ' distribution')

    with open('abs_returns.txt') as g:
        s_p_abs = g.readlines()
    s_p_abs_returns = [100 * float(x) for x in s_p_abs]
    plt.figure()
    sts.probplot(s_p_abs_returns, dist=distribution, plot=plt)
    plt.title('P-P plot S&P abs returns vs ' + distribution + ' distribution')

    plt.show()


""" Creates histogram plots for the simulated and empirical data
"""


def histograms():
    MM = MarketMaker(1, 1, 0.5, 0.5)
    for i in range(5999):
        MM.update_price()
    abs_returns = [abs(x) for x in MM.return_t]

    with open('raw_returns.txt') as g:
        s_p_raw = g.readlines()
    s_p_raw_returns = [100 * float(x) for x in s_p_raw]
    with open('abs_returns.txt') as g:
        s_p_abs = g.readlines()
    s_p_abs_returns = [100 * float(x) for x in s_p_abs]

    plt.figure()
    (mu, sigma) = sts.norm.fit(MM.return_t)
    # the histogram of the data
    # normalized to form a probability density, i.e., n/(len(x)`dbin), i.e., the integral of the histogram will sum to 1
    n, bins, patches = plt.hist(MM.return_t, 100, normed=1, facecolor='green', alpha=0.75)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel('returns')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ raw\ returns:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
    plt.grid(True)

    x=[]
    for i in range(1,len(MM.pf)):
        x.append((MM.pf[i]-MM.pf[i-1]))
    plt.figure()
    (mu, sigma) = sts.norm.fit(x)
    # the histogram of the data
     # normalized to form a probability density, i.e., n/(len(x)`dbin), i.e., the integral of the histogram will sum to 1
    n, bins, patches = plt.hist(x, 100, normed=1, facecolor='green', alpha=0.75)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel('difference in log pf')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ pf\ returns:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))

    plt.figure()
    n, bins, patches = plt.hist(abs_returns, 100, normed=1, facecolor='green', alpha=0.75)
    (mu, sigma) = sts.norm.fit(abs_returns)
    plt.xlabel('volatility')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ abs\ returns:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))

    plt.figure()
    n, bins, patches = plt.hist(MM.volume, 100, normed=1, facecolor='green', alpha=0.75)
    (mu, sigma) = sts.norm.fit(MM.volume)
    plt.xlabel('Volume')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ volume}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
    print(numpy.mean(MM.volume), numpy.std(MM.volume))

    plt.figure()
    (mu, sigma) = sts.norm.fit(s_p_raw_returns)
    # the histogram of the data
    # normalized to form a probability density, i.e., n/(len(x)`dbin), i.e., the integral of the histogram will sum to 1
    n, bins, patches = plt.hist(s_p_raw_returns, 100, normed=1, facecolor='green', alpha=0.75)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel('S&P returns')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ S&P\ raw\ returns:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))

    plt.figure()
    (mu, sigma) = sts.norm.fit(s_p_abs_returns)
    n, bins, patches = plt.hist(s_p_abs_returns, 100, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('S&P volatility')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ S&P\ abs\ returns:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))

    for i in range(1):
        MM = MarketMaker(1, 1, 0.5, 0.5)
        for i in range(5999):
            MM.update_price()

        abs_returns = [abs(x) + 1 for x in MM.return_t if abs(x) > 0.0]
        a, n, c = sts.pareto.fit(abs_returns)
        plt.figure()
        count, bins, patches = plt.hist(abs_returns, 100, normed=1, facecolor='green', alpha=0.75, label='raw returns')
        fit = a * n * a / bins ** (a + 1)
        plt.plot(bins + 1, max(count) * fit / ((max(fit)) * (len(bins)) * (a + 1.2)), linewidth=2, color='r',
                 label='pareto dist')
        plt.title('Raw returns vs Pareto distr')
        plt.legend()

    plt.show()


""" Crosscorrelations between volume, returns, volatility, squared returns
"""


def crosscorrelations():
    MM = MarketMaker(1, 1, 0.5, 0.5)
    for i in range(5999):
        MM.update_price()
    squared_returns = [x ** 2 for x in MM.return_t]

    # Leverage Effect
    plt.figure()
    plt.xcorr(MM.volatility, MM.return_t,maxlags=100, usevlines=True)
    plt.legend()
    plt.xlabel('Volatility(t) and Return(t+j)')
    plt.ylabel('cross correlation')
    plt.xticks(range(-100, 101, 10))
    plt.grid(True)
    plt.title('Leverage Effect')
    # b = tsast.ccf(MM.volume, MM.volatility)
    # plt.figure()
    # plt.plot(b)
    # plt.xlabel('volume(t) and volatility(t+j)')
    # plt.ylabel('cross correlation')
    plt.figure()
    plt.xcorr(MM.volume, MM.volatility, usevlines=False, linestyle='-')
    plt.xlabel('Volume(t) and Volatility(t+j)')
    plt.ylabel('cross correlation')
    plt.xticks(range(-10, 11, 1))
    plt.grid(True)

    # c = tsast.ccf(MM.volume, MM.return_t)
    # plt.figure()
    # plt.plot(c)
    # plt.xlabel('volume(t) and returns(t+j)')
    # plt.ylabel('cross correlation')
    plt.figure()
    plt.xcorr(MM.volume, MM.return_t, usevlines=False, linestyle='-')
    plt.xlabel('Volume(t) and Returns(t+j)')
    plt.ylabel('cross correlation')
    plt.xticks(range(-10, 11, 1))
    plt.grid(True)

    with open('raw_returns.txt') as g:
        s_p_raw = g.readlines()
    s_p_raw_returns = [100 * float(x) for x in s_p_raw]

    with open('abs_returns.txt') as g:
        s_p_abs = g.readlines()
    s_p_abs_returns = [100 * float(x) for x in s_p_abs]

    s_p_sqr_returns = [100 * (float(x) ** 2) for x in s_p_raw]
    # e = tsast.ccf(squared_returns, MM.return_t)
    # plt.figure()
    # plt.plot(e)
    # plt.xlabel('squared returns(t) and returns(t+j)')
    # plt.ylabel('cross correlation')

    # d = tsast.ccf(MM.volatility, MM.return_t)
    # plt.figure()
    # plt.plot(d)
    # plt.xlabel('volatility and return')
    # plt.ylabel('cross correlation')
    plt.figure()
    plt.xcorr(MM.volatility, MM.return_t, usevlines=False, linestyle='-', label='simulated data')
    plt.xcorr(s_p_abs_returns, s_p_raw_returns, usevlines=False, linestyle='-', label='S&P data')
    plt.legend()
    plt.xlabel('Volatility(t) and Return(t+j)')
    plt.ylabel('cross correlation')
    plt.xticks(range(-10, 11, 1))
    plt.grid(True)

    plt.show()


""" Aggregational Gaussainity - increase time over which returns are calculated their dist looks more and more
    like a normal distribution.
    In particular, the shape of the dist is not the same at different time scales
"""


def aggregational_gauss():
    kurt1 = []
    kurt10 = []
    kurt25 = []
    kurt50 = []
    kurt100 = []
    skew1 = []
    skew10 = []
    skew25 = []
    skew50 = []
    skew100 = []

    for j in range(100):
        price10 = []
        returns10 = []
        price25 = []
        returns25 = []
        price50 = []
        returns50 = []
        price100 = []
        returns100 = []

        MM = MarketMaker(1, 1, 0.5, 0.5)
        for i in range(5999):
            MM.update_price()

        returns1 = MM.return_t

        for i in range(0, len(MM.price_t), 10):
            price10.append(MM.price_t[i])
        for i in range(0, len(MM.price_t), 25):
            price25.append(MM.price_t[i])
        for i in range(0, len(MM.price_t), 50):
            price50.append(MM.price_t[i])
        for i in range(0, len(MM.price_t), 100):
            price100.append(MM.price_t[i])

        for i in range(1, len(price10)):
            returns10.append(100 * (price10[i] - price10[i - 1]))
        for i in range(1, len(price25)):
            returns25.append(100 * (price25[i] - price25[i - 1]))
        for i in range(1, len(price50)):
            returns50.append(100 * (price50[i] - price50[i - 1]))
        for i in range(1, len(price100)):
            returns100.append(100 * (price100[i] - price100[i - 1]))

        kurt1.append(sts.kurtosis(returns1))
        skew1.append(sts.skew(returns1))
        kurt10.append(sts.kurtosis(returns10))
        skew10.append(sts.skew(returns10))
        kurt25.append(sts.kurtosis(returns25))
        skew25.append(sts.skew(returns25))
        kurt50.append(sts.kurtosis(returns50))
        skew50.append(sts.skew(returns50))
        kurt100.append(sts.kurtosis(returns100))
        skew100.append(sts.skew(returns100))

    n1, bins1, patches1 = plt.hist(returns1, 50, normed=1, alpha=0.75, label='time lag = 1')
    n10, bins110, patches10 = plt.hist(returns10, 50, normed=1, alpha=0.75, label='time lag = 10')
    n25, bins25, patches25 = plt.hist(returns25, 50, normed=1, alpha=0.75, label='time lag = 25')
    n50, bins50, patches50 = plt.hist(returns50, 50, normed=1, alpha=0.75, label='time lag = 50')
    plt.title('Returns Histogram')
    plt.xlabel('Smarts')
    plt.ylabel('Probalility')
    plt.legend()
    plt.show()

    print('Kurtosis 1: ', float("{0:.4f}".format(numpy.median(kurt1))))
    print('Skewness 1: ', float("{0:.4f}".format(numpy.median(skew1))))
    print('Kurtosis 10: ', float("{0:.4f}".format(numpy.median(kurt10))))
    print('Skewness 10: ', float("{0:.4f}".format(numpy.median(skew10))))
    print('Kurtosis 25: ', float("{0:.4f}".format(numpy.median(kurt25))))
    print('Skewness 25: ', float("{0:.4f}".format(numpy.median(skew25))))
    print('Kurtosis 50: ', float("{0:.4f}".format(numpy.median(kurt50))))
    print('Skewness 50: ', float("{0:.4f}".format(numpy.median(skew50))))
    print('Kurtosis 100: ', float("{0:.4f}".format(numpy.median(kurt100))))
    print('Skewness 100: ', float("{0:.4f}".format(numpy.median(skew100))))


"""Augmented Dickey-Fuller Test for stationarity
"""


def adf():
    MM = MarketMaker(1, 1, 0.5, 0.5)
    for j in range(6000):
        MM.update_price()
    s01 = 0
    s005 = 0
    p01 = 0
    p005 = 0
    n01 = 0
    n005 = 0
    for i in range(101):
        print('iteration ', i)
        MM = MarketMaker(1, 1, 0.5, 0.5)
        for j in range(6000):
            MM.update_price()
        t = tsast.adfuller(MM.price_t)
        if t[0] < t[4]['10%']:
            s01 += 1
        if t[0] < t[4]['5%']:
            s005 += 1

        if t[1] < 0.1:
            p01 += 1
        if t[1] < 0.05:
            p005 += 1

        if t[1] < 0.1 and t[0] < t[4]['10%']:
            n01 += 1
        if t[1] < 0.05 and t[0] < t[4]['5%']:
            n005 += 1

    print('p-val<0.1: ', p01)
    print('p-val<0.05: ', p005)
    print('adf < crit 10%: ', s01)
    print('adf < crit 5%: ', s005)
    print('p-val<0.1 and adf < crit 10%: ', n01)
    print('p-val<0.1 and adf < crit 10%: ', n005)
    print(100 * (p01 + s01 - n01) / 101, '% iteratons are stationary (90%)')
    print(100 * (p005 + s005 - n005) / 101, '% iteratons are stationary (95%)')


#
# prices=[]
# prices.append(1)
# mu=0.005
# sigma=0.015
# for i in range(6000):
# prices.append(generate_next_gbm(prices[-1],mu,sigma))
#
# plt.figure()
# prices=numpy.array(prices)
# plt.plot(prices-1)
# plt.show()
#

# MM=MarketMaker(1,1,0.5,0.5)
# for i in range(6000):
#     MM.update_price()
# square_ret=[x**2 for x in MM.return_t]
#
#
# plt.figure()
# plt.plot(MM.volume)
#
# Vi=numpy.mean(MM.volume)
# volume=[x-Vi for x in MM.volume]
# volume=numpy.sort(volume)
# plt.figure()
# n, bins, patches = plt.hist(volume, 100, normed=1, facecolor='green', alpha=0.75)
# plt.show()
#
# abs_returns = [abs(x) for x in MM.return_t]
# f = open('return.txt', 'w')
# for item in square_ret:
#     f.write("%s\n" % float("{0:.4f}".format(item)))
# f.close()
#
# g = open('price.txt', 'w')
# for item in MM.volume:
#     g.write("%s\n" % float("{0:.4f}".format(item)))
# g.close()
# print(sts.kurtosis(MM.return_t))
# print(sts.kurtosis(abs_returns))
#
# price_impact = numpy.sort(MM.price_t[2:])
# volume = numpy.sort(MM.volume)
#
#
# plt.figure()
# plt.plot(MM.price_t)
#
# plt.figure()
# plt.plot(MM.x_t)
# plt.show()

#
# plt.figure()
# plt.plot(MM.fund.ut,label='fund ut')
# plt.plot(MM.chart.ut,label='chart ut')
# plt.axhline(0, 0, 1, color='black', ls='dotted', lw=1)
# plt.legend()
# plt.show()
# plt.figure()
# plt.plot(MM.price_t[0:500],label= 'price')
# plt.plot(MM.simple_MA.MA[0:500],label='SMA')
# plt.plot(MM.exp_MA.EMA[0:500],label='EMA')
# plt.legend()
# plt.show()


# hurst_abs=[]
# hurst_raw=[]
# for j in range(5):
# print (j)
# MM = MarketMaker(0, 0, 0.5, 0.5)
# for i in range(8500-1):
# MM.update_price()
#     abs_returns = [abs(x) for x in MM.return_t]
#     hurst_abs.append(hurst(abs_returns))
#     hurst_raw.append(hurst(MM.return_t))
# print(numpy.median(hurst_abs))
# print(numpy.median(hurst_raw))
#
# with open('abs_returns.txt') as g:
#     s_p_abs = g.readlines()
# s_p_abs_returns = [float(x) for x in s_p_abs]
# with open('raw_returns.txt') as f:
#     s_p_raw = f.readlines()
# s_p_raw_returns = [float(x) for x in s_p_raw]
# print(hurst(s_p_abs_returns))
# print(hurst(s_p_raw_returns))


# def hill(pt):
# """Returns the Hill Tail index of the price series vector ts"""
#     a = sorted(pt)
#     n = len(a) - 1
#     h = []
#     for k in range(1, 500):
#         s = 0
#         for j in range(2, k):
#             s = s + (numpy.log(a[n - j + 1]) - numpy.log(a[n - k]))
#         h.append(s / k)
#     return h
#
# t0 = time.time()
# dist_compare()
# t1 = time.time()
# total = t1 - t0
# print('time: ', total)
#
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

