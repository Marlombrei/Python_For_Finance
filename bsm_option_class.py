from math import log, sqrt, exp
from scipy import stats


class call_option(object):
    ''' Class for European call options in Black-Scholes-Merton Model
    Attributes
    ====================
    S0 : float
        initial stock level
    K : float
        strike price
    T : float
        maturity (in year fractions)
    r : float
        constant risk-free short-term rate
    
    Methods
    ====================
    value : float
        return Present Value of call option
    vega : float
        return Vega of call option
    imp_vol : float
        return Implied Volatility given option quote
    '''
    
    def __init__(self, S0, K, T, r, sigma):
        self.S0 = float(S0)
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def value(self):
        '''Returns the option value
        #stats.norm.cdf --> Cumulative Standard Normal Distribution'''
        d1 = ((log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T)))
        d2 = ((log(self.S0 / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T)))
        value = (self.S0 * stats.norm.cdf(d1, 0.0, 1.0) - self.K * exp(-self.r * self.T) * stats.norm.cdf(d2, 0.0, 1.0))
        return value

    def vega(self):
        '''Returns Vega of option'''
        d1 = ((log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * sqrt(self.T)))
        vega = self.S0 * stats.norm.cdf(d1, 0.0, 1.0) * sqrt(self.T)
        return vega

    def imp_vol(self, C0, sigma_est=0.2, it=100):
        '''Returns implied volatility given option price'''
        option = call_option(self.S0, self.K, self.T, self.r, sigma_est)
        for i in range(it):
            option.sigma -= (option.value() - C0) / option.vega()
            return option.sigma
