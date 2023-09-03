import math
from scipy import stats


# European Vanilla
class BSM:

    def __init__(self, r, s, k, t, vol):
        self.r = r
        self.s = s
        self.k = k
        self.t = t / 365
        self.vol = vol  # Use a private variable with underscore
        self.vol_squared = self.vol ** 2

    def d1(self):
        return (math.log(self.s / self.k) + (self.r + (self.vol_squared / 2)) * self.t) / (self.vol * math.sqrt(self.t))

    def d2(self):
        return self.d1() - (self.vol * math.sqrt(self.t))

    def nd1(self, pos=True):
        x = self.d1() if pos else -self.d1()

        return stats.norm.cdf(x, loc=0, scale=1)

    def nd2(self, pos=True):
        x = self.d2() if pos else -self.d2()

        return stats.norm.cdf(x, loc=0, scale=1)

    def call_price(self):
        return (self.s * self.nd1()) - (self.k * math.exp(-self.r * self.t) * self.nd2())

    def put_price(self):
        return (self.k * math.exp(-self.r * self.t) * self.nd2(pos=False)) - (self.s * self.nd1(pos=False))

    def delta(self):
        delta_call = self.nd1()
        delta_put = -self.nd1(pos=False)

        return delta_call, delta_put

    def gamma(self):
        return stats.norm.pdf(self.d1(), 0, 1) / (self.s * self.vol * math.sqrt(self.t))

    def vega(self):
        return self.s * stats.norm.pdf(self.d1(), 0, 1) * math.sqrt(self.t) * 0.01

    def theta(self):
        first_term = (-self.s * stats.norm.pdf(self.d1(), 0, 1) * self.vol) / (2 * math.sqrt(self.t))

        call_second_term = self.r * self.k * math.exp(-self.r * self.t) * stats.norm.cdf(self.d2(), 0, 1)
        theta_call = (first_term - call_second_term) / 365

        put_second_term = self.r * self.k * math.exp(-self.r * self.t) * stats.norm.cdf(-self.d2(), 0, 1)

        theta_put = (first_term + put_second_term) / 365

        return theta_call, theta_put

    def rho(self):
        rho_call = self.t * self.k * math.exp(-self.r * self.t) * stats.norm.cdf(self.d2()) * 0.1
        rho_put = -self.t * self.k * math.exp(-self.r * self.t) * stats.norm.cdf(-self.d2(), 0, 1) * .01

        return rho_call, rho_put


if __name__ == '__main__':
    option = BSM(r=0.01, s=30, k=40, t=240, vol=0.3)

    print(f'Call price: {option.call_price()}')
    print(f'Put price: {option.put_price()}')
