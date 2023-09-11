import numpy as np
from scipy import stats


# Base Option class for pricing models
class Option:

    def __init__(self, r, s, k, t, sigma, direction):

        self.s = s  # Current stock price
        self.k = k  # Strike price
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility
        self.t = t / 365  # Time to expiration # todo - this should really be expiration date, and then calculate the time to expiry
        self.direction = self.convert_direction(direction)

        self.position = None  # placeholder for direction long or short
        self.exercise_type = None  # placeholder for exercise type e.g. American, European
        self.V = None  # Represents the option price for calculating IV

        self.transaction_costs = None  # Placeholder for transaction costs

    def __repr__(self):
        return f'Spot: {self.s}\n' \
               f'Strike: {self.k}\n' \
               f'Interest Rate {self.r}\n' \
               f'Time to Delivery: {self.t}\n' \
               f'Historic Volatility: {self.sigma}\n' \
               f'Direction: {self.direction}'

    @property
    def sigma_squared(self):
        return self.sigma ** 2

    @sigma_squared.setter
    def sigma_squared(self, value):
        self.sigma = np.sqrt(value)

    @staticmethod
    def convert_direction(direction):

        direction = direction.lower()
        if 'c' in direction:
            return 'call'
        elif 'p' in direction:
            return 'put'
        else:
            raise ValueError("Direction must contain 'c' for call or 'p' for put")

    def convert_direction_to_int(self):
        direction = self.direction.lower()
        if direction == 'call':
            return 1  # Return 1 for call option
        elif direction == 'put':
            return -1  # Return -1 for put option
        else:
            raise ValueError("Direction must be 'call' or 'put'")


""" Option Pricing Models """


# Black Scholes Merton Model
class BSM:

    def __init__(self, option):
        self.option = option

    def __repr__(self):
        return f'{repr(self.option)}\n' \
               f'Delta: {self.delta()}\n' \
               f'Gamma: {self.gamma()}\n' \
               f'Vega: {self.vega()}\n' \
               f'Rho: {self.rho()}\n' \
               f'Theta: {self.theta()}'

    def d1(self):
        return (np.log(self.option.s / self.option.k) + (
                self.option.r + (self.option.sigma_squared / 2)) * self.option.t) / (
                       self.option.sigma * np.sqrt(self.option.t))

    def d2(self):
        return self.d1() - (self.option.sigma * np.sqrt(self.option.t))

    def nd1(self, pos=True):
        x = self.d1() if pos else -self.d1()

        return stats.norm.cdf(x, loc=0, scale=1)

    def nd2(self, pos=True):
        x = self.d2() if pos else -self.d2()

        return stats.norm.cdf(x, loc=0, scale=1)

    def __call_price(self):
        return (self.option.s * self.nd1()) - (self.option.k * np.exp(-self.option.r * self.option.t) * self.nd2())

    def __put_price(self):
        return (self.option.k * np.exp(-self.option.r * self.option.t) * self.nd2(pos=False)) - (
                self.option.s * self.nd1(pos=False))

    def price(self):

        if self.option.direction == 'call':
            return self.__call_price()

        elif self.option.direction == 'put':
            return self.__put_price()

        else:
            raise ValueError('Invalid option direction provided')

    def __call_delta(self):
        return self.nd1()

    def __put_delta(self):
        return -self.nd1(pos=False)

    def delta(self):

        if self.option.direction == 'call':
            return self.__call_delta()

        elif self.option.direction == 'put':
            return self.__put_delta()

        else:
            raise ValueError('Invalid option direction provided')

    def gamma(self):
        return stats.norm.pdf(self.d1(), 0, 1) / (self.option.s * self.option.sigma * np.sqrt(self.option.t))

    def vega(self):
        return self.option.s * stats.norm.pdf(self.d1(), 0, 1) * np.sqrt(self.option.t) * 0.01

    def __theta_call(self):

        first_term = (-self.option.s * stats.norm.pdf(self.d1(), 0, 1) * self.option.sigma) / (
                2 * np.sqrt(self.option.t))

        call_second_term = self.option.r * self.option.k * np.exp(-self.option.r * self.option.t) * stats.norm.cdf(
            self.d2(), 0, 1)

        return (first_term - call_second_term) / 365

    def __theta_put(self):

        first_term = (-self.option.s * stats.norm.pdf(self.d1(), 0, 1) * self.option.sigma) / (
                2 * np.sqrt(self.option.t))

        put_second_term = self.option.r * self.option.k * np.exp(-self.option.r * self.option.t) * stats.norm.cdf(
            -self.d2(), 0, 1)

        return (first_term + put_second_term) / 365

    def theta(self):

        if self.option.direction == 'call':
            return self.__theta_call()

        elif self.option.direction == 'put':
            return self.__theta_put()

        else:
            raise ValueError('Invalid option direction provided')

    def __rho_call(self):
        return self.option.t * self.option.k * np.exp(-self.option.r * self.option.t) * stats.norm.cdf(
            self.d2()) * 0.1

    def __rho_put(self):
        return -self.option.t * self.option.k * np.exp(-self.option.r * self.option.t) * stats.norm.cdf(-self.d2(), 0,
                                                                                                        1) * .01

    def rho(self):

        if self.option.direction == 'call':
            return self.__rho_call()

        elif self.option.direction == 'put':
            return self.__rho_put()

        else:
            raise ValueError('Invalid option direction provided')

    # Newton method for finding implied volatility 
    def implied_vol(self, max_iterations=100, precision=1.0e-5):  # todo - finish implied volatility code

        if self.option.V is None:
            raise ValueError('No option price provided. Set V equal to market price.')

        # todo - write Newton solver in a different package and use this rather than write it here


# Binomial Options Pricing Model
class BOPM:

    def __init__(self, option):
        self.option = option

    def __repr__(self):
        return f'{repr(self.option)}\n' \
               f'Option Price (BOPM): {self.price():.2f}'

    def __binomial_tree(self, num_steps):
        """
        Calculate option price using the binomial tree model.

        :param num_steps: Number of time steps in the binomial tree
        :return: Option price
        """

        dt = self.option.t / num_steps
        u = np.exp(self.option.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.option.r * dt) - d) / (u - d)

        # Initialize option prices at expiration
        option_prices = np.zeros(num_steps + 1)
        for i in range(num_steps + 1):
            option_prices[i] = max(0, self.option.convert_direction_to_int() * (
                        self.option.s * (u ** (num_steps - i)) * (d ** i) - self.option.k))

        # Calculate option prices at earlier time steps using recursion
        for step in range(num_steps - 1, -1, -1):
            for i in range(step + 1):
                option_prices[i] = (p * option_prices[i] + (1 - p) * option_prices[i + 1]) * np.exp(-self.option.r * dt)

        return option_prices[0]

    def price(self, num_steps=1000):
        """
        Calculate the option price using the binomial tree model with a specified number of time steps.

        :param num_steps: Number of time steps in the binomial tree
        :return: Option price
        """
        return self.__binomial_tree(num_steps)


if __name__ == '__main__':

    option = Option(r=0.01, s=30, k=40, t=240, sigma=0.3, direction='call')

    pricing_models = [BSM, BOPM]
    for l in pricing_models:
        print(l(option).price())
