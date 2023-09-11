import numpy as np
import matplotlib.pyplot as plt


class PriceSimulators:

    def __init__(self, T, N, S0):
        """
        Initialize the PriceSimulators class with the given parameters.

        :param T: Total time horizon for simulation.
        :param N: Number of time steps or intervals.
        :param S0: Initial price or value of the asset.
        """

        self.T = T
        self.N = N
        self.S0 = S0

    def arithmetic_bm(self, mu, sigma, plot=False):
        """
        Simulate Arithmetic Brownian Motion (ABM).

        :param mu: Drift (mean return) of the Brownian Motion.
        :param sigma: Volatility (standard deviation of return) of the Brownian Motion.
        :param plot: If True, plot the ABM simulation.
        :return: Arrays t_abm and S_abm representing time and the corresponding ABM values.
        """
        dt = self.T / self.N
        t_abm = np.linspace(0.0, self.T, self.N + 1)
        W_abm = np.random.normal(0.0, 1.0, self.N) * np.sqrt(dt)
        W_abm = np.insert(W_abm, 0, 0.0)
        S_abm = self.S0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * W_abm))
        if plot:
            self.__plot(t_abm, S_abm, "Arithmetic Brownian Motion (ABM)")
        return t_abm, S_abm

    def geometric_bm(self, mu, sigma, plot=False):
        """
        Simulate Geometric Brownian Motion (GBM).

        :param mu: Drift (mean return) of the GBM.
        :param sigma: Volatility (standard deviation of return) of the GBM.
        :param plot: If True, plot the GBM simulation.
        :return: Arrays t_gbm and S_gbm representing time and the corresponding GBM values.
        """
        dt = self.T / self.N
        t_gbm = np.linspace(0.0, self.T, self.N)
        W_gbm = np.cumsum(np.random.normal(0.0, 1.0, self.N) * np.sqrt(dt))
        S_gbm = self.S0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * W_gbm))
        if plot:
            self.__plot(t_gbm, S_gbm, "Geometric Brownian Motion (GBM)")
        return t_gbm, S_gbm

    def mean_reverting(self, kappa, theta, sigma, plot=False):
        """
        Simulate a Mean-Reverting Model (Ornstein-Uhlenbeck process).

        :param kappa: Mean-reversion speed.
        :param theta: Long-term mean.
        :param sigma: Volatility.
        :param plot: If True, plot the simulation.
        :return: Arrays t_mr and S_mr representing time and the corresponding Mean-Reverting Model values.
        """
        dt = self.T / self.N
        t_mr = np.linspace(0.0, self.T, self.N + 1)
        dW = np.random.normal(0.0, np.sqrt(dt), self.N)
        S_mr = np.zeros(self.N + 1)
        S_mr[0] = self.S0

        for i in range(1, self.N + 1):
            S_mr[i] = S_mr[i - 1] + kappa * (theta - S_mr[i - 1]) * dt + sigma * dW[i - 1]

        if plot:
            self.__plot(t_mr, S_mr, "Mean-Reverting Model")
        return t_mr, S_mr

    def jump_diffusion(self, mu, sigma_jump, jump_intensity, plot=False):
        """
        Simulate a Jump Diffusion Model.

        :param mu: Drift (mean return).
        :param sigma_jump: Jump size volatility.
        :param jump_intensity: Jump intensity (average number of jumps per unit time).
        :param plot: If True, plot the simulation.
        :return: Arrays t_jump and S_jump representing time and the corresponding Jump Diffusion Model values.
        """
        dt = self.T / self.N
        t_jump = np.linspace(0.0, self.T, self.N + 1)
        N_jumps = np.random.poisson(jump_intensity * self.T)
        jump_times = np.sort(np.random.uniform(0, self.T, N_jumps))
        jump_sizes = np.random.normal(0, sigma_jump, N_jumps)
        S_jump = np.zeros(self.N + 1)
        S_jump[0] = self.S0
        jump_index = 0

        for i in range(1, self.N + 1):
            if jump_index < N_jumps and t_jump[i] >= jump_times[jump_index]:
                S_jump[i] = S_jump[i - 1] + mu * dt + jump_sizes[jump_index]
                jump_index += 1
            else:
                S_jump[i] = S_jump[i - 1] + mu * dt

        if plot:
            self.__plot(t_jump, S_jump, "Jump Diffusion Model")
        return t_jump, S_jump

    @staticmethod
    def __plot(t, S, title):
        """
        Internal method to plot a given time series.

        :param t: Array of time values.
        :param S: Array of corresponding values.
        :param title: Title for the plot.
        :return: The created plot.
        """
        plt.figure(figsize=(6, 6))
        plt.plot(t, S)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.tight_layout()
        plt.show()
        return plt


if __name__ == '__main__':
    T = 1.0
    N = 1000
    S0 = 100.0
    mu = 0.1
    sigma = 0.2
    kappa = 0.1
    theta = 100.0
    sigma_jump = 0.5
    jump_intensity = 0.1

    simulator = PriceSimulators(T, N, S0)

    t_abm, S_abm = simulator.arithmetic_bm(mu, sigma, plot=True)

    t_gbm, S_gbm = simulator.geometric_bm(mu, sigma, plot=True)

    t_mr, S_mr = simulator.mean_reverting(kappa, theta, sigma, plot=True)

    t_jump, S_jump = simulator.jump_diffusion(mu, sigma_jump, jump_intensity, plot=True)
