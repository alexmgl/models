#include <iostream>
#include <cmath>
#include <functional>

// Newton-Raphson solver
double newton_solver(const std::function<double(double)>& func, const std::function<double(double)>& derivative, double x0, double epsilon = 1e-6, int max_iterations = 100) {
    for (int i = 0; i < max_iterations; ++i) {
        double f_x0 = func(x0);
        double f_prime_x0 = derivative(x0);

        double x1 = x0 - f_x0 / f_prime_x0;

        if (std::abs(x1 - x0) < epsilon) {
            return x1; // Converged to a root
        }

        x0 = x1;
    }

    throw std::runtime_error("Newton-Raphson method did not converge");
}

// Secant solver
double secant_solver(const std::function<double(double)>& func, double x0, double x1, double epsilon = 1e-6, int max_iterations = 100) {
    for (int i = 0; i < max_iterations; ++i) {
        double f_x0 = func(x0);
        double f_x1 = func(x1);

        if (std::abs(f_x1) < epsilon) {
            return x1;
        }

        double x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0);

        x0 = x1;
        x1 = x_new;
    }

    throw std::runtime_error("Secant method did not converge");
}

// Bisection solver
double bisection_solver(const std::function<double(double)>& func, double x0, double x1, double epsilon = 1e-6, int max_iterations = 100) {
    for (int i = 0; i < max_iterations; ++i) {
        double mid = (x0 + x1) / 2;
        double f_mid = func(mid);

        if (std::abs(f_mid) < epsilon) {
            return mid;
        }

        if (f_mid < 0) {
            x0 = mid;
        } else {
            x1 = mid;
        }
    }

    throw std::runtime_error("Bisection method did not converge");
}

int main() {
    // Example
    auto func = [](double x) { return x * x - 4 * x - 7; };
    auto derivative = [](double x) { return 2 * x - 4; };

    double solution = newton_solver(func, derivative, 5);
    std::cout << "Newton-Raphson solution: " << solution << std::endl;

    solution = secant_solver(func, 5, 10);
    std::cout << "Secant solution: " << solution << std::endl;

    solution = bisection_solver(func, 5, 10);
    std::cout << "Bisection solution: " << solution << std::endl;

    return 0;
}
