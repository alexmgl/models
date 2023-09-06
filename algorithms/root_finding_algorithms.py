import sympy as sp


def newton_solver(expression, x0, epsilon=1e-6, max_iterations=100, variable='x'):
    try:
        expr = sp.sympify(expression)

    except Exception as e:
        raise ValueError(f"Error: {str(e)}")

    x = sp.symbols(variable)  # differentiate function with respect to { arg }
    derivative = sp.diff(expr, x)

    for i in range(max_iterations):

        x1 = x0 - expr.subs(x, x0) / derivative.subs(x, x0)

        if abs(x1 - x0) < epsilon:
            return float(x1)  # Converged to a root
        x0 = x1

    raise ValueError("Newton-Raphson method did not converge")


def secant_solver(expression, x0, x1, epsilon=1e-6, max_iterations=100, variable='x', override=False):
    try:
        expr = sp.sympify(expression)
        x = sp.symbols(variable)  # differentiate function with respect to { arg }

        if expr.subs(x, x0) * expr.subs(x, x1) >= 0 and not override:
            raise ValueError(
                "Initial values x0 and x1 should have different signs for f(x). Update x0 or x1, or set override arg equal to True (not recommended).")

    except Exception as e:
        raise ValueError(f"Error: {str(e)}")

    for i in range(max_iterations):

        f_x0 = expr.subs(x, x0)
        f_x1 = expr.subs(x, x1)

        if abs(f_x1) < epsilon:
            return float(x1)

        x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)

        x0 = x1
        x1 = x_new

    raise ValueError("Secant method did not converge")


def bisection_solver(expression, x0, x1, epsilon=1e-6, max_iterations=100, variable='x', override=False):
    try:
        expr = sp.sympify(expression)
        x = sp.symbols(variable)  # differentiate function with respect to { arg }

        if expr.subs(x, x0) * expr.subs(x, x1) >= 0 and not override:
            raise ValueError(
                "Initial values x0 and x1 should have different signs for f(x). Update x0 or x1, or set override arg equal to True (not recommended).")

    except Exception as e:
        raise ValueError(f"Error: {str(e)}")

    for i in range(max_iterations):

        mid = (x0 + x1) / 2
        f_mid = expr.subs(x, mid)

        if abs(f_mid) < epsilon:
            return float(mid)

        if f_mid < 0:
            x0 = mid
        else:
            x1 = mid

    raise ValueError("Secant method did not converge")


if __name__ == '__main__':

    # Example
    expression_string = "x**2 - 4*x - 7"

    solution = newton_solver(expression=expression_string, x0=5)
    print(solution)

    solution = secant_solver(expression=expression_string, x0=5, x1=10)
    print(solution)

    solution = bisection_solver(expression=expression_string, x0=5, x1=10)
    print(solution)
