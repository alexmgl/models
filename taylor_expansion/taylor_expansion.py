import sympy as sp
import math


def univariate_taylor_expansion(function, variables, nth_order, point_val):

    solution = None

    for i in range(nth_order + 1):

        f_diff = sp.diff(function, variables, i)
        f_diff_sol = f_diff.subs(variables, 1)

        nth_diff = (((variables - 1) ** i) / math.factorial(i)) * f_diff_sol

        solution = nth_diff if solution is None else solution + nth_diff

    return solution.subs(variables, point_val)


if __name__ == '__main__':

    # Example problem
    x1 = sp.symbols('x')
    function = x1 ** 3 - 2 * sp.ln(x1)

    result = univariate_taylor_expansion(function, variables=x1, nth_order=3, point_val=1.5)
    print(result)
