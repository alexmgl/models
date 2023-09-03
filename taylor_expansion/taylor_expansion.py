import sympy as sp
import math


def univariate_taylor_expansion(input_func, var, nth_order, point_val, solution=0):
    f_diff = sp.diff(input_func, var, nth_order)
    f_diff_sol = f_diff.subs(var, 1)

    nth_diff = (((var - 1) ** nth_order) / math.factorial(nth_order)) * f_diff_sol

    solution += nth_diff

    nth_order -= 1

    if nth_order < 0:

        return solution.subs(var, point_val)

    else:

        return univariate_taylor_expansion(input_func, var, nth_order, point_val, solution)


if __name__ == '__main__':
    # Example problem
    x1 = sp.symbols('x')
    my_func = x1 ** 3 - 2 * sp.ln(x1)

    result = univariate_taylor_expansion(input_func=my_func, var=x1, nth_order=3, point_val=1.02)
    print(result)
