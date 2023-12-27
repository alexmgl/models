def solution_31(coin_list, total, current_solution=(), solutions=None):
    if solutions is None:
        solutions = set()
        coin_list.sort(reverse=True)  # Sort only once before the first call.

    if total == 0:
        solutions.add(current_solution)

        return solutions

    if len(current_solution) == 0:
        last_max = max(coin_list)
    else:
        last_max = current_solution[-1]

    for i in coin_list:
        if i <= total and i <= last_max:
            # Recursively solve the reduced problem
            solution_31(coin_list, total - i, current_solution + (i,), solutions)

    return solutions


if __name__ == '__main__':
    
    coins = [200, 100, 50, 20, 10, 5, 2, 1]
    target = 200

    a = solution_31(coins, target)

    print(f'Length: {len(a)}')
