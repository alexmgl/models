from abc import ABC, abstractmethod
from dice_game import DiceGame
import numpy as np
import time


class DiceGameAgent(ABC):

    def __init__(self, game):
        self.game = game

    @abstractmethod
    def play(self, state):
        pass


class MyAgent(DiceGameAgent):
    gamma = 0.95  # discount rate
    epsilon = 0.009  # convergence error bound

    def __init__(self, game):

        super().__init__(game)
        self.final_action = self.game.actions[-1]
        self.delta = 0

    def play(self, state):

        my_score = sum(state)

        while True:

            self.delta = 0

            for action in self.game.actions:
                states, game_over, reward, probabilities = self.game.get_next_states(action=action, dice_state=state)

                if game_over:
                    return self.final_action

                ev = 0

                for next_state, probability in zip(states, probabilities):

                    if next_state[0] is not None:
                        next_state_score = sum(next_state)
                        ev += probability * (reward + self.gamma * next_state_score)

                self.delta = max(self.delta, abs(ev - my_score))
                my_score = ev

            if self.delta < self.epsilon * (1 - self.gamma) / self.gamma:
                break

        print(f'Final value: {my_score}')


def grid_search(sims=1000):

    print(f'Starting grid search.')

    gammas = np.arange(0.9, 1, 0.005)
    epsilons = np.arange(0, 0.02, 0.001)

    print(f'Gamma len: {len(gammas)}, Epsilon len: {len(epsilons)}, ')

    best_value = 0
    best_gamma = None
    best_epsilon = None

    plot_dict = {'x': [], 'y': [], 'z': []}

    for gamma in gammas:
        for epsilon in epsilons:
            score_list = []
            for _ in range(sims):
                np.random.seed()

                game = DiceGame()

                agent1 = MyAgent(game)
                agent1.gamma = gamma
                agent1.epsilon = epsilon

                value = play_game_with_agent(agent1, game, verbose=False)
                score_list.append(value)

            np_score_list = np.array(score_list)
            mean_score = np_score_list.mean()

            plot_dict['x'].append(gamma)
            plot_dict['y'].append(epsilon)
            plot_dict['z'].append(mean_score)

            # Check if the current value is better than the previous best value
            if mean_score > best_value:
                best_value = mean_score
                best_gamma = gamma
                best_epsilon = epsilon

                # Print the best values found
                print(f'Best: Gamma {best_gamma:.3f}, Epsilon {best_epsilon:.3f} - VALUE {best_value:.3f}')

    # Print the best values found
    print("Best Gamma:", best_gamma)
    print("Best Epsilon:", best_epsilon)
    print("Best Value:", best_value)

    x = np.array(plot_dict['x'])
    y = np.array(plot_dict['y'])
    z = np.array(plot_dict['z'])

    import matplotlib.pyplot as plt

    # Create figure and 3D axis
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Plot the 3D scatter plot
    ax.scatter3D(x, y, z, c=z, cmap='viridis')

    ax.set_xlabel('GAMMA')
    ax.set_ylabel('EPSILON')
    ax.set_zlabel('SCORE')

    plt.show()


def play_game_with_agent(agent, game, verbose=True):
    state = game.reset()

    if (verbose): print(f"Testing agent: \n\t{type(agent).__name__}")
    if (verbose): print(f"Starting dice: \n\t{state}\n")

    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)
        actions += 1

        if (verbose): print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        if (verbose and not game_over): print(f"Dice: \t\t{state}")

    if (verbose): print(f"\nFinal dice: {state}, score: {game.score}")

    return game.score


def extended_tests():
    total_score = 0
    total_time = 0
    n = 10

    print("Testing extended rules – two three-sided dice.")
    print()

    game = DiceGame(dice=2, sides=3)

    start_time = time.process_time()
    test_agent = MyAgent(game)
    total_time += time.process_time() - start_time

    for i in range(n):
        start_time = time.process_time()
        score = play_game_with_agent(test_agent, game)
        print(score)
        total_time += time.process_time() - start_time

        print(f"Game {i} score: {score}")
        total_score += score

    print()
    print(f"Average score: {total_score / n}")
    print(f"Average time: {total_time / n:.5f} seconds")


def main():
    # random seed makes the results deterministic
    # change the number to see different results
    #  or delete the line to make it change each time it is run
    # np.random.seed()
    #
    # game = DiceGame()
    #
    # agent1 = MyAgent(game)
    # x = play_game_with_agent(agent1, game, verbose=False)
    # print(x)

    # extended_tests()

    grid_search(sims=100)

    print("\n")

if __name__ == "__main__":
    main()
