import gym
import numpy as np
import matplotlib.pyplot as plt
import time


class FrozenLakeAgent(object):
    def __init__(self, gamma: float, theta: float, m: int, n: int, map: str) -> None:
        self.gamma = gamma
        self.theta = theta

        self.m = m
        self.n = n
        self.num_states = m * n
        self.num_actions = 4

        self.map = map
        self.goal = map.index('G')

    def amap_to_gym(self, amap='FFGG'):
        amap = np.asarray(amap, dtype='c')
        side = int(np.sqrt(amap.shape[0]))
        amap = amap.reshape((side, side))
        return amap

    def train(self) -> np.array:
        value_table = np.zeros(self.num_states)
        delta = np.inf
        iterations = 0

        # precalculate probability transitions
        probability_table = np.zeros(
            [self.num_states, self.num_actions, self.num_states])
        move_probability = 1 / 3
        slip_probability = (1 - move_probability) / 2

        for state in range(self.num_states):
            if self.map[state] == 'H' or self.map[state] == 'G':
                continue

            for action in range(self.num_actions):
                # find new state
                state1, valid1 = self.get_neighbor(state, action)
                state2, valid2 = self.get_neighbor(state, (action + 1) % 4)
                state3, valid3 = self.get_neighbor(state, (action + 3) % 4)

                if valid1:
                    probability_table[state][action][state1] = move_probability
                else:
                    # remain in place
                    probability_table[state][action][state] += move_probability

                if valid2:
                    probability_table[state][action][state2] = slip_probability
                else:
                    # remain in place
                    probability_table[state][action][state] += slip_probability

                if valid3:
                    probability_table[state][action][state3] = slip_probability
                else:
                    # remain in place
                    probability_table[state][action][state] += slip_probability

        # precalculate rewards
        reward_table = np.zeros(self.num_states)
        reward_table[self.goal] = 1

        while delta > self.theta:
            delta = 0

            for state in range(self.num_states):
                old_value = value_table[state]
                new_v = np.zeros(self.num_actions)

                for action in range(self.num_actions):
                    for new_state in range(self.num_states):
                        new_v[action] += probability_table[state, action, new_state] * \
                            (reward_table[new_state] +
                             self.gamma * value_table[new_state])

                value_table[state] = np.max(new_v)
                delta = max(delta, abs(old_value - value_table[state]))

            iterations += 1

        print(value_table.reshape((self.m, self.n)))

        print(f"iterations: {iterations}")

        # find best policy and return
        policy = np.zeros(self.num_states)
        for state in range(self.num_states):
            new_v = np.zeros(self.num_actions)

            for action in range(self.num_actions):
                for new_state in range(self.num_states):
                    new_v[action] += probability_table[state, action, new_state] * \
                        (reward_table[new_state] +
                         self.gamma * value_table[new_state])

            policy[state] = np.argmax(new_v)

        return iterations, policy

    def test(self, policy=None):
        if policy is None:
            _, policy = self.train()
            print(policy)

        env = gym.make('FrozenLake-v1',
                       desc=self.amap_to_gym(self.map), is_slippery=True)

        state = env.reset()
        done = False

        while not done:
            env.render()

            action = int(policy[state])
            print(action)

            new_state, reward, done, info = env.step(action)

            # update state
            state = new_state

        env.close()

    def average_reward(self, policy, iterations):
        env = gym.make('FrozenLake-v1',
                       desc=self.amap_to_gym(self.map), is_slippery=True)

        total_rewards = 0

        for _ in range(iterations):
            state = env.reset()
            done = False

            while not done:
                action = int(policy[state])

                new_state, reward, done, info = env.step(action)
                total_rewards += reward

                # update state
                state = new_state

        env.close()
        return total_rewards / iterations

    def state_to_coordinates(self, state: int) -> tuple:
        x = int(np.floor(state / self.m))
        y = state - x * self.m
        return (x, y)

    def coordinate_to_state(self, m: int, n: int) -> int:
        return m * self.n + n

    def get_neighbor(self, state: int, dir: int) -> tuple:
        x, y = self.state_to_coordinates(state)

        if dir == 0:
            return self.coordinate_to_state(x, y - 1), y - 1 >= 0
        elif dir == 1:
            return self.coordinate_to_state(x + 1, y), x + 1 < self.m
        elif dir == 2:
            return self.coordinate_to_state(x, y + 1), y + 1 < self.n
        else:
            return self.coordinate_to_state(x - 1, y), x - 1 >= 0


if __name__ == "__main__":
    # agent = FrozenLakeAgent(0.9, 0.0000001, 4, 4, 'SFFFHFFFFFFFFFFG')
    # agent = FrozenLakeAgent(0.9, 0.0000001, 5, 5, 'SFFFFHFFFFFFFFFFFFFFFFFFG')
    # agent = FrozenLakeAgent(0.9, 0.0000001, 2, 2, 'SFFG')
    # agent = FrozenLakeAgent(0.9, 0.0000001, 4, 4, 'SFFHHFFHHFFHHFFG')
    # agent = FrozenLakeAgent(0.9, 0.0000001, 5, 5, 'SFFFFHFFFHHFFFFFFFFHHFFFG')
    agent = FrozenLakeAgent(0.9, 0.0000001, 10, 10, 'SHFFFHFHFFFFFFFFFHFFFFFFFHFHFFFHFFFHFHFFFHFFFHFHFFFHFFFFFFFFFFFFFFFFFFFFFFFHFHFFFHFFFHFHFFFHFFFHFHFG')

    # agent.test()
    # policy = agent.train()
    # print(agent.average_reward(policy, 100))

    rewards = []
    iterations = []
    times = []
    step_size = 0.05

    for gamma in np.arange(1, step=step_size):
        agent = FrozenLakeAgent(gamma, 0.0000001, 10, 10, 'SHFFFHFHFFFFFFFFFHFFFFFFFHFHFFFHFFFHFHFFFHFFFHFHFFFHFFFFFFFFFFFFFFFFFFFFFFFHFHFFFHFFFHFHFFFHFFFHFHFG')

        start_time = time.time()
        iteration, policy = agent.train()
        run_time = time.time() - start_time

        iterations.append(iteration)
        rewards.append(agent.average_reward(policy, 100))
        times.append(run_time)

    plt.scatter(np.arange(1, step=step_size), iterations)
    plt.savefig("ValueIteration-Iterations.png")
    plt.close()

    plt.scatter(np.arange(1, step=step_size), rewards)
    plt.savefig("ValueIteration-Rewards.png")
    plt.close()

    plt.scatter(np.arange(1, step=step_size), times)
    plt.savefig("ValueIteration-Times.png")
    plt.close()
