import gym
import numpy as np
import matplotlib.pyplot as plt
import time


class FrozenLakeAgent(object):
    def __init__(self, gamma: float, epsilon: float, alpha: float, episodes: int, m: int, n: int, map: str) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.episodes = episodes

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

    def train(self, q_table = None) -> np.array:
        env = gym.make('FrozenLake-v1',
                       desc=self.amap_to_gym(self.map), is_slippery=True)

        if q_table is None:
            q_table = np.zeros([env.observation_space.n, env.action_space.n])

        for _ in range(self.episodes):
            state = env.reset()
            done = False

            while not done:
                # choose A from S using policy derived from Q
                action = self.get_greedy_action(
                    state, q_table, env.action_space.n)

                # take action A, observe R, S'
                new_state, reward, done, info = env.step(action)

                if new_state == 'F':
                    reward = -0.1
                elif new_state == 'H':
                    reward = -1

                # do scary update
                q_table[state, action] += self.alpha * \
                    (reward + self.gamma *
                     np.max(q_table[new_state]) - q_table[state, action])

                # update state and action
                state = new_state

        policy = np.zeros(env.observation_space.n)

        for state in range(env.observation_space.n):
            policy[state] = np.argmax(q_table[state])

        env.close()

        return policy

    def get_greedy_action(self, state, q_table, size_of_action_space):
        # decide if random
        if np.random.random() < self.epsilon:
            # pick random action
            action = np.random.randint(size_of_action_space)
        else:
            # pick greedy action
            action = np.argmax(q_table[state])

        return action

    def test(self):
        policy = self.train()
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


if __name__ == "__main__":
    # agent = FrozenLakeAgent(0.9, 0.1, 1, 100, 4, 4, 'SFFFHFFFFFFFFFFG')
    # agent = FrozenLakeAgent(0.9, 0.1, 1, 100, 5, 5, 'SFFFFHFFFFFFFFFFFFFFFFFFG')
    # agent = FrozenLakeAgent(0.9, 0.1, 1, 100, 2, 2, 'SFFG')
    # agent = FrozenLakeAgent(0.9, 0.4, 1, 5000, 4, 4, 'SFFHHFFHHFFHHFFG')
    agent = FrozenLakeAgent(0.9, 0.4, 1, 5000, 5, 5,
                            'SFFFFHFFFHHFFFFFFFFHHFFFG')

    rewards = []
    step_size = 0.05

    for epsilon in np.arange(1, step=step_size):
        agent = FrozenLakeAgent(0.9, epsilon, 0.2, 5000, 5, 5,
                            'SFFFFHFFFHHFFFFFFFFHHFFFG')

        policy = agent.train()
        rewards.append(agent.average_reward(policy, 100))


    plt.scatter(np.arange(1, step=step_size), rewards)
    plt.savefig("QLearning-Rewards.png")
    plt.close()
