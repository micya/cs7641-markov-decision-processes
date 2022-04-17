import gym
import numpy as np


class MountainCarAgent(object):
    def __init__(self, gamma: float, epsilon: float, alpha: float, episodes: int, position_partitions: int, velocity_partitions: int) -> None:
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.episodes = episodes

        self.position_partitions = position_partitions
        self.velocity_partitions = velocity_partitions

        # -1.2 <= position <= 0.6 & -0.07 <= velocity <= 0.07
        self.min_position = -1.2
        self.max_position = 0.6
        self.min_velocity = -0.07
        self.max_velocity = 0.07

        self.position_size = (self.max_position -
                              self.min_position) / position_partitions
        self.velocity_size = (self.max_velocity -
                              self.min_velocity) / velocity_partitions

        self.num_actions = 3
        self.goal = 0.5
        self.discrete_goal = int(
            np.floor((self.goal - self.min_position) / self.position_size))

    def train(self) -> np.array:
        env = gym.make('MountainCar-v0')

        q_table = np.zeros([self.position_partitions + 1, self.velocity_partitions + 1, self.num_actions])

        for _ in range(self.episodes):
            state = env.reset()
            position, velocity = self.get_discrete_state(state[0], state[1])
            done = False

            while not done:
                # choose A from S using policy derived from Q
                action = self.get_greedy_action(
                    position, velocity, q_table, env.action_space.n)

                # take action A, observe R, S'
                new_state, reward, done, info = env.step(action)
                new_position, new_velocity = self.get_discrete_state(new_state[0], new_state[1])

                # do scary update
                q_table[position, velocity, action] += self.alpha * \
                    (reward + self.gamma *
                     np.max(q_table[new_position, new_velocity]) - q_table[position, velocity, action])

                # update state and action
                position = new_position
                velocity = new_velocity

        policy = np.zeros([self.position_partitions + 1, self.velocity_partitions + 1])

        for position in range(self.position_partitions):
            for velocity in range(self.velocity_partitions):
                policy[position, velocity] = np.argmax(q_table[position, velocity])

        env.close()

        return policy

    def get_discrete_state(self, position: float, velocity: float) -> tuple:
        discrete_position = int(
            np.floor((position - self.min_position) / self.position_size))
        discrete_velocity = int(
            np.floor((velocity - self.min_velocity) / self.velocity_size))
        return discrete_position, discrete_velocity

    def get_greedy_action(self, position, velocity, q_table, size_of_action_space):
        # decide if random
        if np.random.random() < self.epsilon:
            # pick random action
            action = np.random.randint(size_of_action_space)
        else:
            # pick greedy action
            action = np.argmax(q_table[position, velocity])

        return action

    def test(self, policy=None):
        if policy is None:
            policy = self.train()
            print(policy)

        env = gym.make('MountainCar-v0')

        state = env.reset()
        done = False

        while not done:
            env.render()

            action = int(policy[self.get_discrete_state(state[0], state[1])])
            # print(action)

            new_state, reward, done, info = env.step(action)

            # update state
            state = new_state

        env.close()


if __name__ == "__main__":
    agent = MountainCarAgent(0.9, 0.5, 0.1, 500, 20, 20)
    agent.test()

    # discrete_position, discrete_velocity = agent.get_discrete_state(-1.0, 0)
    # print(agent.get_next_state(discrete_position, discrete_velocity, 0))
    # print(agent.get_next_state(discrete_position, discrete_velocity, 1))
    # print(agent.get_next_state(discrete_position, discrete_velocity, 2))
