import gym
import numpy as np
import matplotlib.pyplot as plt
import time


class MountainCarAgent(object):
    def __init__(self, gamma: float, theta: float, position_partitions: int, velocity_partitions: int) -> None:
        self.gamma = gamma
        self.theta = theta

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
        value_table = np.zeros(
            [self.position_partitions + 1, self.velocity_partitions + 1])
        policy_table = np.zeros(
            [self.position_partitions + 1, self.velocity_partitions + 1], dtype=int)
        # policy_table = np.random.randint(self.num_actions,
        #                                  size=[self.position_partitions + 1, self.velocity_partitions + 1], dtype=int)
        iterations = 0

        # precalculate rewards
        reward_table = np.full(
            [self.position_partitions + 1, self.velocity_partitions + 1], -1)
        reward_table[self.discrete_goal: self.position_partitions + 1] = 0

        print(reward_table)

        policy_stable = False

        while not policy_stable:
            delta = np.inf
            policy_stable = True

            # policy evaluation
            while delta > self.theta:
                delta = 0

                for position in range(self.position_partitions):
                    for velocity in range(self.velocity_partitions):
                        if position <= self.discrete_goal:
                            action = policy_table[position, velocity]

                            # mountain car is deterministic
                            new_state = self.get_next_state(
                                position, velocity, action)
                            new_v = reward_table[new_state] + \
                                self.gamma * value_table[new_state]
                        else:
                            # no update if this state is already goal state
                            new_v = 0

                        delta = max(delta, abs(
                            value_table[position, velocity] - new_v))
                        value_table[position, velocity] = new_v

                iterations += 1

            # policy improvement
            for position in range(self.position_partitions):
                for velocity in range(self.velocity_partitions):
                    old_policy = policy_table[position, velocity]
                    max_value = -np.inf
                    max_action = 0

                    for action in range(self.num_actions):
                        if position <= self.discrete_goal:
                            # mountain car is deterministic
                            new_state = self.get_next_state(
                                position, velocity, action)
                            new_v = reward_table[new_state] + \
                                self.gamma * value_table[new_state]
                        else:
                            # no update if this state is already goal state
                            new_v = 0

                        if new_v > max_value:
                            max_value = new_v
                            max_action = action

                    policy_table[position, velocity] = max_action

                    if policy_table[position, velocity] != old_policy:
                        policy_stable = False

        print(value_table)
        print(f"iterations: {iterations}")

        return iterations, policy_table

    def get_discrete_state(self, position: float, velocity: float) -> tuple:
        discrete_position = int(
            np.floor((position - self.min_position) / self.position_size))
        discrete_velocity = int(
            np.floor((velocity - self.min_velocity) / self.velocity_size))
        return discrete_position, discrete_velocity

    def get_original_state(self, discrete_position: int, discrete_velocity: int) -> tuple:
        position = discrete_position * self.position_size + self.min_position
        velocity = discrete_velocity * self.velocity_size + self.min_velocity
        return position, velocity

    def get_next_state(self, discrete_position: int, discrete_velocity: int, action: int) -> tuple:
        force = 0.001
        gravity = 0.0025

        position, velocity = self.get_original_state(
            discrete_position, discrete_velocity)
        new_velocity = velocity + (action - 1) * \
            force - np.cos(3 * position) * gravity
        new_position = position + new_velocity

        # clip velocity
        new_velocity = np.clip(
            new_velocity, self.min_velocity, self.max_velocity)

        # stop if bump into wall
        if new_position <= self.min_position and velocity < 0:
            new_velocity = 0

        return self.get_discrete_state(new_position, new_velocity)

    def test(self, policy=None):
        if policy is None:
            policy = self.train()
            print(policy)

        env = gym.make('MountainCar-v0')

        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # env.render()

            action = int(policy[self.get_discrete_state(state[0], state[1])])
            # print(action)

            new_state, reward, done, info = env.step(action)
            total_reward += reward

            # update state
            state = new_state

        env.close()
        return total_reward


if __name__ == "__main__":
    # agent = MountainCarAgent(0.9, 0.1, 10, 1000)
    agent = MountainCarAgent(0.9, 0.1, 10, 100)
    # agent.test()

    # policy = agent.train()
    # agent.test(policy)

    # discrete_position, discrete_velocity = agent.get_discrete_state(-1.0, 0)
    # print(agent.get_next_state(discrete_position, discrete_velocity, 0))
    # print(agent.get_next_state(discrete_position, discrete_velocity, 1))
    # print(agent.get_next_state(discrete_position, discrete_velocity, 2))


    rewards = []
    iterations = []
    times = []
    step_size = 10

    # for gamma in np.arange(1, step=step_size):
    for partitions in np.arange(200, step=step_size):
        agent = MountainCarAgent(0.9, 0.0000001, partitions, 100)

        start_time = time.time()
        iteration, policy = agent.train()
        run_time = time.time() - start_time

        iterations.append(iteration)
        rewards.append(agent.test(policy))
        times.append(run_time)

    plt.scatter(np.arange(200, step=step_size), iterations)
    plt.savefig("PolicyIteration-Iterations.png")
    plt.close()

    plt.scatter(np.arange(200, step=step_size), rewards)
    plt.savefig("PolicyIteration-Rewards.png")
    plt.close()

    plt.scatter(np.arange(200, step=step_size), times)
    plt.savefig("PolicyIteration-Times.png")
    plt.close()
