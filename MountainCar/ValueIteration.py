import gym
import numpy as np

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

        self.position_size = (self.max_position - self.min_position) / position_partitions
        self.velocity_size = (self.max_velocity - self.min_velocity) / velocity_partitions

        self.num_actions = 3
        self.goal = 0.5
        self.discrete_goal = int(np.floor((self.goal - self.min_position) / self.position_size))
    
    def train(self) -> np.array:
        value_table = np.zeros([self.position_partitions + 1, self.velocity_partitions + 1])
        delta = np.inf
        iterations = 0

        # precalculate rewards
        reward_table = np.full([self.position_partitions + 1, self.velocity_partitions + 1], -1)
        reward_table[self.discrete_goal : self.position_partitions + 1] = 10

        print(reward_table)

        while delta > self.theta:
            delta = 0

            for position in range(self.position_partitions):
                for velocity in range(self.velocity_partitions):
                    max_value = -np.inf

                    for action in range(self.num_actions):
                        if position <= self.discrete_goal:
                            # mountain car is deterministic
                            new_state = self.get_next_state(position, velocity, action)
                            new_v = reward_table[new_state] + self.gamma * value_table[new_state]
                        else:
                            # no update if this state is already goal state
                            new_v = 0

                        if new_v > max_value:
                            max_value = new_v

                    delta = max(delta, abs(value_table[position, velocity] - max_value))
                    value_table[position, velocity] = max_value
                    
            iterations += 1
            
        print(value_table)
        print(f"iterations: {iterations}")

        # find best policy and return
        policy = np.zeros([self.position_partitions, self.velocity_partitions])
        for position in range(self.position_partitions):
                for velocity in range(self.velocity_partitions):
                    max_value = -np.inf
                    max_action = 0

                    for action in range(self.num_actions):
                        if position <= self.discrete_goal:
                            # mountain car is deterministic
                            new_state = self.get_next_state(position, velocity, action)
                            new_v = reward_table[new_state] + self.gamma * value_table[new_state]
                        else:
                            # no update if this state is already goal state
                            new_v = 0

                        if new_v > max_value:
                            max_value = new_v
                            max_action = action

                    policy[position, velocity] = max_action

        return policy

    def get_discrete_state(self, position: float, velocity: float) -> tuple:
        discrete_position = int(np.floor((position - self.min_position) / self.position_size))
        discrete_velocity = int(np.floor((velocity - self.min_velocity) / self.velocity_size))
        return discrete_position, discrete_velocity

    def get_original_state(self, discrete_position: int, discrete_velocity: int) -> tuple:
        position = discrete_position * self.position_size + self.min_position
        velocity = discrete_velocity * self.velocity_size + self.min_velocity
        return position, velocity

    def get_next_state(self, discrete_position: int, discrete_velocity: int, action: int) -> tuple:
        force = 0.001
        gravity = 0.0025

        position, velocity = self.get_original_state(discrete_position, discrete_velocity)
        new_velocity = velocity + (action - 1) * force - np.cos(3 * position) * gravity
        new_position = position + new_velocity

        # clip velocity
        new_velocity = np.clip(new_velocity, self.min_velocity, self.max_velocity)

        # stop if bump into wall
        if new_position <= self.min_position and velocity < 0:
            new_velocity = 0

        return self.get_discrete_state(new_position, new_velocity)

    def test(self, policy = None):
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
    agent = MountainCarAgent(0.9, 0.1, 10, 1000)
    agent.test()

    # discrete_position, discrete_velocity = agent.get_discrete_state(-1.0, 0)
    # print(agent.get_next_state(discrete_position, discrete_velocity, 0))
    # print(agent.get_next_state(discrete_position, discrete_velocity, 1))
    # print(agent.get_next_state(discrete_position, discrete_velocity, 2))