import gym
import numpy as np


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
        policy_table = np.zeros(self.num_states, dtype=int)
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

        policy_stable = False

        while not policy_stable:
            delta = np.inf
            policy_stable = True

            # policy evaluation
            while delta > self.theta:
                delta = 0

                for state in range(self.num_states):
                    new_value = 0

                    for new_state in range(self.num_states):
                        new_value += probability_table[state, policy_table[state], new_state] * \
                            (reward_table[new_state] +
                             self.gamma * value_table[new_state])

                    delta = max(delta, abs(value_table[state] - new_value))
                    value_table[state] = new_value

                iterations += 1

            # policy improvement
            for state in range(self.num_states):
                old_policy = policy_table[state]
                new_v = np.zeros(self.num_actions)

                for action in range(self.num_actions):
                    for new_state in range(self.num_states):
                        new_v[action] += probability_table[state, action, new_state] * \
                            (reward_table[new_state] +
                             self.gamma * value_table[new_state])

                policy_table[state] = np.argmax(new_v)

                if policy_table[state] != old_policy:
                    policy_stable = False

        print(value_table.reshape((self.m, self.n)))
        print(f"iterations: {iterations}")

        return policy_table

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
    agent = FrozenLakeAgent(0.9, 0.0000001, 4, 4, 'SFFHHFFHHFFHHFFG')
    # agent = FrozenLakeAgent(0.9, 0.0000001, 5, 5, 'SFFFFHFFFHHFFFFFFFFHHFFFG')

    agent.test()
