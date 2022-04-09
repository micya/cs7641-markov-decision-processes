import gym
import numpy as np

class FrozenLakeAgent(object): 
    def __init__(self, gamma: float, theta: float, m: int, n: int, map: str, goal: int) -> None:
        self.gamma = gamma
        self.theta = theta

        self.m = m
        self.n = n
        self.num_states = m * n
        self.num_actions = 4

        self.map = map
        self.goal = goal

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
        probability_table = np.zeros([self.num_states, self.num_actions, self.num_states])
        move_probability = 1 / 3
        slip_probability = (1 - move_probability) / 2

        for state in range(self.num_states):
            if self.map[state] == 'H':
                continue

            for action in range(self.num_actions):
                # find new state
                if action == 0:
                    if state - 1 >= 0:
                        probability_table[state][action][state - 1] = move_probability
                    if state - self.n >= 0:
                        probability_table[state][action][state - self.n] = slip_probability
                    if state + self.n < self.num_states:
                        probability_table[state][action][state + self.n] = slip_probability
                elif action == 1:
                    if state + self.n < self.num_states:
                        probability_table[state][action][state + self.n] = move_probability
                    if state - 1 >= 0:
                        probability_table[state][action][state - 1] = slip_probability
                    if state + 1 < self.num_states:
                        probability_table[state][action][state + 1] = slip_probability
                elif action == 2:
                    if state + 1 < self.num_states:
                        probability_table[state][action][state + 1] = move_probability
                    if state - self.n >= 0:
                        probability_table[state][action][state - self.n] = slip_probability
                    if state + self.n < self.num_states:
                        probability_table[state][action][state + self.n] = slip_probability
                elif action == 3:
                    if state - self.n >= 0:
                        probability_table[state][action][state - self.n] = move_probability
                    if state - 1 >= 0:
                        probability_table[state][action][state - 1] = slip_probability
                    if state + 1 < self.num_states:
                        probability_table[state][action][state + 1] = slip_probability

        # precalculate rewards
        reward_table = np.zeros(self.num_states)
        reward_table[self.goal] = 1

        while delta > self.theta:
            delta = 0

            for state in range(self.num_states):
                tmp = value_table[state]
                new_v = np.zeros(self.num_actions)

                for action in range(self.num_actions):
                    for new_state in range(self.num_states):
                        new_v[action] += probability_table[state, action, new_state] * \
                            (reward_table[new_state] + self.gamma * value_table[new_state])

                value_table[state] = np.max(new_v)
                delta = max(delta, abs(tmp - value_table[state]))

            iterations += 1
            
        print(f"iterations: {iterations}")

        # find best policy and return
        policy = np.zeros(self.num_states)
        for state in range(self.num_states):
                new_v = np.zeros(self.num_actions)

                for action in range(self.num_actions):
                    for new_state in range(self.num_states):
                        new_v[action] += probability_table[state, action, new_state] * \
                            (reward_table[new_state] + self.gamma * value_table[new_state])

                policy[state] = np.argmax(new_v)

        return policy

    def test(self):
        policy = self.train()
        print(policy)

        env = gym.make('FrozenLake-v1', desc=self.amap_to_gym(self.map), is_slippery=True)

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

if __name__ == "__main__":
    # agent = FrozenLakeAgent(0.9, 0.0000001, 4, 4, 'SFFFHFFFFFFFFFFG', 15)
    # agent = FrozenLakeAgent(0.9, 0.0000001, 5, 5, 'SFFFFHFFFFFFFFFFFFFFFFFFG', 24)
    # agent = FrozenLakeAgent(0.9, 0.0000001, 2, 2, 'SFFG', 3)
    agent = FrozenLakeAgent(0.9, 0.0000001, 4, 4, 'SFFHHFFHHFFHHFFG', 15)
    # agent = FrozenLakeAgent(0.9, 0.0000001, 5, 5, 'SFFFFHFFFHHFFFFFFFFHHFFFG', 24)
    
    agent.test()