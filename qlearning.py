import numpy as np

from snake import SnakeEnv, ModelFreeAgent, TableAgent, eval_game
from policy_iter import PolicyIteration
from monte_carlo import MonteCarlo, timer


class QLearning(object):
    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    def policy_improve(self, agent):
        new_policy = np.zeros_like(agent.pi)
        for i in range(1, agent.s_len):
            new_policy[i] = np.argmax(agent.value_q[i, :])
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        else:
            agent.pi = new_policy
            return True

    # q learning
    def q_learning(self, agent, env):
        for i in range(10):
            for j in range(3000):
                self.q_learn_eval(agent, env)
            self.policy_improve(agent)

    def q_learn_eval(self, agent, env):
        state = env.reset()
        prev_state = -1
        prev_act = -1
        while True:
            act = agent.play(state, self.epsilon)
            next_state, reward, terminate, _ = env.step(act)
            if prev_act != -1:
                return_val = reward + agent.gamma * (0 if terminate else np.max(agent.value_q[state, :]))
                agent.value_n[prev_state][prev_act] += 1
                agent.value_q[prev_state][prev_act] += (return_val - \
                                                        agent.value_q[prev_state][prev_act]) / \
                                                       agent.value_n[prev_state][prev_act]

            prev_act = act
            prev_state = state
            state = next_state

            if terminate:
                break


def monte_carlo_demo():
    np.random.seed(101)
    env = SnakeEnv(10, [3, 6])
    agent = ModelFreeAgent(env)
    mc = MonteCarlo(0.5)
    with timer('Timer Monte Carlo Iter'):
        mc.monte_carlo_opt(agent, env)
    print('return_pi={}'.format(eval_game(env, agent)))
    print(agent.pi)

    np.random.seed(101)
    agent2 = TableAgent(env)
    pi_algo = PolicyIteration()
    with timer('Timer PolicyIter'):
        pi_algo.policy_iteration(agent2)
    print('return_pi={}'.format(eval_game(env, agent2)))
    print(agent2.pi)

    np.random.seed(101)
    agent3 = ModelFreeAgent(env)
    mc = QLearning(0.5)
    with timer('Timer Monte Carlo Iter'):
        mc.q_learning(agent3, env)
    print('return_pi={}'.format(eval_game(env, agent3)))
    print(agent3.pi)


if __name__ == '__main__':
    monte_carlo_demo()
