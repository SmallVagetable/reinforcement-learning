# value iteration
import numpy as np
from contextlib import contextmanager
import time

from snake import SnakeEnv, TableAgent, eval_game
from policy_iter import PolicyIteration


@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print('{} COST:{}'.format(name, end - start))


class PolicyIterationWithTimer(PolicyIteration):
    def policy_iteration(self, agent, max_iter=-1):
        iteration = 0
        while True:
            iteration += 1
            with timer('Timer PolicyEval'):
                self.policy_evaluation(agent, max_iter)
            with timer('Timer PolicyImprove'):
                ret = self.policy_improvement(agent)
            if not ret:
                break


def policy_iteration_demo():
    np.random.seed(0)
    env = SnakeEnv(10, [3, 6])
    agent = TableAgent(env)
    pi_algo = PolicyIterationWithTimer()
    pi_algo.policy_iteration(agent)
    print('return_pi={}'.format(eval_game(env, agent)))
    print(agent.pi)


def value_iteration_demo():
    np.random.seed(0)
    env = SnakeEnv(10, [3, 6])
    agent = TableAgent(env)
    vi_algo = ValueIteration()
    vi_algo.value_iteration(agent)
    print('return_pi={}'.format(eval_game(env, agent)))
    print(agent.pi)


class ValueIteration(object):
    def value_iteration(self, agent, max_iter=-1):
        iteration = 0
        while True:
            iteration += 1
            new_value_pi = np.zeros_like(agent.value_pi)
            for i in range(1, agent.s_len):  # for each state
                value_sas = []
                for j in range(0, agent.a_len):  # for each act
                    value_sa = np.dot(agent.p[j, i, :], agent.r + agent.gamma * agent.value_pi)
                    value_sas.append(value_sa)
                new_value_pi[i] = max(value_sas)
            diff = np.sqrt(np.sum(np.power(agent.value_pi - new_value_pi, 2)))
            if diff < 1e-6:
                break
            else:
                agent.value_pi = new_value_pi
            if iteration == max_iter:
                break
        print('Iter {} rounds converge'.format(iteration))
        for i in range(1, agent.s_len):
            for j in range(0, agent.a_len):
                agent.value_q[i, j] = np.dot(agent.p[j, i, :], agent.r + agent.gamma * agent.value_pi)
            max_act = np.argmax(agent.value_q[i, :])
            agent.pi[i] = max_act


if __name__ == '__main__':
    policy_iteration_demo()
    value_iteration_demo()
