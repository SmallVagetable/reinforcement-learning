import numpy as np
from snake import SnakeEnv, TableAgent, eval_game
from value_iter import PolicyIterationWithTimer, ValueIteration, timer


# general iteration
class GeneralizedPolicyIteration(object):
    def __init__(self):
        self.pi_algo = PolicyIterationWithTimer()
        self.vi_algo = ValueIteration()

    def generalized_policy_iteration(self, agent):
        self.vi_algo.value_iteration(agent, 10)
        self.pi_algo.policy_iteration(agent, 1)


def policy_iteration_demo():
    np.random.seed(0)
    env = SnakeEnv(10, [3, 6])
    agent = TableAgent(env)
    pi_algo = PolicyIterationWithTimer()
    with timer('Timer PolicyIter'):
        pi_algo.policy_iteration(agent)
    print('return_pi={}'.format(eval_game(env, agent)))


def value_iteration_demo():
    np.random.seed(0)
    env = SnakeEnv(10, [3, 6])
    agent = TableAgent(env)
    pi_algo = ValueIteration()
    with timer('Timer ValueIter'):
        pi_algo.value_iteration(agent)
    print('return_pi={}'.format(eval_game(env, agent)))


def generalized_iteration_demo():
    np.random.seed(0)
    env = SnakeEnv(10, [3, 6])
    agent = TableAgent(env)
    pi_algo = GeneralizedPolicyIteration()
    with timer('Timer GeneralizedIter'):
        pi_algo.generalized_policy_iteration(agent)
    print('return_pi={}'.format(eval_game(env, agent)))


if __name__ == '__main__':
    policy_iteration_demo()
    value_iteration_demo()
    generalized_iteration_demo()
