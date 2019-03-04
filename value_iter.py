# value iteration
import numpy as np
from contextlib import contextmanager
import time

from snake import SnakeEnv, TableAgent, eval_game
from policy_iter import PolicyIteration


# 创建一个上下文管理器
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
        """
        :param obj agent: 智能体
        :param int max_iter: 最大迭代数
        """
        iteration = 0
        while True:
            iteration += 1
            # 保存算出的值函数
            new_value_pi = np.zeros_like(agent.value_pi)
            for i in range(1, agent.s_len):
                value_sas = []
                for j in range(0, agent.a_len):
                    # 对每一个状态s和行动a，计算值函数
                    value_sa = np.dot(agent.p[j, i, :], agent.r + agent.gamma * agent.value_pi)
                    value_sas.append(value_sa)
                # 从每个行动中，选出最好的值函数
                new_value_pi[i] = max(value_sas)

            # 前后2次值函数的变化小于一个阈值，结束
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
                # 计算收敛后值函数的状态-行动值函数
                agent.value_q[i, j] = np.dot(agent.p[j, i, :], agent.r + agent.gamma * agent.value_pi)
            # 取出最大的行动
            max_act = np.argmax(agent.value_q[i, :])
            agent.pi[i] = max_act


if __name__ == '__main__':
    policy_iteration_demo()
    value_iteration_demo()
