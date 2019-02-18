import numpy as np
from contextlib import contextmanager
import time

from snake import SnakeEnv, ModelFreeAgent, TableAgent, eval_game
from policy_iter import PolicyIteration

# 上下文管理器
@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print('{} COST:{}'.format(name, end - start))


class MonteCarlo(object):
    def __init__(self, epsilon=0.0):
        self.epsilon = epsilon

    def monte_carlo_eval(self, agent, env):
        state = env.reset()
        # episode是整个从开始到结束的序列
        episode = []
        while True:
            ac = agent.play(state, self.epsilon)
            next_state, reward, terminate, _ = env.step(ac)
            episode.append((state, ac, reward))
            state = next_state
            if terminate:
                break

        value = []
        return_val = 0
        for item in reversed(episode):
            return_val = return_val * agent.gamma + item[2]
            value.append((item[0], item[1], return_val))

        # 累计计算每一个状态行动
        for item in reversed(value):
            agent.value_n[item[0]][item[1]] += 1
            agent.value_q[item[0]][item[1]] += (item[2] - agent.value_q[item[0]][item[1]]) / agent.value_n[item[0]][item[1]]

    # 策略提升，这一步和策略迭代是一样的
    def policy_improve(self, agent):
        new_policy = np.zeros_like(agent.pi)
        for i in range(1, agent.s_len):
            new_policy[i] = np.argmax(agent.value_q[i, :])
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        else:
            agent.pi = new_policy
            return True

    # monte carlo迭代
    def monte_carlo_opt(self, agent, env):
        for i in range(10):
            for j in range(100):
                self.monte_carlo_eval(agent, env)
            self.policy_improve(agent)


def monte_carlo_demo():
    env = SnakeEnv(10, [3, 6])
    agent = ModelFreeAgent(env)
    mc = MonteCarlo()
    with timer('Timer Monte Carlo Iter'):
        mc.monte_carlo_opt(agent, env)
    print('return_pi={}'.format(eval_game(env, agent)))
    print(agent.pi)

    agent2 = TableAgent(env)
    pi_algo = PolicyIteration()
    with timer('Timer PolicyIter'):
        pi_algo.policy_iteration(agent2)
    print('return_pi={}'.format(eval_game(env, agent2)))
    print(agent2.pi)


def monte_carlo_demo2():
    env = SnakeEnv(10, [3, 6])
    agent = ModelFreeAgent(env)
    mc = MonteCarlo(0.5)
    with timer('Timer Monte Carlo Iter'):
        mc.monte_carlo_opt(agent, env)
    print('return_pi={}'.format(eval_game(env, agent)))
    print(agent.pi)


if __name__ == '__main__':
    monte_carlo_demo()
    monte_carlo_demo2()
