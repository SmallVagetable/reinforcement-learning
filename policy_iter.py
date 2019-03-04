import numpy as np

from snake import SnakeEnv, TableAgent, eval_game

# 前97个选择1号行动，后3个选择0号行动
policy_ref = [1] * 97 + [0] * 3
policy_0 = [0] * 100
policy_1 = [1] * 100


def first_easy():
    sum_opt = 0
    sum_0 = 0
    sum_1 = 0
    env = SnakeEnv(0, [3, 6])
    countNum = 10000
    for i in range(countNum):
        sum_opt += eval_game(env, policy_ref)
        sum_0 += eval_game(env, policy_0)
        sum_1 += eval_game(env, policy_1)
    print('policy_ref avg={}'.format(sum_opt / countNum))
    print('policy_0 avg={}'.format(sum_0 / countNum))
    print('policy_1 avg={}'.format(sum_1 / countNum))


class PolicyIteration(object):

    # 策略评估
    def policy_evaluation(self, agent, max_iter=-1):
        """
        :param obj agent: 智能体
        :param int max_iter: 最大迭代数
        """
        iteration = 0

        while True:
            iteration += 1
            new_value_pi = agent.value_pi.copy()
            # 对每个state计算v(s)
            for i in range(1, agent.s_len):
                ac = agent.pi[i]
                transition = agent.p[ac, i, :]
                value_sa = np.dot(transition, agent.r + agent.gamma * agent.value_pi)
                new_value_pi[i] = value_sa

            # 前后2次值函数的变化小于一个阈值，结束
            diff = np.sqrt(np.sum(np.power(agent.value_pi - new_value_pi, 2)))
            if diff < 1e-6:
                break
            else:
                agent.value_pi = new_value_pi
            if iteration == max_iter:
                break

    # 策略提升
    def policy_improvement(self, agent):
        """
        :param obj agent: 智能体
        """

        # 初始化新策略
        new_policy = np.zeros_like(agent.pi)
        for i in range(1, agent.s_len):
            for j in range(0, agent.a_len):
                # 计算每一个状态行动值函数
                agent.value_q[i, j] = np.dot(agent.p[j, i, :], agent.r + agent.gamma * agent.value_pi)

            # 选出每个状态下的最优行动
            max_act = np.argmax(agent.value_q[i, :])
            new_policy[i] = max_act
        if np.all(np.equal(new_policy, agent.pi)):
            return False
        else:
            agent.pi = new_policy
            return True

    # 策略迭代
    def policy_iteration(self, agent):
        """
        :param obj agent: 智能体
        """
        iteration = 0
        while True:
            iteration += 1
            self.policy_evaluation(agent)
            ret = self.policy_improvement(agent)
            if not ret:
                break
        print('Iter {} rounds converge'.format(iteration))


# 测试没有梯子时，最优的策略
def policy_iteration_demo1():
    env = SnakeEnv(0, [3, 6])
    agent = TableAgent(env)
    pi_algo = PolicyIteration()
    pi_algo.policy_iteration(agent)
    print('return_pi={}'.format(eval_game(env, agent)))
    print(agent.pi)


# 测试有梯子时，不同策略和最优的策略的差别
def policy_iteration_demo2():
    env = SnakeEnv(10, [3, 6])
    agent = TableAgent(env)
    agent.pi[:] = 0
    print('return3={}'.format(eval_game(env, agent)))
    agent.pi[:] = 1
    print('return6={}'.format(eval_game(env, agent)))
    agent.pi[97:100] = 0
    print('return_ensemble={}'.format(eval_game(env, agent)))
    pi_algo = PolicyIteration()
    pi_algo.policy_iteration(agent)
    print('return_pi={}'.format(eval_game(env, agent)))
    print(agent.pi)


if __name__ == '__main__':
    first_easy()
    policy_iteration_demo1()
    policy_iteration_demo2()
