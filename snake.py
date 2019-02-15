import numpy as np
import gym
from gym.spaces import Discrete

np.random.seed(0)


class SnakeEnv(gym.Env):
    SIZE = 100

    def __init__(self, ladder_num, actions):
        """
        :param int ladder_num: 梯子的个数
        :param list actions: 可选择的行为
        """
        self.ladder_num = ladder_num
        self.actions = actions
        # 在整个范围内，随机生成梯子
        self.ladders = dict(np.random.randint(1, self.SIZE, size=(self.ladder_num, 2)))
        self.observation_space = Discrete(self.SIZE + 1)
        self.action_space = Discrete(len(actions))

        # 因为梯子是两个方向的，所以添加反方向的梯子
        new_ladders = {}
        for k, v in self.ladders.items():
            new_ladders[k] = v
            new_ladders[v] = k
        self.ladders = new_ladders
        self.pos = 1

    # 重置初始状态
    def reset(self):
        self.pos = 1
        return self.pos

    def step(self, action):
        """
        :param int action: 选择的行动

        :return: 下一个状态，奖励值，是否结束，其它内容
        """
        step = np.random.randint(1, self.actions[action] + 1)
        self.pos += step
        if self.pos == 100:
            return 100, 100, 1, {}
        elif self.pos > 100:
            self.pos = 200 - self.pos

        if self.pos in self.ladders:
            self.pos = self.ladders[self.pos]
        return self.pos, -1, 0, {}

    # 返回状态s的奖励值
    def reward(self, s):
        if s == 100:
            return 100
        else:
            return -1

    # 绘制
    def render(self):
        pass


# 表格智能体
class TableAgent(object):
    def __init__(self, env):
        # 状态个数
        self.s_len = env.observation_space.n
        # 行动个数
        self.a_len = env.action_space.n
        # 每个状态的奖励,shape=[1,self.s_len]
        self.r = [env.reward(s) for s in range(0, self.s_len)]
        # 每个状态的行动策略,默认为0,shape=[1,self.s_len]
        self.pi = np.array([0 for s in range(0, self.s_len)])
        # 行动状态转移矩阵,shape=[self.a_len, self.s_len, self.s_len]
        self.p = np.zeros([self.a_len, self.s_len, self.s_len], dtype=np.float)
        # 梯子
        ladder_move = np.vectorize(lambda x: env.ladders[x] if x in env.ladders else x)

        # 计算状态s和行动a确定，下一个状态s'的概率
        for i, action in enumerate(env.actions):
            prob = 1.0 / action
            for src in range(1, 100):
                step = np.arange(action)
                step += src
                step = np.piecewise(step, [step > 100, step <= 100],
                                    [lambda x: 200 - x, lambda x: x])
                step = ladder_move(step)
                for dst in step:
                    self.p[i, src, dst] += prob

        self.p[:, 100, 100] = 1
        # 状态值函数
        self.value_pi = np.zeros((self.s_len))
        # 状态行动值函数
        self.value_q = np.zeros((self.s_len, self.a_len))
        # 衰减因子
        self.gamma = 0.8


    def play(self, state):
        """
        :param int state: 当前状态

        :return: 当前策略给出的下一步行动
        """
        return self.pi[state]


# 无模型的智能体
class ModelFreeAgent(object):
    def __init__(self, env):
        # 状态个数
        self.s_len = env.observation_space.n
        # 行动个数
        self.a_len = env.action_space.n
        # 每个状态的行动策略,默认为0,shape=[1,self.s_len]
        self.pi = np.array([0 for s in range(0, self.s_len)])
        # 状态行动值函数
        self.value_q = np.zeros((self.s_len, self.a_len))
        # 状态行动出现的次数
        self.value_n = np.zeros((self.s_len, self.a_len))
        # 衰减因子
        self.gamma = 0.8

    def play(self, state, epsilon=0):
        """
        :param int state: 当前状态
        :param int epsilon: 探索率，有一定概率随机选择下一步行动

        :return: 当前策略给出的下一步行动
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.a_len)
        else:
            return self.pi[state]


def eval_game(env, policy):
    """
    :param gym.Env env: 环境对象
    :param object policy: 运行环境的策略

    :return: 最终获得的奖励
    """
    state = env.reset()
    return_val = 0
    while True:
        if isinstance(policy, TableAgent) or isinstance(policy, ModelFreeAgent):
            act = policy.play(state)
        elif isinstance(policy, list):
            act = policy[state]
        else:
            raise Exception('Illegal policy')
        state, reward, terminate, _ = env.step(act)
        return_val += reward

        if terminate:
            break
    return return_val


if __name__ == '__main__':
    env = SnakeEnv(10, [3, 6])
    env.reset()
    while True:
        state, reward, terminate, _ = env.step(0)
        print(reward, state)
        if terminate == 1:
            break
