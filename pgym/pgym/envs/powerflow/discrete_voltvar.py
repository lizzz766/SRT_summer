from pgym.envs.powerflow.voltvar import VoltVarEnv
from pgym.pf_cases import case5bw
import numpy as np
from numpy import array
from pypower.idx_gen import QG, QMAX, QMIN
from gym import spaces


class DiscreteVoltVarEnv(VoltVarEnv):

    def __init__(self, case=None, **kwargs):
        if case is None:
            case = case5bw()
        info = {
            'case_name': 'dvvpfe',
            'sigma': 5,                 # discrete pieces
        }
        info.update(kwargs)
        self.sigma = info['sigma']
        super().__init__(case, **info)

    def get_action_space(self):
        cnt = len(self.case0['controlled_gen'][:, 0])
        min_action = np.array([self.case0['gen'][i, QMIN]
                               for i in self.case0['controlled_gen'][:, 0]])
        max_action = np.array([self.case0['gen'][i, QMAX]
                               for i in self.case0['controlled_gen'][:, 0]])
        self.action_scale = (max_action - min_action) / self.sigma
        return spaces.Discrete(cnt * 2), min_action, max_action

    def put_action(self, action):
        case = self.case
        k = action // 2
        sign = 1 - 2 * (action % 2)
        # print(sign)
        tov = case['gen'][case['controlled_gen']
                          [k, 0], QG] + sign * self.action_scale[k]
        tov = max([tov, self.min_action[k]])
        tov = min([tov, self.max_action[k]])
        case['gen'][case['controlled_gen'][k, 0], QG] = tov
        return case
