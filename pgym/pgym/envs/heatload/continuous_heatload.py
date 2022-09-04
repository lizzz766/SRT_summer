# Heat Load environment

from copy import deepcopy
import numpy as np
import gym
from pgym.hl_cases import hl_case
from pgym.hl_cases.idx import LOAD_I, PL, LMAX, LMIN, TR, TA, TMAX, TMIN, Rh, Ch, Th, Sh, Qh, Pout
from pypower.api import runpf, ppoption
from pypower.idx_bus import VM, VA, PD
from pypower.idx_gen import PG, PMAX, PMIN


ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)

class Observation(dict):

    def numpy(self):
        return np.concatenate([v for v in self.values()])


class ContinuousHeatLoadEnv(gym.Env):

    # Parameters:
    ## Name     Value               Default
    ### VR      volt punish ratio   None
    ### TVSpd   time-variant speed  None
    def __init__(self, **kwargs):
        info = {
            'T': 96,                            # period
            'case_name': 'heatload_case',       # case name
            'log_dir': 'data/',                 # save image directory
            'reward_func': 'heat_comfort',    # reward function
            # 'reward_func': 'auxi_service',    # reward function
            # 'reward_func': 'power_losses',    # reward function
            # 'reward_func': 'comprehensive',     # reward function
            'failed_reward': -500,
            'load_amp_factor': 1000.0,          # load amplification factor
            'ratio_heat_comfort': 1,            # heat comfort ratio
            'ratio_auxi_service': 10,           # auxi service ratio
            'ratio_power_losses': 80,          # power losses ratio
        }
        case = hl_case()
        info.update(kwargs)
        self.T = info['T']
        self.case_name = info['case_name']
        self.log_dir = info['log_dir']
        self.failed_reward = info['failed_reward']
        self.reward_func = info['reward_func']
        self.load_amp = info['load_amp_factor']
        self.rhc = info['ratio_heat_comfort']
        self.ras = info['ratio_auxi_service']
        self.rpl = info['ratio_power_losses']
        self.case0 = deepcopy(case)
        self.case = deepcopy(case)

        self.time = 0
        self.abs_time = 0 # absolute time
        self.ax = None
        self.rendered = False

        self.action_space, self.min_action, self.max_action = self.get_action_space()
        self.observation_space, self.low_state, self.high_state = self.get_observation_space()
        self.last_obs = self.get_observation()
        self.reward = self.get_reward_from_results(self.case)

        self.seed()
        self.reset()

    # return seed
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # return self.observation_space, self.low_state, self.high_state
    def get_observation_space(self):
        def get_obs_bound(case0):
            pgl = case0['gen'][:, PMIN]
            pdl = np.array([0 for p in case0['bus'][:, PD]])
            trl = case0['load'][:, TMIN]
            tal = case0['load'][:, TMIN]
            pll = case0['load'][:, LMIN]

            pgm = case0['gen'][:, PMAX] + 1
            pdm = np.array([p * 5 for p in case0['bus'][:, PD]]) + 1
            trm = case0['load'][:, TMAX] + 1
            tam = case0['load'][:, TMAX] + 1
            plm = case0['load'][:, LMAX] + 1

            return np.concatenate([pgl, pdl, trl, tal, pll]), np.concatenate([pgm, pdm, trm, tam, plm])

        self.low_state, self.high_state = get_obs_bound(self.case0)
        self.observation_space = gym.spaces.Box(
            low=self.low_state.astype(np.float32),
            high=self.high_state.astype(np.float32),
            dtype=np.float32
        )
        return self.observation_space, self.low_state, self.high_state

    # return Observation
    def get_observation(self, case=None):
        if case is None:
            case = self.case
        obs = Observation()
        obs['pg'] = case['gen'][:, PG].copy()
        obs['pd'] = case['bus'][:, PD].copy()
        obs['tr'] = case['load'][:, TR].copy()
        obs['ta'] = case['load'][:, TA].copy()
        obs['pl'] = case['load'][:, PL].copy()
        return obs

    # return indices
    def get_indices(self, obs=None):
        if not obs:
            obs = self.get_observation()
        # TViol = sum(obs['tr'] > 299) + sum(obs['tr'] < 295)
        TViol = np.sum(obs['tr'][obs['tr'] > 297 + 2] - (297 + 2)) - \
            np.sum(obs['tr'][obs['tr'] < 297 - 2] - (297 - 2))
        # AQual = np.sum(np.abs(obs['pl'])) * (2 * self.case['load_curve'][self.time] - np.max(self.case['load_curve'][0:self.T]) - np.min(self.case['load_curve'][0:self.T])) / (np.max(self.case['load_curve'][0:self.T]) - np.min(self.case['load_curve'][0:self.T]))
        AQual = obs['pg'][0] * (2 * self.case['load_curve'][self.time] - np.max(self.case['load_curve'][0:self.T]) - np.min(
            self.case['load_curve'][0:self.T])) / (np.max(self.case['load_curve'][0:self.T]) - np.min(self.case['load_curve'][0:self.T]))
        PLoss = np.abs(sum(obs['pg']) - sum(obs['pd']))
        return {
            'TViol': TViol,
            'AQual': AQual,
            'PLoss': PLoss,
        }

    # return reward
    def get_reward_from_results(self, results):
        indices = self.get_indices(self.get_observation(results))
        if self.reward_func == 'heat_comfort':
            return - indices['TViol']
        if self.reward_func == 'auxi_service':
            return - indices['AQual']
        if self.reward_func == 'power_losses':
            return - indices['PLoss']
        if self.reward_func == 'comprehensive':
            return - self.rhc * indices['TViol'] - self.ras * indices['AQual'] - self.rpl * indices['PLoss']
        return 0

    # return spaces.Discrete, low_ctrl, high_ctrl
    def get_action_space(self):
        min_action = np.array([
            self.case0['load'][i, LMIN] for i in self.case0['controlled_load'][:, 0]
        ]) * 1e-3
        max_action = np.array([
            self.case0['load'][i, LMAX] for i in self.case0['controlled_load'][:, 0]
        ]) * 1e-3
        return gym.spaces.Box(min_action.astype(np.float32), max_action.astype(np.float32)), min_action, max_action

    # return case
    def put_action(self, action):
        case = self.case
        tov = np.clip(action, self.min_action, self.max_action)
        case['load'][case['controlled_load'][:, 0], PL] = tov * 1e3
        return case

    # return observation
    def step(self, action):
        # manage time-variant variables
        self.last_obs = self.get_observation()
        # run action
        self.put_action(action)
        self.case['bus'][self.case['load'][:, LOAD_I].astype(int), PD] = self.case0['bus'][self.case['load'][:, LOAD_I].astype(
            int), PD] + np.abs(self.case['load'][:, PL]) / 1e6 * self.load_amp
        # reset to default origin
        tmp_case = deepcopy(self.case)
        self.case['bus'][:, VM] = 1
        self.case['bus'][:, VA] = 0
        # run powerflow
        results, self.success = runpf(self.case, ppopt)
        if not self.success:
            done = True
            results = tmp_case
            obs = self.get_observation(results)
            self.reward = self.failed_reward
        else:
            self.case = results
            self.case['load'][:, Pout] = self.case['gen'][0, PG]
            # change outdoor temperature
            self.case['load'][:, TA] = np.array([
                self.case0['temperature'][self.time] for i in self.case0['load'][:, 0]
            ])
            # change indoor temperature
            self.case['load'][:, TR] = self.case['load'][:, Th] / self.case['load'][:, Ch] * ((self.case['load'][:, TA] - self.case['load'][:, TR]) / self.case['load'][:, Rh] + (self.case['load'][:, PL] + self.case['load'][:, Qh]) / self.case['load'][:, Sh]) + self.case['load'][:, TR]
            # get observation
            done = False
            obs = self.get_observation(self.case)
            self.reward = self.get_reward_from_results(self.case)


        if (self.time + 1) % self.T == 0:
            done = True
        # next state
        self.time += 1
        self.abs_time += 1
        ## manage time-variant variables
        return obs.numpy(), self.reward, done, {}


    def reset(self, absolute=False):
        # reset state
        self.time = 0
        self.case = deepcopy(self.case0)
        if absolute:
            self.abs_time = 0
        results, self.success = runpf(self.case, ppopt)
        assert self.success, "Reset case unsolvable"
        self.case = results
        obs = self.get_observation()
        self.last_obs = obs
        return obs.numpy()
