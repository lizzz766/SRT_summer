import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import random

import ray
from ray import tune
# from ray.rllib.agents import ppo, sac
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.typing import EnvConfigDict
from ray.tune.logger import pretty_print
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=20,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=None,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=0.1,
    help="Reward at which we stop training.")
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.")
parser.add_argument(
    "--test-env",
    action="store_true",
    help="Test the environment by random.")


class SimpleStorage(gym.Env):
    def __init__(self, config: EnvContext):
        self.data = np.array([0.19590032137490215, 0.198648461574976, 0.16198401085551553, 0.1769507588221408, 0.1508672025370697, 0.18112487503607225, 0.16158449636743275, 0.15697892098339658, 0.15218193289626028, 0.17671253552302713, 0.16794988198147923, 0.17877865769122483, 0.16320551221178312, 0.1973489352910529, 0.2074261505440902, 0.21925665020231877, 0.2459062931892704, 0.24902034685973493, 0.2507833481728078, 0.27051346903632145, 0.29904946328151927, 0.29087324206099147, 0.33606776449466763, 0.35692761574754467, 0.35925948798028956, 0.3785021406509885, 0.4099711187155093, 0.43655766624961095, 0.4707101830408831, 0.4915292800146178, 0.5418797327432164, 0.5326618795691772, 0.5660487906748679, 0.5937431063052329, 0.6449086837236409, 0.6593251384338568, 0.7002604245257896, 0.7233588199118013, 0.7513456779483263, 0.7784651542546142, 0.8266109265207954, 0.8244393847252524, 0.8394023602224316, 0.8504633696262806, 0.8630545795711453, 0.8151144073447284, 0.8200178772050766,
                              0.7568183893719281, 0.7591796049264569, 0.7035066830359146, 0.693981470875513, 0.6538913127146639, 0.6446040696220682, 0.6119535416523315, 0.629759793663206, 0.6394425677263711, 0.6705421754660823, 0.692159829500676, 0.7332149337438222, 0.7717383700813122, 0.7914917133177101, 0.8508542285984754, 0.8762977434380528, 0.8917646566125703, 0.9277414026338864, 0.9313921303345174, 0.9695530386863624, 0.9980655473024352, 0.9813760871136922, 1.0, 0.9759536632985123, 0.9931578462314223, 0.9615313910210191, 0.9369062578375564, 0.9278146309110206, 0.8728145195747647, 0.8713422683923363, 0.7976314592798595, 0.7841620629260145, 0.7232578082851697, 0.697904671760363, 0.6620927254465575, 0.5868066349872124, 0.5558690996428556, 0.5128631095518045, 0.46986724864657003, 0.4150001409598635, 0.40797724428059917, 0.35192165242233203, 0.33734189488513494, 0.30366628971318566, 0.2636151622509874, 0.2670304924185861, 0.2364739264150809, 0.24963769165980274, 0.2360557659148955])
        self.CAP = 10#上面给定了一条曲线，但是没有那么复杂；这个程序的写，本质就是维护了一个状态机。
        self.cur_cap = 8#容值；目前负载
        self.t = 0#时间
        self.actual = np.zeros_like(self.data)
        self.action_space = Box(-0.7, 0.7, shape=(1, ), dtype=np.float32)
        # obs: p(t-2),p(t-1),p(t),cap(t)
        self.observation_space = Box(
            np.array([-1, -1, -1, 0]),
            np.array([1, 1, 1, self.CAP]),
            dtype=np.float32
        )

        self.seed(config.worker_index * config.num_workers)

    def reset(self):
        self.cur_cap = 8
        self.t = 0
        self.actual = np.zeros_like(self.data)
        return [0, 0, 0, self.cur_cap]

    def step(self, action):
        sign = np.sign(action[0])#符号函数
        val = np.abs(action[0])#一种状态机

        if sign < 0:
            if self.cur_cap + val * 0.81 > self.CAP:
                val = (self.CAP - self.cur_cap) / 0.81
            self.actual[self.t] = self.data[self.t] + val#data：未储能的模型；actual：储能后的值
            # 充电成本 + 电能质量
            reward = -val * 0.1 - np.abs(self.actual[self.t] - 0.75) * 3#越接近0.75越好
            # 充电
            self.cur_cap += val * 0.81
        else:
            if self.cur_cap - val / 0.81 < 0:
                val = self.cur_cap * 0.81
            self.actual[self.t] = self.data[self.t] - val
            # 放电成本 + 电能质量
            reward = -val * 0.1 - np.abs(self.actual[self.t] - 0.75) * 3#偏离0.75就要惩罚一些；储能就是响应这条曲线
            # 放电
            self.cur_cap -= val / 0.81
        self.cur_cap = np.clip(self.cur_cap, 0, self.CAP)

        obs = np.array([
            self.data[self.t - 2] if self.t > 1 else 0,
            self.data[self.t - 1] if self.t > 0 else 0,
            self.data[self.t], self.cur_cap
        ])#更新

        self.t += 1
        done = self.t == len(self.data)
        if done:
            self.t = 0

        return obs, reward, done, {}

    def seed(self, seed=None):
        random.seed(seed)


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    if args.test_env:
        env = SimpleStorage(EnvContext(EnvConfigDict(), worker_index=0, num_workers=0))
        for t in range(192):
            obs, rew, done, info = env.step(env.action_space.sample())
            print(obs, rew, done)
            if done:
                env.reset()
        exit(0)

    ray.init(local_mode=args.local_mode)

    config = {
        "env": SimpleStorage,
        "env_config": {
        },#去RLlib查参数的作用

        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "vf_share_layers": True,
        },
        "num_workers": 1,  # parallelism
        "framework": "torch"
    }

    if args.run == 'SAC':
        config["initial_alpha"] = 0.5
        # config["optimization"] = {"entropy_learning_rate": 1e-4}

    stop = {
        "training_iteration": args.stop_iters,
        "episode_reward_mean": args.stop_reward,
    }
    if args.stop_timesteps is not None:
        stop["timesteps_total"] = args.stop_timesteps

    # automated run with Tune and grid search and TensorBoard
    print("Training automatically with Ray Tune")
    results = tune.run(args.run, config=config, stop=stop, checkpoint_freq=5)

    if args.as_test:
        print("Checking if learning goals were achieved")
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
#跑；复现；理解；目前1. Establish more complicated storage models from papers;建立更加复杂的模型
#2. Optimize multiple storages simultaneously;actionspace变成多个了；在一个复杂网络的不同节点上；学习“潮流计算”，电力系统的静态仿真
#给了控制无功功率的程序
#3. Introduce power flow model to the environment (see `pgym.zip`);