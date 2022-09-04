import argparse
import copy
# import importlib
# import json
import os
from pgym.pf_cases import case5bw
import time

import numpy as np
import torch
from torch import nn
import gym
from gym.wrappers import RescaleAction
from tensorboardX import SummaryWriter

import DDPG
import utils

summary = None


def interact_with_environment(env, replay_buffer, device, args, parameters):
    # For saving files
    setting = f"ddpg_{parameters['seed']}"

    # Initialize and load agent
    agent = DDPG.DDPG(
        env.action_space,
        env.observation_space,
        device=device,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
        discount=parameters["discount"],
        optimizer=parameters["optimizer"],
        optimizer_parameters=parameters["optimizer_parameters"],
        q_lr=1e-3,
        pi_lr=5e-4,
        polyak_target_update=parameters["polyak_target_update"],
        target_update_frequency=parameters["target_update_freq"],
        tau=parameters["tau"],
        noise=parameters["noise"],
    )

    evaluations = []

    state, done = env.reset(), False
    episode_start = True
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    pgym_pLoss = 0
    pgym_vViol = 0
    pgym_vViolRate = 0
    pgym_reward = 0

    # Interact with the environment for max_timesteps
    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        if t < parameters["start_timesteps"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(np.array(state))

        # Perform action and log results
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        done_float = float(done)

        replay_buffer.add(state, action, next_state, reward,
                          done_float, done, episode_start)
        state = copy.copy(next_state)
        episode_start = False

        # Train agent after collecting sufficient data
        if t >= parameters["start_timesteps"] and (t+1) % parameters["train_freq"] == 0:
            info = agent.train(replay_buffer)
            for k, v in info.items():
                summary.add_scalar(k, v, t)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_start = True
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        ind = env.get_indices()
        # pgym_pLoss += ind['pLoss']
        # pgym_vViol += ind['vViol']
        # pgym_vViolRate += ind['vViolRate']
        pgym_reward += reward
        if (t + 1) % 96 == 0:
            pgym_pLoss /= 96
            pgym_vViol /= 96
            pgym_vViolRate /= 96
            # summary.add_scalar('pLoss', pgym_pLoss, t)
            # summary.add_scalar('vViol', pgym_vViol, t)
            # summary.add_scalar('vViolRate', pgym_vViolRate, t)
            # summary.add_scalar('reward', pgym_reward, t)
            pgym_pLoss = 0
            pgym_vViol = 0
            pgym_vViolRate = 0
            pgym_reward = 0

        # Evaluate episode
        if (t + 1) % parameters["eval_freq"] == 0:
            evaluations.append(eval_agent(agent, parameters['env'], parameters['seed'], 1))
            np.save(f"./data/results/behavioral_{setting}", evaluations)
            agent.save(f"./data/models/behavioral_{setting}")

    # Save final agent
    agent.save(f"./data/models/behavioral_{setting}")


# Runs agent for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_agent(agent, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name, case=case5bw())
    eval_env = RescaleAction(eval_env, -1, 1)
    # eval_env = utils.RescaleObservation(eval_env, -1, 1)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = agent.select_action(np.array(state), eval=True)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    regular_parameters = {
        # Exploration
        "start_timesteps": int(1e2),
        "noise": 0.50,
        # Evaluation
        "eval_freq": int(5e2),
        # Learning
        "discount": 0.99,
        "buffer_size": int(1e6),
        "batch_size": 128,
        "optimizer": "Adam",
        "optimizer_parameters": {},
        "train_freq": 1,
        "polyak_target_update": True,
        "target_update_freq": 1,
        "tau": 0.005
    }

    # Load parameters
    parser = argparse.ArgumentParser()
    # OpenAI gym environment name
    parser.add_argument("--env", default="ContinuousVoltVarPowerflow-v0")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)
    # Max time steps to run environment or train for
    parser.add_argument("--max_timesteps", default=int(1e6), type=int)
    parser.add_argument("--buffer_size", default=int(1e6), type=int)
    args = parser.parse_args()

    _ = [os.makedirs(p) if not os.path.exists(p) else None
         for p in ['./data/results', './data/models', './data/buffers']]

    # Make env and determine properties
    env = gym.make(args.env, case=case5bw())
    env = RescaleAction(env, -1, 1)
    # env = utils.RescaleObservation(env, -1, 1)

    parameters = regular_parameters
    parameters.update(vars(args))

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize buffer
    replay_buffer = utils.StandardBuffer(
        env.observation_space.shape[0], env.action_space.shape[0], parameters["batch_size"], parameters["buffer_size"], device)

    _stamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
    base_path = f'{args.env}_{_stamp}'
    summary = SummaryWriter(f'./data/results/{base_path}')
    interact_with_environment(env, replay_buffer, device, args, parameters)
