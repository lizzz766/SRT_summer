import argparse
import copy
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
import pgym
import gym

from DDPG import Agent

def make_env(env_name):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    return (
        env,
        state_dim,
        env.action_space.n
    )


summary = None


def interact_with_environment(env, num_actions, state_dim, device, args, parameters):
    # For saving files
    setting = f"{args.env}_{args.seed}"

    # Initialize and load policy
    policy = Agent(
        num_actions,
        state_dim,
        device,
        parameters["discount"],
        parameters["optimizer"],
        parameters["actor_optimizer_parameters"],
        parameters["critic_optimizer_parameters"],
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"],
        parameters["initial_eps"],
        parameters["end_eps"],
        parameters["eps_decay_period"],
        parameters["eval_eps"],
        parameters["buffer_size"],
        parameters["batch_size"],
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
        action = policy.select_action(np.array(state))

        # Perform action and log results
        next_state, reward, done, _ = env.step(int(action.argmax()))
        episode_reward += reward
        done_float = float(done)

        policy.put(state, action, next_state, reward,
                   done_float, done, episode_start)
        state = copy.copy(next_state)
        episode_start = False

        # Train agent after collecting sufficient data
        if t >= parameters["start_timesteps"] and (t + 1) % parameters["train_freq"] == 0:
            policy.learn()

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T:{t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}"
            )
            # reset environment
            state, done = env.reset(), False
            episode_start = True
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        ind = env.get_indices()
        pgym_pLoss += ind['pLoss']
        pgym_vViol += ind['vViol']
        pgym_vViolRate += ind['vViolRate']
        pgym_reward += reward
        if (t + 1) % 96 == 0:
            pgym_pLoss /= 96
            pgym_vViol /= 96
            pgym_vViolRate /= 96
            summary.add_scalar('pLoss', pgym_pLoss, t)
            summary.add_scalar('vViol', pgym_vViol, t)
            summary.add_scalar('vViolRate', pgym_vViolRate, t)
            summary.add_scalar('reward', pgym_reward, t)
            pgym_pLoss = 0
            pgym_vViol = 0
            pgym_vViolRate = 0
            pgym_reward = 0

        # Evaluate episode
        if (t + 1) % parameters["eval_freq"] == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed, 1))
            np.save(f"./data/results/behavioral_{setting}", evaluations)
            policy.save(f"./data/models/behavioral_{setting}")

    # Save final policy
    policy.save(f"./data/models/behavioral_{setting}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env, _, _ = make_env(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), eval=True)
            state, reward, done, _ = eval_env.step(int(action.argmax()))
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
        "initial_eps": 0.1,
        "end_eps": 0.1,
        "eps_decay_period": 1,
        # Evaluation
        "eval_freq": int(5e2),
        "eval_eps": 0,
        # Learning
        "discount": 0.99,
        "buffer_size": int(2e6),
        "batch_size": 64,
        "optimizer": "Adam",
        "actor_optimizer_parameters": {
            "lr": 3e-4
        },
        "critic_optimizer_parameters": {
            "lr": 3e-4
        },
        "train_freq": 1,
        "polyak_target_update": True,
        "target_update_freq": 1,
        "tau": 0.005
    }

    # Load parameters
    parser = argparse.ArgumentParser()
    # OpenAI gym environment name
    parser.add_argument("--env", default="DiscreteVoltVarPowerflow-v0")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)
    # Prepends name to filename
    parser.add_argument("--buffer_name", default="Default")
    # Max time steps to run environment or train for
    parser.add_argument("--max_timesteps", default=2 * 1e4, type=int)
    parser.add_argument("--buffer_size", default=2e6, type=int)
    args = parser.parse_args()

    if not os.path.exists("./data/results"):
        os.makedirs("./data/results")

    if not os.path.exists("./data/models"):
        os.makedirs("./data/models")

    if not os.path.exists("./data/buffers"):
        os.makedirs("./data/buffers")

    # Make env and determine properties
    env, state_dim, num_actions = make_env(args.env)
    parameters = regular_parameters
    parameters.update(vars(args))

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary = SummaryWriter('./data/results')

    interact_with_environment(env, num_actions, state_dim, device, args, parameters)
