import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class Actor(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, 400)
        self.linear2 = nn.Linear(400, 100)
        self.linear3 = nn.Linear(100, 100)
        self.linear4 = nn.Linear(100, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.softmax(self.linear4(x), dim=1)

        return x


class Critic(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.linearS1 = nn.Linear(state_dim, 200)
        self.linearS2 = nn.Linear(200, 100)
        self.linearA1 = nn.Linear(num_actions, 50)
        self.linearA2 = nn.Linear(50, 100)
        self.out = nn.Linear(100, 1)

    def forward(self, state, action):
        x = F.relu(self.linearS1(state))
        x = F.relu(self.linearS2(x))
        y = F.relu(self.linearA1(action))
        y = F.relu(self.linearA2(y))
        action_value = F.relu(x + y)
        action_value = self.out(action_value)

        return action_value


class Agent(object):
    def __init__(
            self,
            num_actions,
            state_dim,
            device,
            discount=0.99,
            optimizer="Adam",
            actor_optimizer_parameters={},
            critic_optimizer_parameters={},
            polyak_target_update=False,
            target_update_frequency=1,
            tau=0.005,
            initial_eps=1,
            end_eps=0.001,
            eps_decay_period=25e4,
            eval_eps=0.001,
            buffer_size=2000000,
            batch_size=64,
    ):

        self.device = device

        # Determine network type
        self.actor = Actor(state_dim, num_actions).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, num_actions).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optim = getattr(torch.optim, optimizer)(self.actor.parameters(), **actor_optimizer_parameters)
        self.critic_optim = getattr(torch.optim, optimizer)(self.critic.parameters(), **critic_optimizer_parameters)
        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.discount = discount

        # Target update rule(to be continued)
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Decay for eps
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyper-parameters
        self.state_dim = state_dim
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eval=False):
        eps = self.eval_eps if eval \
            else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0, 1) > eps:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            action = self.actor(state).squeeze(0).detach().numpy()
            return action
        else:
            action = np.zeros(self.num_actions)
            action[np.random.randint(self.num_actions)] = 1
            return action

    def put(self, *transition):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        # Sample from buffer
        samples = random.sample(self.buffer, self.batch_size)

        s0, a0, s1, r1, done_float, done, episode_start = zip(*samples)

        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)

        # Critic learn
        def critic_learn():
            a1 = self.actor_target(s1).detach()

            # Compute the target value
            y_true = r1 + self.discount * self.critic_target(s1, a1).detach()

            # Get current estimate
            y_pred = self.critic(s0, a0)

            # Compute loss and optimize
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():
            loss = -torch.mean(self.critic(s0, self.actor(s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        critic_learn()
        actor_learn()
        self.iterations += 1
        soft_update(self.critic_target, self.critic)
        soft_update(self.actor_target, self.actor)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optim.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim.load_state_dict(torch.load(filename + "_optimizer"))
