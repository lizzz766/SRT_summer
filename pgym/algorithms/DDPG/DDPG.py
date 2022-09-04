import copy
import numpy as np
import torch
import torch.nn as nn
from utils import mlp

# TODO: to device


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_high, act_low):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_high = act_high
        self.act_low = act_low

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        pi_action = self.pi(obs)
        pi_action = 0.5 * torch.FloatTensor(self.act_high - self.act_low) * pi_action + \
            0.5 * torch.FloatTensor(self.act_high + self.act_low)
        return pi_action


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] +
                     list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class DDPG(object):
    def __init__(
            self,
            action_space,
            state_space,
            hidden_sizes=(256, 256),
            activation=nn.ReLU,
            device="cpu",
            discount=0.99,
            optimizer="Adam",
            optimizer_parameters={},
            q_lr=1e-3,
            pi_lr=5e-4,
            polyak_target_update=False,
            target_update_frequency=8e3,
            tau=0.005,
            noise=0.05,
        ):
        state_dim = state_space.shape[0]
        self.act_dim = action_space.shape[0]

        self.device = device
        self.act_high = action_space.high
        self.act_low = action_space.low

        # Determine network type
        self.Q = MLPQFunction(state_dim, self.act_dim, hidden_sizes, activation).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), lr=q_lr, **optimizer_parameters)

        self.pi = MLPActor(state_dim, self.act_dim, hidden_sizes, activation,
                           self.act_high, self.act_low).to(self.device)
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optimizer = getattr(torch.optim, optimizer)(self.pi.parameters(), lr=pi_lr, **optimizer_parameters)

        self.discount = discount
        self.noise = noise

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eval=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            a = self.pi(state).clone().detach().cpu().numpy()
            if not eval:
                a += self.noise * np.random.randn(self.act_dim) * (self.act_high - self.act_low)
            return np.clip(a, self.act_low, self.act_high)

    def train(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample()

        current_Q = self.Q(state, action)
        next_state_values = self.Q_target(next_state, self.pi_target(next_state))

        target_Q = reward.squeeze() + not_done.squeeze() * next_state_values * self.discount
        Q_loss = nn.MSELoss()(current_Q, target_Q)

        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        for p in self.Q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        pi_loss = -self.Q(state, self.pi(state)).mean()
        pi_loss.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.Q.parameters():
            p.requires_grad = True

        self.iterations += 1
        self.maybe_update_target()

        return {
            'q_loss': Q_loss.mean().detach().cpu().numpy(),
            'q': current_Q.mean().detach().cpu().numpy(),
            'pi_loss': pi_loss.mean().detach().cpu().numpy(),
        }

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.pi.parameters(), self.pi_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())
            self.pi_target.load_state_dict(self.pi.state_dict())

    def save(self, filename):
        torch.save(self.Q.state_dict(), filename + "_Q")
        torch.save(self.Q_optimizer.state_dict(), filename + "_optimizer")
        torch.save(self.pi.state_dict(), filename + "_pi")
        torch.save(self.pi_optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.Q.load_state_dict(torch.load(filename + "_Q"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "_optimizer"))
        self.pi.load_state_dict(torch.load(filename + "_pi"))
        self.pi_target = copy.deepcopy(self.pi)
        self.pi_optimizer.load_state_dict(torch.load(filename + "_optimizer"))
