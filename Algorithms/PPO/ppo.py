import os
import gym
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import numpy as np
from collections import deque, namedtuple
import random
import torch.distributions as dist

# 定义经验回放池
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# 定义Actor网络（策略网络）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.mu = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-5, max=2)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal_dist = dist.Normal(mu, std)
        action = normal_dist.rsample()
        log_prob = normal_dist.log_prob(action).sum(1, keepdim=True)
        action = torch.tanh(action) * self.max_action
        return action, log_prob

# 定义Critic网络（值函数网络）
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# PPO算法
class PPO:
    name = "PPO"
    def __init__(self, state_dim, action_dim, max_action, device):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer(1000000)
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.95
        self.epsilon = 0.2
        self.device = device

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action, log_prob = self.actor.sample(state)
        value = self.critic(state)
        return action.cpu().data.numpy()[0], log_prob.cpu().data.numpy()[0], value.cpu().data.numpy()[0]

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device).squeeze(1)
        action_batch = torch.FloatTensor(np.array(batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device).squeeze(1)
        done_batch = torch.FloatTensor(np.array(batch.done)).to(self.device).unsqueeze(1)
        log_prob_batch = torch.FloatTensor(np.array(batch.log_prob)).to(self.device)
        value_batch = torch.FloatTensor(np.array(batch.value)).to(self.device)

        # 计算GAE
        with torch.no_grad():
            next_value = self.critic(next_state_batch)
            delta = reward_batch + (1 - done_batch) * self.gamma * next_value - value_batch
            advantage = torch.zeros_like(delta).to(self.device)
            advantage[-1] = delta[-1]
            for t in reversed(range(len(delta) - 1)):
                advantage[t] = delta[t] + (1 - done_batch[t]) * self.gamma * self.tau * advantage[t + 1]
            returns = advantage + value_batch

        # 更新Critic
        values = self.critic(state_batch)
        critic_loss = F.mse_loss(values, returns)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor
        new_actions, new_log_probs = self.actor.sample(state_batch)
        ratio = (new_log_probs - log_prob_batch).exp()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()