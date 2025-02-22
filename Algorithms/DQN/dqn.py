import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# Define the Neural Network Model for Q-Learning
class DQN(nn.Module):
    name = 'DQN'
    def __init__(self, state_size, action_size, arch=(400,300)):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, arch[0])
        self.fc2 = nn.Linear(arch[0], arch[1])
        self.fc3 = nn.Linear(arch[1], action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Prioritized Experience Replay Memory
class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = []
        self.pos = 0

    def __len__(self):
        return len(self.memory)

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if len(self.priorities) > 0 else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.memory[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*samples)

        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)

        # return samples, indices, weights
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, errors, offset=1e-6):
        # print(indices, errors)
        for i, error in zip(indices, errors):
            self.priorities[i] = (abs(error) + offset)

# DQN Agent
class DQNAgent:
    name = 'DQN'

    def __init__(self, state_size, action_size, buffer=None, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = buffer if buffer is not None else PrioritizedReplayMemory(100000)
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001

        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def select_action(self, state, device='cpu'):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_state_v = self.model(state)[0]
        _, action_v = torch.max(q_state_v, dim=1)
        _action = action_v.item()
        return _action

    def act(self, state, device='cpu'):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self, batch_size, beta=0.4, device='cpu'):
        # minibatch, indices, weights = self.memory.sample(batch_size, beta)
        # states, actions, rewards, next_states, dones = zip(*minibatch)
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size, beta)

        states = torch.FloatTensor(np.array(states)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)

        weights = torch.FloatTensor(weights).unsqueeze(1).to(device)  # 将权重转换为张量

        q_values = self.model(states).squeeze(1)
        next_q_values = self.target_model(next_states).squeeze(1)
        max_next_q_v = torch.max(next_q_values, dim=1)[0].unsqueeze(1)
        target = rewards + self.gamma * max_next_q_v * (1 - dones)
        predicted = q_values.gather(1, actions)

        errors = torch.abs(predicted - target).squeeze(1).detach().cpu().numpy()
        self.memory.update_priorities(indices, errors)

        loss = (weights * nn.MSELoss(reduction='none')(predicted, target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

