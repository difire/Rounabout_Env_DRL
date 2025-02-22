import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()  # 获取输入形状

        # Linear transformations
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scale_factor = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=x.device))
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale_factor
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, V)

        # Concatenate heads
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.embed_dim)

        # Final linear layer
        out = self.fc_out(out)
        return out


# 定义Actor网络（策略网络）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # 调整注意力机制
        self.attention = MultiHeadAttention(embed_dim=state_dim, num_heads=2)  # 减少头数
        # 调整网络结构
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)  # 添加层归一化
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)

        # 调整输出层初始化
        self.mu = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

        # 初始化参数
        nn.init.orthogonal_(self.mu.weight, gain=0.01)
        nn.init.constant_(self.mu.bias, 0)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.constant_(self.log_std.bias, 0)

        self.max_action = max_action
        self.action_scale = torch.tensor(max_action, dtype=torch.float32)
        self.action_bias = torch.tensor(0.0, dtype=torch.float32)

    def forward(self, state):
        # 调整注意力机制
        state = self.attention(state.unsqueeze(1)).squeeze(1)
        # 添加噪声和正则化
        state = state + 0.1 * torch.randn_like(state)  # 添加噪声
        x = F.elu(self.ln1(self.fc1(state)))  # 使用ELU激活函数
        x = F.elu(self.ln2(self.fc2(x)))

        mu = self.mu(x)          # mu代表动作的均值，log_std代表动作的标准差的对数。
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-5, max=2)  # 调整范围
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal_dist = dist.Normal(mu, std)

        # 改进采样过程
        x_t = normal_dist.rsample()  # 重新采样,通过重参数化技巧（reparameterization trick）从正态分布中采样动作，即生成一个符合 N(mu, std) 分布的样本 x_t。
        y_t = torch.tanh(x_t)        # 修正输出范围,将动作缩放到 [-1, 1] 之间

        # 调整动作缩放
        action = y_t * self.action_scale + self.action_bias  # 调整动作范围,将动作缩放到实际的动作范围内

        # 计算对数概率
        log_prob = normal_dist.log_prob(x_t)

        # 修正对数概率计算
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


# 定义Critic网络（Q网络）
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.attention = MultiHeadAttention(embed_dim=state_dim, num_heads=2)
        self.fc1 = nn.Linear(state_dim + action_dim, 500)
        self.fc2 = nn.Linear(500, 400)
        self.fc3 = nn.Linear(400, 1)

    def forward(self, state, action):
        # Apply attention
        state = self.attention(state.unsqueeze(1)).squeeze(1)
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# SAC算法
class SAC:
    name = "Attention_SAC"
    def __init__(self, state_dim, action_dim, max_action, replay_buffer, device, batch_size=256, gamma=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1 = Critic(state_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim).to(device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        # 自动调整温度参数α
        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.replay_buffer = replay_buffer
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).to(self.device)
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)
        # 更新Critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = reward + not_done * self.gamma * target_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 更新Actor
        action, log_prob = self.actor.sample(state)
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        q = torch.min(q1, q2)
        actor_loss = (self.log_alpha.exp() * log_prob - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新温度参数α
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # 更新目标网络
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")
        torch.save(self.critic2.state_dict(), filename + "_critic2")
        torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer"))
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.critic2.load_state_dict(torch.load(filename + "_critic2"))
        self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer"))
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
