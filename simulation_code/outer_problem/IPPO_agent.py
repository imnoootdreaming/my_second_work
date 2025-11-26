import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Beta, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


# orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Gaussian(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_low, action_high, device):
        super(Actor_Gaussian, self).__init__()
        self.max_action = action_high
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = nn.Tanh()
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)
        self.device = device
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = (torch.tanh(self.mean_layer(s)) + 1) / 2 * (self.action_high - self.action_low) + self.action_low
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Actor_Beta(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_low, action_high, device):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha_layer = nn.Linear(hidden_dim, action_dim)
        self.beta_layer = nn.Linear(hidden_dim, action_dim)
        self.activate_func = nn.Tanh()  # Trick10: use tanh
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.alpha_layer, gain=0.01)
        orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.activate_func = nn.Tanh()  # use tanh insted of relu
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class IPPO:
    def __init__(self, state_dim, hidden_dim, action_dim, action_low, action_high,
                 actor_lr, critic_lr, lmbda, eps, gamma, epochs, num_episodes, device,
                 policy_dist="Beta", entropy_coef=0.01):
        self.policy_dist = policy_dist
        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(state_dim, hidden_dim, action_dim, action_low, action_high, device).to(device)
        else:
            self.actor = Actor_Gaussian(state_dim, hidden_dim, action_dim, action_low, action_high, device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic = Critic(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.epochs = epochs  # 更新次数
        self.num_episodes = num_episodes  # 总轮次
        self.device = device
        self.entropy_coef = entropy_coef  # 熵策略项
        self.action_low = action_low
        self.action_high = action_high

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device)  # shape (1, state_dim)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # Sample the action according to the probability distribution
                a_logprob = dist.log_prob(a).sum(dim=-1, keepdim=True)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()  # sampled action (1, action_dim)
                a = torch.clamp(a, torch.tensor(self.action_low, device=a.device),
                                torch.tensor(self.action_high, device=a.device))
                a_logprob = dist.log_prob(a).sum(dim=-1, keepdim=True)  # keepdim for PPO
        return a.squeeze(0).cpu().numpy(), a_logprob.squeeze(0).cpu().numpy()  # returns (action_vec,), (1,) or (,) depending

    def update(self, transition_dict, step=None, writer=None, agent_name="Agent"):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        old_log_probs = torch.tensor(transition_dict['old_log_probs'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        real_dones = torch.tensor(transition_dict['real_dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # ---------------- 计算TD目标和优势 ----------------
        adv = []
        gae = 0.0
        with torch.no_grad():
            vs = self.critic(states)
            vs_ = self.critic(next_states)
            # 计算 TD-error (td_delta)
            td_target = rewards + self.gamma * vs_ * (1 - real_dones)  # real_dones 始终为 False 因为都是人为截断
            td_delta = td_target - vs

            # 为了循环，转为 numpy
            td_delta = td_delta.cpu().detach().numpy()
            dones = dones.cpu().detach().numpy()

            # 从后向前计算 GAE
            for delta, d in zip(reversed(td_delta), reversed(dones)):
                # 当一个回合结束时 (d=True)，重置 gae 的传播
                gae = delta + self.gamma * self.lmbda * gae * (1.0 - d)
                adv.insert(0, gae)
            # 将 adv 和 v_target 转为 Tensor
            adv = torch.tensor(adv, dtype=torch.float, device=self.device).view(-1, 1)
            v_target = adv + self.critic(states)

            # # --- 加入随机噪声 (效果不好就删了) ---
            # alpha = 0.05  # 噪声权重
            # noise = torch.randn_like(advantage).to(self.device)  # shape 与 advantage 一致
            # advantage = (1 - alpha) * advantage + alpha * noise

            # # ++++++++++  优势函数归一化  ++++++++++
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)
            # # +++++++++++++++++++++++++++++++++++

        # tensorboard 中可视化
        actor_losses = []
        critic_losses = []

        # ++++++++++ 对同一批数据进行多轮优化 ++++++++++
        for epoch in range(self.epochs):
            # ---------------- 连续动作 log_prob (使用当前策略) ----------------
            dist_now = self.actor.get_dist(states)
            log_probs = dist_now.log_prob(actions).sum(dim=1, keepdim=True)
            # ---------------- PPO 损失 ----------------
            # 在第二轮及以后, log_probs 和 old_log_probs 将不同
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv

            # # ++++++++++ 加上策略熵 ++++++++++
            # dist_entropy = dist_now.entropy().sum(dim=1, keepdim=True)  # 多维动作空间求和
            # actor_loss = torch.mean(-torch.min(surr1, surr2) - self.entropy_coef * dist_entropy)
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            critic_loss = torch.mean(F.mse_loss(self.critic(states), v_target.detach()))

            # ---------------- 梯度更新 ----------------
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            # ++++++++++ 梯度裁剪 ++++++++++
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.lr_decay(step)

        # --------- TensorBoard 记录指标 ---------
        if writer is not None and step is not None:
            writer.add_scalar(f"{agent_name}/Actor_Loss", np.mean(actor_losses), step)
            writer.add_scalar(f"{agent_name}/Critic_Loss", np.mean(critic_losses), step)
            kl = (old_log_probs - log_probs).mean().item()
            writer.add_scalar(f"{agent_name}/KL_Divergence", kl, step)
            entropy = dist_now.entropy().mean().item()
            writer.add_scalar(f"{agent_name}/Policy_Entropy", entropy, step)

    def lr_decay(self, total_steps):
        lr_a_now = self.actor_optimizer.defaults['lr'] * (1 - total_steps / self.num_episodes)
        lr_a_now = max(lr_a_now, 1e-6)  # 防止学习率为负
        lr_c_now = self.critic_optimizer.defaults['lr'] * (1 - total_steps / self.num_episodes)
        lr_c_now = max(lr_c_now, 1e-6)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_c_now
