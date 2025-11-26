import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from utils import rl_utils


class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DDPG:
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim,
                                       hidden_dim).to(device)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic = TwoLayerFC(critic_input_dim, 1,
                                        hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        # 添加探索噪声参数（衰减策略）
        self.noise_scale = 0.2  # 初始噪声较大
        self.noise_decay = 0.9995
        self.min_noise = 0.1
        self.device = device

    def take_action(self, state, action_low, action_high, explore):
        action = torch.tanh(self.actor(state))  # 连续动作范围 [-1, 1]
        if explore:
            noise = torch.randn_like(action) * self.noise_scale
            action = torch.clamp(action + noise, -1, 1)
            # 噪声衰减
            self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class MADDPG:
    def __init__(self, env, agent_num, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, action_lows, action_highs, critic_input_dim, gamma, tau):
        self.agents = []

        for i in range(agent_num):
            self.agents.append(
                DDPG(state_dims[i], action_dims[i], critic_input_dim,
                     hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.action_lows = action_lows
        self.action_highs = action_highs

    @property
    def policies(self):
        return [agt.actor for agt in self.agents]

    @property
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]

    def take_action(self, states, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(len(self.agents))
        ]
        return [
            agent.take_action(
                state,
                torch.tensor(self.action_lows[i], dtype=torch.float, device=self.device),
                torch.tensor(self.action_highs[i], dtype=torch.float, device=self.device),
                explore)
            for i, agent, state in zip(range(len(self.agents)), self.agents, states)
        ]

    def update(self, sample, i_agent):
        obs, act, rew, next_obs, done = sample
        cur_agent = self.agents[i_agent]

        # ===== Critic 更新 =====
        cur_agent.critic_optimizer.zero_grad()
        with torch.no_grad():
            all_target_act = [
                torch.tanh(pi(_next_obs)) for pi, _next_obs in zip(self.target_policies, next_obs)
            ]
            target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
            target_critic_value = rew[i_agent].view(-1, 1) + \
                                  self.gamma * cur_agent.target_critic(target_critic_input) * (
                                              1 - done[i_agent].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(cur_agent.critic.parameters(), max_norm=10)
        cur_agent.critic_optimizer.step()

        # ===== Actor 更新 =====
        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = torch.tanh(cur_agent.actor(obs[i_agent]))
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_actor_out)
            else:
                all_actor_acs.append(torch.tanh(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out ** 2).mean() * 1e-3  # 正则项
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(cur_agent.actor.parameters(), max_norm=10)
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)