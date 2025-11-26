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


class TD3:
    def __init__(self, state_dim, action_dim, critic_input_dim, hidden_dim,
                 actor_lr, critic_lr, device):
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim).to(device)
        self.target_actor = TwoLayerFC(state_dim, action_dim,
                                       hidden_dim).to(device)

        # TD3: 双Critic网络
        self.critic_1 = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.critic_2 = TwoLayerFC(critic_input_dim, 1, hidden_dim).to(device)
        self.target_critic_1 = TwoLayerFC(critic_input_dim, 1,
                                          hidden_dim).to(device)
        self.target_critic_2 = TwoLayerFC(critic_input_dim, 1,
                                          hidden_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)

        # DDPG: 探索噪声参数
        self.noise_scale = 0.1

        # TD3: 目标策略平滑噪声
        self.target_noise = 0.1
        self.target_noise_clip = 0.2

        self.device = device

    def take_action(self, state, action_low, action_high, explore):
        action = torch.tanh(self.actor(state))  # 连续动作范围 [-1, 1]
        if explore:
            noise = torch.randn_like(action) * self.noise_scale
            action = torch.clamp(action + noise, -1, 1)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class MATD3:
    def __init__(self, env, agent_num, device, actor_lr, critic_lr, hidden_dim,
                 state_dims, action_dims, action_lows, action_highs, critic_input_dim,
                 gamma, tau, policy_freq=2):
        self.agents = []

        for i in range(agent_num):
            self.agents.append(
                TD3(state_dims[i], action_dims[i], critic_input_dim,
                    hidden_dim, actor_lr, critic_lr, device))
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq  # TD3: 延迟策略更新频率
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        self.action_lows = action_lows
        self.action_highs = action_highs

        # TD3: 记录更新次数
        self.update_count = 0

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

        # ===== Critic 更新（双Q网络） =====
        cur_agent.critic_1_optimizer.zero_grad()
        cur_agent.critic_2_optimizer.zero_grad()

        with torch.no_grad():
            # TD3: 目标策略平滑
            all_target_act = []
            for pi, _next_obs in zip(self.target_policies, next_obs):
                target_action = torch.tanh(pi(_next_obs))
                # 添加裁剪噪声
                noise = torch.randn_like(target_action) * cur_agent.target_noise
                noise = torch.clamp(noise, -cur_agent.target_noise_clip, cur_agent.target_noise_clip)
                target_action = torch.clamp(target_action + noise, -1, 1)
                all_target_act.append(target_action)

            target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)

            # TD3: 取两个目标Q值的最小值
            target_q1 = cur_agent.target_critic_1(target_critic_input)
            target_q2 = cur_agent.target_critic_2(target_critic_input)
            target_q = torch.min(target_q1, target_q2)

            target_value = rew[i_agent].view(-1, 1) + \
                           self.gamma * target_q * (1 - done[i_agent].view(-1, 1))

        critic_input = torch.cat((*obs, *act), dim=1)

        # 更新两个Critic
        q1_value = cur_agent.critic_1(critic_input)
        q2_value = cur_agent.critic_2(critic_input)

        critic_1_loss = self.critic_criterion(q1_value, target_value.detach())
        critic_2_loss = self.critic_criterion(q2_value, target_value.detach())

        critic_1_loss.backward()
        critic_2_loss.backward()

        torch.nn.utils.clip_grad_norm_(cur_agent.critic_1.parameters(), max_norm=10)
        torch.nn.utils.clip_grad_norm_(cur_agent.critic_2.parameters(), max_norm=10)

        cur_agent.critic_1_optimizer.step()
        cur_agent.critic_2_optimizer.step()

        # ===== Actor 更新（延迟更新） =====
        # TD3: 每policy_freq次才更新一次策略
        if self.update_count % self.policy_freq == 0:
            cur_agent.actor_optimizer.zero_grad()
            cur_actor_out = torch.tanh(cur_agent.actor(obs[i_agent]))
            all_actor_acs = []
            for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
                if i == i_agent:
                    all_actor_acs.append(cur_actor_out)
                else:
                    all_actor_acs.append(torch.tanh(pi(_obs)))
            vf_in = torch.cat((*obs, *all_actor_acs), dim=1)

            # TD3: 只使用critic_1来更新策略
            actor_loss = -cur_agent.critic_1(vf_in).mean()
            actor_loss += (cur_actor_out ** 2).mean() * 1e-3  # 正则项

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(cur_agent.actor.parameters(), max_norm=10)
            cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        # TD3: 每policy_freq次才更新一次目标网络
        if self.update_count % self.policy_freq == 0:
            for agt in self.agents:
                agt.soft_update(agt.actor, agt.target_actor, self.tau)
                agt.soft_update(agt.critic_1, agt.target_critic_1, self.tau)
                agt.soft_update(agt.critic_2, agt.target_critic_2, self.tau)

        self.update_count += 1
