
import os
import random
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import warnings
from my_env_beta_distribution import MyEnv
from utils.normalization import Normalization, RewardScaling
from utils import rl_utils
from MATD3_agent import MATD3
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    # ========= 保证仿真可复现 ========= #
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ========= 日志与设备 ========= #
    today_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/MATD3/{today_str}_MATD3_experiment"  # 改为MATD3
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========= 超参数 ========= #
    num_episodes = 2000
    total_time_slots = 50
    buffer_size = int(1e6)
    hidden_dim = 128
    actor_lr = 3e-4
    critic_lr = 3e-4
    gamma = 0.95
    tau = 1e-3
    batch_size = 128
    minimal_size = 2048
    policy_freq = 2  # TD3: 延迟策略更新频率

    # ========= 环境参数 ============ #
    K = 10
    noma_group_num = 3
    uav_num = noma_group_num
    users_num_per_noma_group = 2
    num_users = noma_group_num * users_num_per_noma_group
    center = (0, 200)
    radius = 60
    ris_pos = [0, 0, 20]
    # 初始化用户与 UAV 位置
    angles_users = np.random.uniform(0, 2 * np.pi, num_users)
    r_users = radius * np.sqrt(np.random.uniform(0, 1, num_users))
    users_x = center[0] + r_users * np.cos(angles_users)
    users_y = center[1] + r_users * np.sin(angles_users)
    users_z = np.zeros(num_users)
    users_pos = np.stack([users_x, users_y, users_z], axis=1)
    angles_uavs = np.random.uniform(0, 2 * np.pi, uav_num)
    r_uavs = radius * np.sqrt(np.random.uniform(0, 1, uav_num))
    uavs_x = center[0] + r_uavs * np.cos(angles_uavs)
    uavs_y = center[1] + r_uavs * np.sin(angles_uavs)
    uavs_z = np.full(uav_num, 50.0)
    uavs_pos = np.stack([uavs_x, uavs_y, uavs_z], axis=1)

    env = MyEnv(K=K, noma_group_num=noma_group_num, uav_num=uav_num,
                users_num_per_noma_group=users_num_per_noma_group,
                uav_pos=uavs_pos, ris_pos=ris_pos, users_pos=users_pos,
                users_center=center, users_radius=radius,
                total_time_slots=total_time_slots, seed=seed)
    agent_num = uav_num + 1

    # 拼接所有 Agent 的观测空间
    state_dims = []
    for _ in range(uav_num):
        state_dims.append(env.observation_space["uav"]["uav_0"].shape[0])
    state_dims.append(env.observation_space["ris"].shape[0])
    # 拼接所有 Agent 的动作
    action_dims = []
    action_dim_uav = env.action_space["uav"]["uav_0"].shape[0]
    action_dim_ris = env.action_space["ris"].shape[0]
    for _ in range(uav_num):
        action_dims.append(action_dim_uav)
    action_dims.append(action_dim_ris)
    # 拼接所有 Agent 的最大动作和最小动作
    action_lows = []
    action_highs = []
    action_uav_low = env.action_space["uav"]["uav_0"].low
    action_uav_high = env.action_space["uav"]["uav_0"].high
    for _ in range(uav_num):
        action_lows.append(action_uav_low)
        action_highs.append(action_uav_high)
    action_ris_low = env.action_space["ris"].low
    action_ris_high = env.action_space["ris"].high
    action_lows.append(action_ris_low)
    action_highs.append(action_ris_high)

    # critic 输入是所有 Agent 的观测 + 动作
    critic_input_dim = sum(state_dims) + sum(action_dims)

    # 创建MATD3实例，添加policy_freq参数
    matd3 = MATD3(env, agent_num, device, actor_lr, critic_lr, hidden_dim,
                  state_dims, action_dims, action_highs, action_lows, 
                  critic_input_dim, gamma, tau, policy_freq=policy_freq)

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)

    # ========= 状态归一化 & 奖励缩放 ========= #
    running_norms = [Normalization(s_dim) for s_dim in state_dims]
    # reward_scalers = [RewardScaling(shape=1, gamma=gamma) for _ in range(agent_num)]

    # ========= 训练主循环 ========= #
    total_step = 0
    reward_res = []
    best_mean_reward = -np.inf

    with tqdm(total=num_episodes, desc="Training Progress") as pbar:
        for i_episode in range(num_episodes):
            s = env.reset()
            s = flatten_state_dict(s)
            # 归一化状态
            s_norm = [running_norms[i](np.array(s[i])) for i in range(agent_num)]
            episode_rewards = np.zeros(agent_num)
            total_rewards = []

            # for rs in reward_scalers:
            #     rs.reset()

            for t in range(total_time_slots):
                if replay_buffer.size() < minimal_size:
                    # 随机选择动作 [-1, 1] 范围
                    actions = [np.random.uniform(-1, 1, size=action_dims[i]) for i in range(agent_num)]
                else:
                    # 使用策略网络输出动作
                    actions = matd3.take_action(s_norm, explore=True)
                # 转换为 env.step 所需字典形式 [0, 1]
                actions_dict = convert_action_list_to_dict(actions, uav_num)
                next_s, total_reward, r, done = env.step(actions_dict, i_episode)
                next_s = flatten_state_dict(next_s)
                next_s_norm = [running_norms[i](np.array(next_s[i])) for i in range(agent_num)]
                r = flatten_reward_dict(r)
                # reward scaling
                # r_scaled = [reward_scalers[i](r[i]) for i in range(agent_num)]
                replay_buffer.add(s_norm, actions, r, next_s_norm, done)
                s_norm = next_s_norm
                episode_rewards += np.array(r)
                total_rewards.append(total_reward)

                total_step += 1
                # 更新参数
                if replay_buffer.size() >= minimal_size:
                    sample = replay_buffer.sample(batch_size)
                    # 返回的 sample 都是 list 类型
                    def stack_array(x):
                        """
                        将采样到的 batch 动作或状态整理为 (batch, agent_dim) 数组
                        x: list of dict 或 list of list
                        """
                        rearranged = [[sub_x[i] for sub_x in x]
                                      for i in range(len(x[0]))]
                        return [torch.FloatTensor(np.vstack(aa)).to(device)
                                for aa in rearranged]

                    state_list, action_list, reward_list, next_state_list, done_list = sample

                    proc_state = stack_array(state_list)
                    proc_action = stack_array(action_list)
                    proc_reward = stack_array(reward_list)
                    proc_next_state = stack_array(next_state_list)

                    # 处理done：done_list 是长度 batch_size 的 list，每项是 int/float
                    done_arr = np.array(done_list).reshape(-1, 1)  # shape (batch,1)
                    proc_done = torch.FloatTensor(done_arr).to(device)

                    # 组合成与原来代码期望的 sample 结构：
                    # sample_processed 的前 4 项是 list(按 agent) of tensors；最后一项是 done tensor
                    sample_processed = [proc_state, proc_action, proc_reward, proc_next_state, proc_done]

                    # 替代原本的 sample = [stack_array(x) for x in sample]
                    sample = sample_processed
                    for a_i in range(agent_num):
                        matd3.update(sample, a_i)
                    matd3.update_all_targets()

                if done:
                    break

            avg_reward = np.mean(total_rewards)
            reward_res.append(avg_reward)

            # TensorBoard 日志
            writer.add_scalar("Reward/episode", avg_reward, i_episode)
            for a_i in range(agent_num):
                writer.add_scalar(f"Reward/agent_{a_i}", episode_rewards[a_i] / total_time_slots, i_episode)

            # tqdm 状态栏更新
            pbar.set_postfix({"avg_reward": f"{avg_reward:.3f}"})
            pbar.update(1)

            # 保存最佳表现
            if avg_reward > best_mean_reward:
                best_mean_reward = avg_reward

    writer.close()

    # ========= 绘制与保存结果 ========= #
    plt.plot(np.arange(len(reward_res)), reward_res)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("MATD3 Training Performance")  # 改为MATD3
    plt.show()

    df = pd.DataFrame({
        "episode": np.arange(len(reward_res)),
        "reward": reward_res
    })
    filename = f"{today_str}_MATD3_training_rewards_seed{seed}.csv"  # 改为MATD3
    df.to_csv(filename, index=False)
    print(f"✅ 奖励结果已保存到 {filename}")


def flatten_state_dict(s):
    """
    把环境返回的 dict 状态转成数组列表 [uav_0, uav_1, ..., ris]
    """
    flat_list = []
    # UAV 部分
    for key in sorted(s["uav"].keys(), key=lambda x: int(x.split("_")[1])):
        flat_list.append(s["uav"][key])
    # RIS
    flat_list.append(s["ris"])
    return flat_list


def flatten_reward_dict(r):
    """把奖励字典转成列表"""
    r_uav = [np.mean(v) if isinstance(v, (list, np.ndarray)) else v for v in r["uav"]]
    return r_uav + [r["ris"] if not isinstance(r["ris"], (list, np.ndarray)) else np.mean(r["ris"])]


def convert_action_list_to_dict(actions_list, uav_num):
    """
    将 matd3 输出的列表动作转换为 env.step 所需字典格式，
    并归一化到 [0, 1]。
    """
    uav_actions = {}
    for i in range(uav_num):
        # 转 np.array 后归一化到 0~1
        uav_actions[f"uav_{i}"] = ((np.array(actions_list[i]) + 1) / 2).tolist()

    # RIS 动作
    ris_action = ((np.array(actions_list[uav_num]) + 1) / 2).tolist()

    return {"uav": uav_actions, "ris": ris_action}


if __name__ == "__main__":
    main()