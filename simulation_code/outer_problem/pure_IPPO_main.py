import os
import random
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from my_env_pure_learning import MyEnv
import warnings
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils.normalization import Normalization, RewardScaling
from IPPO_agent import IPPO

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":
    # ========= 保证仿真可复现 ========= #
    seed = 42
    # Python 内置随机
    random.seed(seed)
    # NumPy 随机
    np.random.seed(seed)
    # PyTorch 随机
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # 确保 PyTorch 的卷积操作确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ========= 保证仿真可复现 ========= #

    # ========= 可视化训练 ========= #
    today_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/IPPO/{today_str}_IPPO_experiment"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    # ========= 可视化训练 ========= #

    # ========= PPO 参数 ============ #
    actor_lr = 3e-4
    critic_lr = 3e-4  # 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.99
    lmbda = 0.95
    eps = 0.2  # 裁剪因子
    epochs = 10  # 每次更新迭代10次
    # ========= PPO 参数 ============ #

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    # ========= 环境参数 ============ #
    K = 10
    noma_group_num = 3
    uav_num = noma_group_num
    users_num_per_noma_group = 2
    num_users = noma_group_num * users_num_per_noma_group
    center = (0, 200)  # 圆心
    radius = 60  # 半径60m
    ris_pos = [0, 0, 20]  # RIS 位置
    # ---------------- 用户位置 ----------------
    angles_users = np.random.uniform(0, 2 * np.pi, num_users)
    r_users = radius * np.sqrt(np.random.uniform(0, 1, num_users))
    users_x = center[0] + r_users * np.cos(angles_users)
    users_y = center[1] + r_users * np.sin(angles_users)
    users_z = np.zeros(num_users)
    users_pos = np.stack([users_x, users_y, users_z], axis=1)  # shape (num_users, 3)
    # ---------------- UAV位置 ----------------
    angles_uavs = np.random.uniform(0, 2 * np.pi, uav_num)
    r_uavs = radius * np.sqrt(np.random.uniform(0, 1, uav_num))
    uavs_x = center[0] + r_uavs * np.cos(angles_uavs)
    uavs_y = center[1] + r_uavs * np.sin(angles_uavs)
    uavs_z = np.full(uav_num, 50.0)  # UAV 高度固定 50m
    uavs_pos = np.stack([uavs_x, uavs_y, uavs_z], axis=1)  # shape (num_uavs, 3)
    # ---------------- 仿真时隙 ----------------
    total_time_slots = 50

    # ========= 环境参数 ============ #
    env = MyEnv(K=K, noma_group_num=noma_group_num, uav_num=uav_num, users_num_per_noma_group=users_num_per_noma_group,
                uav_pos=uavs_pos, ris_pos=ris_pos, users_pos=users_pos,
                users_center=center, users_radius=radius,
                total_time_slots=total_time_slots, seed=seed)  # 创建环境

    state_dim_uav = env.observation_space["uav"]["uav_0"].shape[0]
    state_dim_ris = env.observation_space["ris"].shape[0]

    action_dim_uav = env.action_space["uav"]["uav_0"].shape[0]
    action_dim_ris = env.action_space["ris"].shape[0]

    action_uav_low = env.action_space["uav"]["uav_0"].low
    action_uav_high = env.action_space["uav"]["uav_0"].high

    action_ris_low = env.action_space["ris"].low
    action_ris_high = env.action_space["ris"].high

    # UAV 和 RIS Agent
    agents_uav = {}
    running_norms_uav = {}
    for i in range(uav_num):
        agent_id = f"uav_{i}"
        agents_uav[agent_id] = IPPO(state_dim_uav, hidden_dim, action_dim_uav, action_uav_low, action_uav_high,
                                    actor_lr, critic_lr, lmbda, eps, gamma, epochs, num_episodes, device)
        running_norms_uav[agent_id] = Normalization(state_dim_uav)  # 为每个 agent 各建一个动态归一化状态空间

    agent_ris = IPPO(state_dim_ris, hidden_dim, action_dim_ris, action_ris_low, action_ris_high,
                     actor_lr, critic_lr, lmbda, eps, gamma, epochs, num_episodes, device)
    running_norm_ris = Normalization(state_dim_ris)  # 为每个 agent 各建一个动态归一化状态空间

    reward_scalers_uav = [RewardScaling(shape=1, gamma=gamma) for _ in range(uav_num)]
    reward_scaler_ris = RewardScaling(shape=1, gamma=gamma)

    reward_res = []
    all_agents_rewards = []  # 每个agent的奖励记录：[[uav_0, uav_1, ..., ris], ...]
    max_avg_reward = -np.inf  # 用于记录最大的平均奖励
    best_uav_trajectory = None
    user_trajectory = None

    with tqdm(total=int(num_episodes), desc='Training Progress') as pbar:
        for i_episode in range(int(num_episodes)):
            # --- 每个episode存储各UAV奖励 ---
            episode_rewards_total = []  # 总奖励（环境返回的 total_reward）
            episode_rewards_uav = np.zeros(uav_num)  # 每个UAV奖励累计
            episode_reward_ris = 0.0  # RIS 奖励累计

            # --- 为每个 UAV agent 创建独立的 transition_dict ---
            transition_dicts_uav = {
                f"uav_{j}": {'states': [],
                             'actions': [],
                             'next_states': [],
                             'rewards': [],
                             'old_log_probs': [],
                             'dones': [],
                             'real_dones': []}
                for j in range(uav_num)}
            # --- ADDED END ---
            transition_dict_ris = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'old_log_probs': [],
                'dones': [],
                'real_dones': []}

            s = env.reset()
            terminal = False
            uav_positions_episode = []  # 初始化当前 episode 的 UAV 轨迹存储
            user_positions_episode = []

            # reset reward scaling
            for rs in reward_scalers_uav:
                rs.reset()
            reward_scaler_ris.reset()

            while not terminal:
                # --- 为每个 UAV agent 独立处理状态、动作 ---
                actions_uav_dict = {}
                old_log_probs_uav_dict = {}
                states_uav_norm_dict = {}  # 临时存储归一化后的状态
                uav_positions_episode.append(env.getPosUAV())  # --- 存储当前时隙 UAV 坐标 ---
                user_positions_episode.append(env.getPosUser())  # --- 存储当前时隙用户坐标 ---

                # --- UAV agents 选择动作 ---
                for uav_i in range(uav_num):
                    agent_id = f"uav_{uav_i}"
                    # 更新并归一化该 UAV 的状态
                    s_uav_norm = running_norms_uav[agent_id](np.array(s["uav"][agent_id]))
                    states_uav_norm_dict[agent_id] = s_uav_norm

                    # 该 UAV agent 采取动作
                    a_uav, old_log_probs_uav = agents_uav[agent_id].choose_action(s_uav_norm)
                    # 存储 UAV
                    actions_uav_dict[agent_id] = a_uav
                    old_log_probs_uav_dict[agent_id] = old_log_probs_uav

                # --- RIS agent 选择动作 ---
                s_ris_norm = running_norm_ris(np.array(s["ris"]))
                a_ris, old_log_probs_ris = agent_ris.choose_action(s_ris_norm)
                # --- 环境执行 ---
                next_s, total_reward, r_dict, done = env.step({"uav": actions_uav_dict, "ris": a_ris}, i_episode)
                # =====================================================
                #  r 是 dict： r["uav"], r["ris"]
                # =====================================================
                # 解析奖励
                r_uav_list = [np.mean(r_val) if len(r_val) > 0 else 0.0 for r_val in r_dict["uav"]]
                r_ris = np.mean(r_dict["ris"]) if len(r_dict["ris"]) > 0 else 0.0

                # 奖励归一化
                r_uav_norm = [reward_scalers_uav[j](r_uav_list[j]) for j in range(uav_num)]
                r_ris_norm = reward_scaler_ris(r_ris)

                # 累计各agent奖励
                episode_rewards_total.append(total_reward)
                episode_rewards_uav += np.array(r_uav_list)
                episode_reward_ris += r_ris

                # --- 为每个 UAV agent 存储经验 ---
                for uav_i in range(uav_num):
                    agent_id = f"uav_{uav_i}"
                    # 对 next_state 同样进行 normalize
                    next_s_uav_norm = running_norms_uav[agent_id](next_s["uav"][agent_id])

                    # 存储已归一化的数据到对应的字典
                    transition_dicts_uav[agent_id]['states'].append(states_uav_norm_dict[agent_id])
                    transition_dicts_uav[agent_id]['actions'].append(actions_uav_dict[agent_id])
                    transition_dicts_uav[agent_id]['next_states'].append(next_s_uav_norm)
                    transition_dicts_uav[agent_id]['rewards'].append(r_uav_norm[uav_i])  # 共享奖励
                    transition_dicts_uav[agent_id]['old_log_probs'].append(old_log_probs_uav_dict[agent_id])
                    transition_dicts_uav[agent_id]['dones'].append(bool(done))
                    transition_dicts_uav[agent_id]['real_dones'].append(False)

                # --- 为 RIS agent 存储经验 ---
                next_s_ris_norm = running_norm_ris(next_s["ris"])
                # 存储已归一化的数据
                transition_dict_ris['states'].append(s_ris_norm)
                transition_dict_ris['actions'].append(a_ris)
                transition_dict_ris['next_states'].append(next_s_ris_norm)
                transition_dict_ris['rewards'].append(r_ris_norm)
                transition_dict_ris['old_log_probs'].append(old_log_probs_ris)
                transition_dict_ris['dones'].append(bool(done))
                transition_dict_ris['real_dones'].append(False)

                s = next_s
                terminal = done

            if np.mean(episode_rewards_total) > max_avg_reward:
                max_avg_reward = np.mean(episode_rewards_total)
                best_uav_trajectory = np.array(uav_positions_episode)  # shape: (50, uav_num, 3)
                user_trajectory = np.array(user_positions_episode)

            # ---  更新所有 UAV Agents ---
            for uav_i in range(uav_num):
                agent_id = f"uav_{uav_i}"
                agents_uav[agent_id].update(transition_dicts_uav[agent_id], i_episode, writer,
                                            agent_name=f"UAV_{uav_i}")
            # RIS Agent 更新
            agent_ris.update(transition_dict_ris, i_episode, writer, agent_name="RIS")

            # 统计奖励
            avg_total_reward = np.mean(episode_rewards_total)
            avg_uav_rewards = episode_rewards_uav / len(episode_rewards_total)
            avg_ris_reward = episode_reward_ris / len(episode_rewards_total)

            reward_res.append(np.mean(episode_rewards_total))  # 统计平均奖励
            all_agents_rewards.append(np.concatenate([avg_uav_rewards, [avg_ris_reward]]))

            # Average Reward 写入 TensorBoard
            writer.add_scalar("Reward/episode", avg_total_reward, i_episode)
            for uav_i in range(uav_num):
                writer.add_scalar(f"Reward/UAV_{uav_i}", avg_uav_rewards[uav_i], i_episode)
            writer.add_scalar("Reward/RIS", avg_ris_reward, i_episode)

            # --- 每轮 episode 结束后，立即更新 tqdm 显示平均奖励 ---
            pbar.set_postfix({
                'avg_reward': f'{np.mean(episode_rewards_total):.3f}'
            })
            pbar.update(1)
    # 关闭 TensorBoard writer
    writer.close()

    # ========= 保存总平均奖励 ========= #
    reward_array = np.array(reward_res)
    episodes_list = np.arange(reward_array.shape[0])
    plt.plot(episodes_list, reward_array)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('IPPO training performance')
    plt.show()

    # =========== 保存 reward csv 文件 ===========
    # 构造文件名
    filename = f'{today_str}_Pure_IPPO_training_rewards_{noma_group_num}_seed_{seed}.csv'
    # 保存 reward_array 和 episodes_list 到 CSV
    df = pd.DataFrame({
        'episode': episodes_list,
        'reward': reward_array
    })
    df.to_csv(filename, index=False)
    print(f"训练奖励已保存到 {filename} 中.csv")
    # =========== 保存 reward csv 文件 ===========

    # # ========= 保存每个Agent奖励 ========= #
    # all_agents_rewards = np.array(all_agents_rewards)
    # columns = [f"UAV_{i}_reward" for i in range(uav_num)] + ["RIS_reward"]
    # df_agents = pd.DataFrame(all_agents_rewards, columns=columns)
    # df_agents.insert(0, "episode", np.arange(num_episodes))
    # filename_agents = f'{today_str}_IPPO_all_agents_rewards_seed_{seed}.csv'
    # df_agents.to_csv(filename_agents, index=False)
    # print(f"✅ 所有 Agent 的奖励已保存到 {filename_agents}")
    #
    # # =================== 保存最佳 UAV 轨迹到 CSV ===================
    # if best_uav_trajectory is not None:
    #     # ===== 保存 UAV 轨迹 =====
    #     uav_traj_list = []
    #     for t in range(best_uav_trajectory.shape[0]):
    #         for uav_i in range(best_uav_trajectory.shape[1]):
    #             x, y, z = best_uav_trajectory[t, uav_i]
    #             uav_traj_list.append([t, uav_i, x, y, z])
    #     df_uav = pd.DataFrame(uav_traj_list, columns=['time_slot', 'uav_id', 'x', 'y', 'z'])
    #     df_uav.to_csv(f'{today_str}_SEED{seed}_best_uav_trajectory.csv', index=False)
    #     print(f"✅ 最佳 UAV 轨迹已保存：{today_str}_best_uav_trajectory.csv")
    #
    #     # ===== 保存 User 轨迹 =====
    #     user_traj_list = []
    #     for t in range(user_trajectory.shape[0]):
    #         for user_i in range(user_trajectory.shape[1]):
    #             x, y, z = user_trajectory[t, user_i]
    #             user_traj_list.append([t, user_i, x, y, z])
    #     df_user = pd.DataFrame(user_traj_list, columns=['time_slot', 'user_id', 'x', 'y', 'z'])
    #     df_user.to_csv(f'SEED{seed}_user_trajectory.csv', index=False)
    #     print(f"✅ 用户轨迹已保存：user_trajectory.csv")

    # =================== 绘制最佳 UAV 轨迹 ===================
    plt.figure(figsize=(8, 6))
    for uav_i in range(best_uav_trajectory.shape[1]):
        traj = best_uav_trajectory[:, uav_i, :]  # shape: (time_slots, 3)
        plt.plot(traj[:, 0], traj[:, 1], marker='o', label=f'UAV {uav_i}')

    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('Best UAV trajectories over 50 time slots')
    plt.legend()
    plt.grid(True)
    plt.show()