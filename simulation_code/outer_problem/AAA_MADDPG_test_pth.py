import os
import random
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from my_env_beta_distribution import MyEnv
from utils.normalization import Normalization
from MADDPG_agent import MADDPG
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_trained_models(maddpg, model_dir, device, uav_num):
    """
    加载训练好的MADDPG模型权重

    Args:
        maddpg: MADDPG实例
        model_dir: 模型文件所在目录
        device: 设备 (cpu/cuda)
        uav_num: UAV数量
    """
    print("\n正在加载训练好的模型...")

    # 加载 UAV agents
    for uav_idx in range(uav_num):
        actor_path = os.path.join(model_dir, f'uav_{uav_idx}_actor.pth')
        critic_path = os.path.join(model_dir, f'uav_{uav_idx}_critic.pth')

        if os.path.exists(actor_path):
            maddpg.agents[uav_idx].actor.load_state_dict(
                torch.load(actor_path, map_location=device)
            )
            print(f"✅ 成功加载 uav_{uav_idx} Actor 模型")
        else:
            print(f"❌ 未找到 uav_{uav_idx} Actor 模型文件: {actor_path}")

        if os.path.exists(critic_path):
            maddpg.agents[uav_idx].critic.load_state_dict(
                torch.load(critic_path, map_location=device)
            )
            print(f"✅ 成功加载 uav_{uav_idx} Critic 模型")
        else:
            print(f"❌ 未找到 uav_{uav_idx} Critic 模型文件")

    # 加载 RIS agent
    ris_actor_path = os.path.join(model_dir, 'RIS_actor.pth')
    ris_critic_path = os.path.join(model_dir, 'RIS_critic.pth')

    if os.path.exists(ris_actor_path):
        maddpg.agents[uav_num].actor.load_state_dict(
            torch.load(ris_actor_path, map_location=device)
        )
        print(f"✅ 成功加载 RIS Actor 模型")
    else:
        print(f"❌ 未找到 RIS Actor 模型文件: {ris_actor_path}")

    if os.path.exists(ris_critic_path):
        maddpg.agents[uav_num].critic.load_state_dict(
            torch.load(ris_critic_path, map_location=device)
        )
        print(f"✅ 成功加载 RIS Critic 模型")
    else:
        print(f"❌ 未找到 RIS Critic 模型文件")


def test_episode(env, maddpg, running_norms, uav_num, total_time_slots):
    """
    执行一个测试回合

    Args:
        env: 环境
        maddpg: MADDPG实例
        running_norms: 状态归一化器列表
        uav_num: UAV数量
        total_time_slots: 总时隙数

    Returns:
        episode_rewards: 每个时隙的总奖励
        agent_rewards: 每个agent的累计奖励
        uav_trajectory: UAV轨迹
        user_trajectory: 用户轨迹
    """
    s = env.reset()
    s = flatten_state_dict(s)
    terminal = False

    episode_rewards = []
    agent_rewards = np.zeros(uav_num + 1)  # UAV agents + RIS agent
    uav_trajectory = []
    user_trajectory = []

    # 设置为评估模式
    for agent in maddpg.agents:
        agent.actor.eval()

    with torch.no_grad():  # 测试时不需要梯度
        for t in range(total_time_slots):
            # 记录当前位置
            uav_trajectory.append(env.getPosUAV())
            user_trajectory.append(env.getPosUser())

            # 归一化状态
            s_norm = [running_norms[i](np.array(s[i])) for i in range(len(s))]

            # 选择动作（不添加探索噪声）
            actions = maddpg.take_action(s_norm, explore=False)

            # 转换为环境所需的动作格式
            actions_dict = convert_action_list_to_dict(actions, uav_num)

            # 环境执行
            next_s, total_reward, r_dict, done = env.step(actions_dict, 0)

            # 记录奖励
            episode_rewards.append(total_reward)
            r = flatten_reward_dict(r_dict)
            agent_rewards += np.array(r)

            # 更新状态
            s = flatten_state_dict(next_s)
            terminal = done

            if done:
                break

    return episode_rewards, agent_rewards, np.array(uav_trajectory), np.array(user_trajectory)


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
    将 maddpg 输出的列表动作转换为 env.step 所需字典格式，
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
    # ========= 设置随机种子 ========= #
    seed = 1119
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ========= 设备配置 ========= #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ========= 模型路径配置 ========= #
    # 请修改为你的模型文件所在目录
    model_dir = "./checkpoint/MADDPG_models"  # 修改为你的模型保存路径

    # ========= MADDPG 参数（与训练时保持一致）========= #
    hidden_dim = 128
    actor_lr = 3e-4
    critic_lr = 3e-4
    gamma = 0.95
    tau = 1e-3

    # ========= 环境参数（与训练时保持一致）========= #
    K = 10
    noma_group_num = 3
    uav_num = noma_group_num
    users_num_per_noma_group = 2
    num_users = noma_group_num * users_num_per_noma_group
    center = (0, 200)
    radius = 60
    ris_pos = [0, 0, 20]
    total_time_slots = 10

    # 用户位置
    angles_users = np.random.uniform(0, 2 * np.pi, num_users)
    r_users = radius * np.sqrt(np.random.uniform(0, 1, num_users))
    users_x = center[0] + r_users * np.cos(angles_users)
    users_y = center[1] + r_users * np.sin(angles_users)
    users_z = np.zeros(num_users)
    users_pos = np.stack([users_x, users_y, users_z], axis=1)

    # UAV位置
    angles_uavs = np.random.uniform(0, 2 * np.pi, uav_num)
    r_uavs = radius * np.sqrt(np.random.uniform(0, 1, uav_num))
    uavs_x = center[0] + r_uavs * np.cos(angles_uavs)
    uavs_y = center[1] + r_uavs * np.sin(angles_uavs)
    uavs_z = np.full(uav_num, 50.0)
    uavs_pos = np.stack([uavs_x, uavs_y, uavs_z], axis=1)

    # ========= 创建环境 ========= #
    env = MyEnv(
        K=K,
        noma_group_num=noma_group_num,
        uav_num=uav_num,
        users_num_per_noma_group=users_num_per_noma_group,
        uav_pos=uavs_pos,
        ris_pos=ris_pos,
        users_pos=users_pos,
        users_center=center,
        users_radius=radius,
        total_time_slots=total_time_slots,
        seed=seed
    )

    agent_num = uav_num + 1

    # ========= 获取状态和动作维度 ========= #
    state_dims = []
    for _ in range(uav_num):
        state_dims.append(env.observation_space["uav"]["uav_0"].shape[0])
    state_dims.append(env.observation_space["ris"].shape[0])

    action_dims = []
    action_dim_uav = env.action_space["uav"]["uav_0"].shape[0]
    action_dim_ris = env.action_space["ris"].shape[0]
    for _ in range(uav_num):
        action_dims.append(action_dim_uav)
    action_dims.append(action_dim_ris)

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

    critic_input_dim = sum(state_dims) + sum(action_dims)

    # ========= 创建MADDPG实例 ========= #
    maddpg = MADDPG(
        env, agent_num, device, actor_lr, critic_lr, hidden_dim,
        state_dims, action_dims, action_highs, action_lows,
        critic_input_dim, gamma, tau
    )

    # ========= 创建状态归一化器 ========= #
    running_norms = [Normalization(s_dim) for s_dim in state_dims]

    # ========= 加载训练好的模型 ========= #
    load_trained_models(maddpg, model_dir, device, uav_num)

    # ========= 执行测试 ========= #
    num_test_episodes = 1  # 测试回合数
    print(f"\n开始测试，共 {num_test_episodes} 个回合...")

    all_test_rewards = []
    all_agent_rewards = []
    all_trajectories = []
    uav_traj = []
    user_traj = []
    episode_rewards = []
    for test_ep in tqdm(range(num_test_episodes), desc="测试进度"):
        episode_rewards, agent_rewards, uav_traj, user_traj = test_episode(
            env, maddpg, running_norms, uav_num, total_time_slots
        )

        avg_reward = np.mean(episode_rewards)
        all_test_rewards.append(avg_reward)
        all_agent_rewards.append(agent_rewards)
        all_trajectories.append((uav_traj, user_traj))

        print(f"  回合 {test_ep + 1} 平均奖励: {avg_reward:.3f}")

    # ========= 统计测试结果 ========= #
    mean_reward = np.mean(all_test_rewards)
    std_reward = np.std(all_test_rewards)
    mean_agent_rewards = np.mean(all_agent_rewards, axis=0)

    print("\n" + "=" * 50)
    print("测试结果统计:")
    print(f"  平均总奖励: {mean_reward:.3f} ± {std_reward:.3f}")
    print("\n各Agent平均奖励:")
    for i in range(uav_num):
        print(f"  UAV {i}: {mean_agent_rewards[i]:.3f}")
    print(f"  RIS: {mean_agent_rewards[uav_num]:.3f}")
    print("=" * 50)

    # ========= 保存测试结果 ========= #
    today_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    time_slots, uav_num, _ = uav_traj.shape
    _, num_users, _ = user_traj.shape
    # -------------------- 保存 UAV 轨迹 --------------------
    uav_data = []
    for t in range(time_slots):
        for i in range(uav_num):
            x, y, z = uav_traj[t, i]
            uav_data.append([t, i, x, y, z])

    df_uav = pd.DataFrame(uav_data, columns=['time_slot', 'uav_id', 'x', 'y', 'z'])
    df_uav.to_csv(f'{today_str}_MADDPG_test_uav_traj_seed{seed}.csv', index=False)
    print("✅ UAV轨迹已按长表格格式保存")

    # -------------------- 保存用户轨迹 --------------------
    user_data = []
    for t in range(time_slots):
        for i in range(num_users):
            x, y, z = user_traj[t, i]
            user_data.append([t, i, x, y, z])

    df_user = pd.DataFrame(user_data, columns=['time_slot', 'user_id', 'x', 'y', 'z'])
    df_user.to_csv(f'{today_str}_MADDPG_test_user_traj_seed{seed}.csv', index=False)
    print("✅ 用户轨迹已按长表格格式保存")

    # -------------------- 保存每时隙奖励 --------------------
    df_rewards = pd.DataFrame({
        'time_slot': np.arange(time_slots),
        'reward': episode_rewards
    })
    df_rewards.to_csv(f'{today_str}_MADDPG_test_rewards_seed{seed}.csv', index=False)
    print("✅ 每时隙奖励已保存到CSV")