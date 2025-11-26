import os
import random
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from my_env_beta_distribution import MyEnv
import warnings
from datetime import datetime
from utils.normalization import Normalization
from IPPO_agent import IPPO

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_trained_models(agents_uav, agent_ris, model_dir, device):
    """
    加载训练好的模型权重

    Args:
        agents_uav: UAV agents 字典
        agent_ris: RIS agent
        model_dir: 模型文件所在目录
        device: 设备 (cpu/cuda)
    """
    # 加载 UAV agents
    for agent_id in agents_uav.keys():
        uav_idx = agent_id.split('_')[1]
        actor_path = os.path.join(model_dir, f'uav_{uav_idx}_actor.pth')
        critic_path = os.path.join(model_dir, f'uav_{uav_idx}_critic.pth')

        if os.path.exists(actor_path) and os.path.exists(critic_path):
            agents_uav[agent_id].actor.load_state_dict(
                torch.load(actor_path, map_location=device)
            )
            agents_uav[agent_id].critic.load_state_dict(
                torch.load(critic_path, map_location=device)
            )
            print(f"✅ 成功加载 {agent_id} 模型")
        else:
            print(f"❌ 未找到 {agent_id} 模型文件")

    # 加载 RIS agent
    ris_actor_path = os.path.join(model_dir, 'RIS_actor.pth')
    ris_critic_path = os.path.join(model_dir, 'RIS_critic.pth')

    if os.path.exists(ris_actor_path) and os.path.exists(ris_critic_path):
        agent_ris.actor.load_state_dict(
            torch.load(ris_actor_path, map_location=device)
        )
        agent_ris.critic.load_state_dict(
            torch.load(ris_critic_path, map_location=device)
        )
        print(f"✅ 成功加载 RIS 模型")
    else:
        print(f"❌ 未找到 RIS 模型文件")


def test_episode(env, agents_uav, agent_ris, running_norms_uav, running_norm_ris, uav_num):
    """
    执行一个测试回合

    Args:
        env: 环境
        agents_uav: UAV agents 字典
        agent_ris: RIS agent
        running_norms_uav: UAV 状态归一化器字典
        running_norm_ris: RIS 状态归一化器
        uav_num: UAV 数量

    Returns:
        episode_rewards: 每个时隙的奖励
        uav_trajectory: UAV 轨迹
        user_trajectory: 用户轨迹
    """
    s = env.reset()
    terminal = False

    episode_rewards = []
    uav_trajectory = []
    user_trajectory = []

    # 设置模型为评估模式
    for agent_id in agents_uav.keys():
        agents_uav[agent_id].actor.eval()
    agent_ris.actor.eval()

    with torch.no_grad():  # 测试时不需要梯度
        while not terminal:
            # 记录当前位置
            uav_trajectory.append(env.getPosUAV())
            user_trajectory.append(env.getPosUser())

            # UAV agents 选择动作
            actions_uav_dict = {}
            for uav_i in range(uav_num):
                agent_id = f"uav_{uav_i}"
                s_uav_norm = running_norms_uav[agent_id](np.array(s["uav"][agent_id]))
                a_uav, _ = agents_uav[agent_id].choose_action(s_uav_norm)

                actions_uav_dict[agent_id] = a_uav

            # RIS agent 选择动作
            s_ris_norm = running_norm_ris(np.array(s["ris"]))
            a_ris, _ = agent_ris.choose_action(s_ris_norm)

            # 环境执行
            next_s, total_reward, r_dict, done = env.step({"uav": actions_uav_dict, "ris": a_ris}, 0)

            episode_rewards.append(total_reward)
            s = next_s
            terminal = done

    return episode_rewards, np.array(uav_trajectory), np.array(user_trajectory)


if __name__ == "__main__":
    # ========= 设置随机种子 ========= #
    seed = 6
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ========= 设备配置 ========= #
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"使用设备: {device}")

    # ========= 模型路径配置 ========= #
    # 请修改为你的模型文件所在目录
    model_dir = "./checkpoint/IPPO_NCO_NOMA_models"  # 修改为你的模型保存路径

    # ========= PPO 参数 ========= #
    actor_lr = 3e-4
    critic_lr = 3e-4
    hidden_dim = 128
    gamma = 0.99
    lmbda = 0.95
    eps = 0.2
    epochs = 10
    num_episodes = 1000

    # ========= 环境参数（与训练时保持一致）========= #
    K = 10
    noma_group_num = 3
    uav_num = noma_group_num
    users_num_per_noma_group = 2
    num_users = noma_group_num * users_num_per_noma_group
    center = (0, 200)
    radius = 60
    ris_pos = [0, 0, 20]

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

    total_time_slots = 10

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

    # ========= 获取状态和动作维度 ========= #
    state_dim_uav = env.observation_space["uav"]["uav_0"].shape[0]
    state_dim_ris = env.observation_space["ris"].shape[0]
    action_dim_uav = env.action_space["uav"]["uav_0"].shape[0]
    action_dim_ris = env.action_space["ris"].shape[0]
    action_uav_low = env.action_space["uav"]["uav_0"].low
    action_uav_high = env.action_space["uav"]["uav_0"].high
    action_ris_low = env.action_space["ris"].low
    action_ris_high = env.action_space["ris"].high

    # ========= 创建 Agents ========= #
    agents_uav = {}
    running_norms_uav = {}
    for i in range(uav_num):
        agent_id = f"uav_{i}"
        agents_uav[agent_id] = IPPO(
            state_dim_uav, hidden_dim, action_dim_uav,
            action_uav_low, action_uav_high,
            actor_lr, critic_lr, lmbda, eps, gamma,
            epochs, num_episodes, device
        )
        running_norms_uav[agent_id] = Normalization(state_dim_uav)

    agent_ris = IPPO(
        state_dim_ris, hidden_dim, action_dim_ris,
        action_ris_low, action_ris_high,
        actor_lr, critic_lr, lmbda, eps, gamma,
        epochs, num_episodes, device
    )
    running_norm_ris = Normalization(state_dim_ris)

    # ========= 加载训练好的模型 ========= #
    print("\n正在加载训练好的模型...")
    load_trained_models(agents_uav, agent_ris, model_dir, device)

    # ========= 执行测试 ========= #
    num_test_episodes = 1  # 测试回合数
    print(f"\n开始测试，共 {num_test_episodes} 个回合...")

    all_test_rewards = []
    all_trajectories = []
    episode_rewards = []
    uav_traj  = []
    user_traj = []
    for test_ep in tqdm(range(num_test_episodes), desc="测试进度"):
        episode_rewards, uav_traj, user_traj = test_episode(
            env, agents_uav, agent_ris,
            running_norms_uav, running_norm_ris,
            uav_num
        )

        avg_reward = np.mean(episode_rewards)
        all_test_rewards.append(avg_reward)
        all_trajectories.append((uav_traj, user_traj))

        print(f"  回合 {test_ep + 1} 平均奖励: {avg_reward:.3f}")

    # ========= 统计测试结果 ========= #
    mean_reward = np.mean(all_test_rewards)
    std_reward = np.std(all_test_rewards)

    print("\n" + "=" * 50)
    print("测试结果统计:")
    print(f"  平均奖励: {mean_reward:.3f} ± {std_reward:.3f}")
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
    df_uav.to_csv(f'{today_str}_IPPO_test_uav_traj_seed{seed}.csv', index=False)
    print("✅ UAV轨迹已按长表格格式保存")

    # -------------------- 保存用户轨迹 --------------------
    user_data = []
    for t in range(time_slots):
        for i in range(num_users):
            x, y, z = user_traj[t, i]
            user_data.append([t, i, x, y, z])


    df_user = pd.DataFrame(user_data, columns=['time_slot', 'user_id', 'x', 'y', 'z'])
    df_user.to_csv(f'{today_str}_IPPO_test_user_traj_seed{seed}.csv', index=False)
    print("✅ 用户轨迹已按长表格格式保存")

    # -------------------- 保存每时隙奖励 --------------------
    df_rewards = pd.DataFrame({
        'time_slot': np.arange(time_slots),
        'reward': episode_rewards
    })
    df_rewards.to_csv(f'{today_str}_IPPO_test_rewards_seed{seed}.csv', index=False)
    print("✅ 每时隙奖励已保存到CSV")