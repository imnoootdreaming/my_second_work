import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['font.family'] = 'Times New Roman'

# --- 1. 配置参数 ---

# 定义要处理的随机种子列表
SEEDS = [42, 43, 44]

# 定义CSV文件的命名模板 (请确保您的文件名符合这个格式)
OVERALL_REWARD_TEMPLATE = 'IPPO_training_rewards_seed{}.csv'
ALL_AGENTS_REWARD_TEMPLATE = 'IPPO_all_agents_rewards_seed{}.csv'

# 滑动平均窗口大小，用于平滑曲线
SMOOTHING_WINDOW = 1

# Agent 的名字列表
AGENT_NAMES = ['UAV_0', 'UAV_1', 'UAV_2', 'RIS']

# 图像输出配置
OUTPUT_FILENAME = 'ippo_combined_convergence_curve.png'


# --- 2. 数据加载与处理函数 (与之前相同) ---

def load_and_process_rewards(file_template, seeds, column_name):
    """
    加载多个随机种子的奖励数据，并计算均值和标准差。
    """
    all_rewards_data = []

    for seed in seeds:
        try:
            filepath = file_template.format(seed)
            df = pd.read_csv(filepath)
            all_rewards_data.append(df[column_name])
        except FileNotFoundError:
            print(f"警告: 找不到文件 {filepath}，将跳过该种子。")
            continue

    if not all_rewards_data:
        print(f"错误: 无法为 '{column_name}' 加载任何数据，请检查文件名和路径。")
        return None, None, None

    min_length = min(len(data) for data in all_rewards_data)
    aligned_data = [data.head(min_length).values for data in all_rewards_data]

    rewards_matrix = np.array(aligned_data)
    mean_rewards = np.mean(rewards_matrix, axis=0)
    std_rewards = np.std(rewards_matrix, axis=0)

    mean_smoothed = pd.Series(mean_rewards).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    std_smoothed = pd.Series(std_rewards).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    episodes = np.arange(min_length)

    return episodes, mean_smoothed, std_smoothed


def plot_combined_curves(plot_data, output_filename):
    """
    在同一张图上绘制所有实体（整体及各个Agent）的奖励曲线。
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 定义颜色方案
    colors = {
        'Overall': ('#ff7f0e', '#ffbb78'),
        'UAV_0': ('#1f77b4', '#aec7e8'),
        'UAV_1': ('#e377c2', '#f7b6d2'),
        'UAV_2': ('#9467bd', '#c5b0d5'),
        'RIS': ('#2ca02c', '#98df8a')
    }

    label_mapping = {
        'Overall': 'Global reward',
        'RIS': 'IRS agent',
        'UAV_0': 'UAV agent 1',
        'UAV_1': 'UAV agent 2',
        'UAV_2': 'UAV agent 3'
    }

    # ✅ 固定绘制顺序
    plot_order = ['Overall', 'RIS', 'UAV_0', 'UAV_1', 'UAV_2']

    for name in plot_order:
        data = next((d for d in plot_data if d['name'] == name), None)
        if data is None:
            continue

        episodes = data['episodes']
        mean_rewards = data['mean']
        std_rewards = data['std']

        main_color, shade_color = colors.get(name, ('blue', 'lightblue'))
        linestyle = '-.' if name == 'Overall' else '-'
        linewidth = 1 if name == 'Overall' else 1
        alpha = 0.14
        zorder = 10 - plot_order.index(name)  # ✅ 顺序控制层级（Global 最上）

        # # --- 绘制阴影 ---
        # ax.fill_between(
        #     episodes,
        #     mean_rewards - std_rewards,
        #     mean_rewards + std_rewards,
        #     color=shade_color,
        #     alpha=alpha,
        #     zorder=zorder
        # )

        # --- 绘制主线 ---
        ax.plot(
            episodes,
            mean_rewards,
            color=main_color,
            label=label_mapping.get(name, name),
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder + 0.5
        )

    # --- 美化 ---
    ax.set_xlabel('Episodes', fontsize=24)
    ax.set_ylabel('Average reward', fontsize=24)
    ax.legend(fontsize=24, loc='lower right')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, linestyle='--', color='#E0E0E0', alpha=0.8, zorder=1)

    plt.tight_layout()
    # plt.savefig(output_filename, dpi=300)
    plt.show()



# --- 4. 主程序 ---

if __name__ == '__main__':
    all_plot_data = []

    # a) 处理整体奖励
    print("正在处理 Overall Rewards...")
    overall_episodes, overall_mean, overall_std = load_and_process_rewards(
        OVERALL_REWARD_TEMPLATE, SEEDS, 'reward'
    )
    if overall_episodes is not None:
        all_plot_data.append({
            'name': 'Overall',
            'episodes': overall_episodes,
            'mean': overall_mean,
            'std': overall_std
        })

    # b) 处理每个 Agent 的奖励
    print("\n正在处理 Individual Agent Rewards...")
    for agent_name in AGENT_NAMES:
        column = f'{agent_name}_reward'
        print(f"--- Processing for {agent_name} ---")
        agent_episodes, agent_mean, agent_std = load_and_process_rewards(
            ALL_AGENTS_REWARD_TEMPLATE, SEEDS, column
        )
        if agent_episodes is not None:
            all_plot_data.append({
                'name': agent_name,
                'episodes': agent_episodes,
                'mean': agent_mean,
                'std': agent_std
            })

    # c) 如果成功加载了数据，则进行绘图
    if all_plot_data:
        print("\n所有数据处理完毕，开始绘图...")
        plot_combined_curves(all_plot_data, OUTPUT_FILENAME)
    else:
        print("\n未能加载任何有效数据，无法生成图像。请检查您的文件名和路径。")