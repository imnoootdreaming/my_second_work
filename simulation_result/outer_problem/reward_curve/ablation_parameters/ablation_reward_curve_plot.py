import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import ConnectionPatch

plt.rcParams['font.family'] = 'Times New Roman'


def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    # 定义扩展后的边界
    rect_bottom = ylim_bottom - 0.01
    rect_top = ylim_top + 0.01
    rect_right = xlim_right

    # 绘制外框（使用扩展后的上下边界）
    ax.plot([xlim_left, rect_right, rect_right, xlim_left, xlim_left],
            [rect_bottom, rect_bottom, rect_top, rect_top, rect_bottom],
            color='black', linestyle='--', linewidth=1.5, zorder=5)

    # 连接线端点位置同步更新
    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, rect_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (rect_right, rect_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, rect_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (rect_right, rect_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (rect_right, rect_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (rect_right, rect_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (rect_right, rect_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (rect_right, rect_bottom)

    # 添加连接线（设置为虚线）
    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax,
                          linestyle='--', linewidth=1.5, color='black')
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax,
                          linestyle='--', linewidth=1.5, color='black')
    axins.add_artist(con)


# --- 1. 配置参数 ---

# 定义要处理的随机种子列表
SEEDS = [42, 43, 44]

# 定义不同算法的CSV文件命名模板（加载顺序）
ALGORITHM_TEMPLATES = {
    'No concurrent transmission time optimization':'IPPO_random_transmission_training_rewards_seed{}.csv',
    'No UAV trajectories optimization': 'IPPO_random_trajectoryUAV_training_rewards_seed{}.csv',
    'No UAV selection strategy optimization': 'IPPO_random_selectionUAV_training_rewards_seed{}.csv',
    'No IRS phase shift optimization': 'IPPO_randomRIS_training_rewards_seed{}.csv',
    'Full optimization': 'IPPO_training_rewards_seed{}.csv',
}

# 滑动平均窗口大小，用于平滑曲线
SMOOTHING_WINDOW = 1

# 图像输出配置
OUTPUT_FILENAME = 'algorithm_comparison_convergence_curve.png'


# --- 2. 数据加载与处理函数 ---
def load_and_process_rewards(file_template, seeds, column_name='reward'):
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
        except KeyError:
            print(f"警告: 文件 {filepath} 中找不到列 '{column_name}'，将跳过该种子。")
            continue

    if not all_rewards_data:
        print(f"错误: 无法为模板 '{file_template}' 加载任何数据，请检查文件名和路径。")
        return None, None, None

    # 找到最短的数据长度，对齐所有数据
    min_length = min(len(data) for data in all_rewards_data)
    aligned_data = [data.head(min_length).values for data in all_rewards_data]

    # 计算均值和标准差
    rewards_matrix = np.array(aligned_data)
    mean_rewards = np.mean(rewards_matrix, axis=0)
    std_rewards = np.std(rewards_matrix, axis=0)

    # 平滑处理
    mean_smoothed = pd.Series(mean_rewards).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    std_smoothed = pd.Series(std_rewards).rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    episodes = np.arange(min_length)

    return episodes, mean_smoothed, std_smoothed


# --- 3. 算法对比绘图函数 ---
def plot_algorithm_comparison(plot_data, output_filename):
    """
    在同一张图上绘制不同算法的奖励曲线进行对比，并添加局部放大图。
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 定义每个算法的颜色方案 (主色, 阴影色)
    colors = {
        'Full optimization': ('#ff7f0e', '#ffbb78'),  # 橙色
        'No IRS phase shift optimization': ('#1f77b4', '#aec7e8'),  # 蓝色
        'No UAV trajectories optimization': ('#2ca02c', '#98df8a'),  # 绿色
        'No UAV selection strategy optimization': ('#9467bd', '#c5b0d5'),  # 紫色系，高雅、对比柔和
        'No concurrent transmission time optimization': ('#e377c2', '#f7b6d2'),  # 浅粉红系，柔和优雅，与橙色区分明显
    }

    # 算法显示名称映射
    label_mapping = {
        'Full optimization': 'Full optimization',
        'No IRS phase shift optimization': 'No IRS phase shift optimization',
        'No UAV selection strategy optimization': 'No UAV selection strategy optimization',
        'No UAV trajectories optimization':'No UAV trajectories optimization',
        'No concurrent transmission time optimization':'No concurrent transmission time optimization'
    }

    # 存储所有的均值曲线数据，用于局部放大
    all_mean_rewards = []

    # 绘制每个算法的曲线
    for data in plot_data:
        algo_name = data['algorithm']
        episodes = data['episodes']
        mean_rewards = data['mean']
        std_rewards = data['std']

        main_color, shade_color = colors.get(algo_name, ('blue', 'lightblue'))
        linewidth = 1
        alpha = 0.14

        # # 绘制标准差阴影
        # ax.fill_between(
        #     episodes,
        #     mean_rewards - std_rewards,
        #     mean_rewards + std_rewards,
        #     color=shade_color,
        #     alpha=alpha,
        #     zorder=1
        # )

        # 绘制均值曲线
        ax.plot(
            episodes,
            mean_rewards,
            color=main_color,
            label=label_mapping.get(algo_name, algo_name),
            linewidth=linewidth,
            zorder=2
        )
        #
        # # 只保留需要放大的算法
        # if algo_name != 'Gaussian-IPPO-CCCP':
        #     all_mean_rewards.append(mean_rewards)

    # 美化图表
    ax.set_xlabel('Episodes', fontsize=24)
    ax.set_ylabel('Average reward', fontsize=24)
    # 获取当前图例句柄和标签
    handles, labels = ax.get_legend_handles_labels()
    desired_order = ['Full optimization',
                     'No IRS phase shift optimization',
                     'No UAV selection strategy optimization',
                     'No UAV trajectories optimization',
                     'No concurrent transmission time optimization']  # 按你想要的顺序来
    order = [labels.index(name) for name in desired_order if name in labels]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    ax.legend(handles, labels, fontsize=24, loc='lower right')
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, linestyle='--', color='#E0E0E0', alpha=0.8, zorder=1)

    # --- 添加局部放大图 ---
    # 创建内嵌子图 (位置: 左, 下, 宽, 高)，可根据需要调整位置
    # 格式：(left, bottom, width, height)，取值范围都是0-1
    # axins = ax.inset_axes((0.665, 0.48, 0.3, 0.3))  # 中间偏右位置

    # # 在内嵌图中重新绘制所有算法的曲线
    # for data in plot_data:
    #     algo_name = data['algorithm']
    #     if algo_name == 'Gaussian-IPPO-CCCP':  # 👈 跳过 Gaussian-IPPO
    #         continue
    #     episodes = data['episodes']
    #     mean_rewards = data['mean']
    #     std_rewards = data['std']
    #
    #     main_color, shade_color = colors.get(algo_name, ('blue', 'lightblue'))
    #
    #     # 在内嵌图中绘制
    #     axins.fill_between(
    #         episodes,
    #         mean_rewards - std_rewards,
    #         mean_rewards + std_rewards,
    #         color=shade_color,
    #         alpha=alpha,
    #         zorder=1
    #     )
    #     axins.plot(
    #         episodes,
    #         mean_rewards,
    #         color=main_color,
    #         linewidth=linewidth,
    #         zorder=2
    #     )
    #
    # # 设置内嵌图的网格和刻度
    # axins.grid(True, which='both', linestyle='--', linewidth=0.5)
    # axins.tick_params(axis='both', which='major', labelsize=12)
    #
    # # 使用zone_and_linked函数进行局部放大 (1750-1999 episodes)
    # # 注意：索引从0开始，所以2000个episodes的索引范围是0-1999
    # zone_and_linked(ax, axins, 1750, 1999,
    #                 episodes, all_mean_rewards,
    #                 linked='bottom',
    #                 x_ratio=0.05, y_ratio=0.05)

    plt.tight_layout()
    # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    # print(f"\n图像已保存至: {output_filename}")
    plt.show()


# --- 4. 主程序 ---

if __name__ == '__main__':
    all_plot_data = []

    # 遍历所有算法，加载并处理数据
    for algo_name, file_template in ALGORITHM_TEMPLATES.items():
        print(f"\n正在处理 {algo_name} 算法的数据...")
        episodes, mean_rewards, std_rewards = load_and_process_rewards(
            file_template, SEEDS, column_name='reward'
        )

        if episodes is not None:
            all_plot_data.append({
                'algorithm': algo_name,
                'episodes': episodes,
                'mean': mean_rewards,
                'std': std_rewards
            })
            print(f"{algo_name}: 成功加载 {len(episodes)} 个episode的数据")
        else:
            print(f"{algo_name}: 数据加载失败")

    # 如果成功加载了数据，则进行绘图
    if all_plot_data:
        print("\n所有算法数据处理完毕，开始绘制对比图...")
        plot_algorithm_comparison(all_plot_data, OUTPUT_FILENAME)
    else:
        print("\n未能加载任何有效数据，无法生成图像。请检查您的文件名和路径。")
        print("\n请确保以下文件存在:")
        for algo_name, template in ALGORITHM_TEMPLATES.items():
            for seed in SEEDS:
                print(f"  - {template.format(seed)}")