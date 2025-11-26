import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

# 读取CSV文件
df = pd.read_csv('user_num.csv')

# 打印列名查看
print("列名:", df.columns.tolist())

DDPG_IPM = '#4169E1'  # Royal Blue
TD3_IPM = '#FF6B6B'  # Coral Red
SAC_IPM = "#0FB433" # Coral Green
RA_IPM = "#F39E00" # Coral Yellow
Pure_TD3_IPM = "#C910EE"
GRID_COLOR = '#E0E0E0'  # 柔和的网格线颜色

# 创建子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 第一个子图：时延 (奇数列索引: 1, 3, 5, 7)
ax2.plot(df['user'], df.iloc[:, 1], marker='D', linestyle='-', markersize=8,
         label="Beta-IPPO-CCCP (Proposed)", color=TD3_IPM, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)

ax2.plot(df['user'], df.iloc[:, 3], marker='v', linestyle=':', markersize=8,
         label="ITD3-CCCP", color=SAC_IPM, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)

ax2.plot(df['user'], df.iloc[:, 5], marker='o', linestyle='--', markersize=8,
         label="IDDPG-CCCP", color=DDPG_IPM, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)

ax2.plot(df['user'], df.iloc[:, 7], marker='^', linestyle='-.', markersize=8,
         label="Pure Beta-IPPO", color=Pure_TD3_IPM, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)

ax2.set_xlabel("The number of users", fontsize=24)
ax2.set_ylabel("Delay (s)", fontsize=24)
ax2.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8, zorder=1)
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.set_xticks(df['user'])

# 第二个子图：奖励 (偶数列索引: 2, 4, 6, 8)
ax1.plot(df['user'], df.iloc[:, 2], marker='D', linestyle='-', markersize=8,
         label="Beta-IPPO-CCCP (Proposed)", color=TD3_IPM, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)

ax1.plot(df['user'], df.iloc[:, 4], marker='v', linestyle=':', markersize=8,
         label="ITD3-CCCP", color=SAC_IPM, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)

ax1.plot(df['user'], df.iloc[:, 6], marker='o', linestyle='--', markersize=8,
         label="IDDPG-CCCP", color=DDPG_IPM, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)

ax1.plot(df['user'], df.iloc[:, 8], marker='^', linestyle='-.', markersize=8,
         label="Pure Beta-IPPO", color=Pure_TD3_IPM, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)

ax1.set_ylabel("Average reward", fontsize=24)
ax1.set_xlabel("The number of users", fontsize=24)
ax1.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8, zorder=1)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.legend(fontsize=20, loc='upper left')
ax1.set_xticks(df['user'])
ax1.set_xticks(df['user'])
ax2.set_xticks(df['user'])
plt.tight_layout()
plt.show()