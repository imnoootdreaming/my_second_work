import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = 'Times New Roman'

# === 1. 读取数据 ===
uav_df = pd.read_csv("TEST_uav_traj_seed1208.csv")
user_df = pd.read_csv("TEST_user_traj_seed1208.csv")
# reward_df = pd.read_csv("TEST_rewards_seed1208.csv")  # 不再使用 reward

# === 2. 筛选前 50 个时隙 ===
uav_df = uav_df[uav_df['time_slot'] < 50]
user_df = user_df[user_df['time_slot'] < 50]
# reward_df = reward_df[reward_df['time_slot'] < 50]  # 不再使用 reward

# === 3. 创建 3D 图 ===
fig = plt.figure(figsize=(10, 8))
ax3d = fig.add_subplot(111, projection='3d')  # 不再使用 GridSpec，不需要第二个子图

# === 4. 用户轨迹绘制，中间点用三角形 ===
user_colors = plt.cm.tab10.colors
for uid in sorted(user_df['user_id'].unique()):
    color = user_colors[0 % len(user_colors)]
    u = user_df[user_df['user_id'] == uid]
    ax3d.plot(u['x'], u['y'], u['z'], color=color, linewidth=2, alpha=0.8)
    ax3d.scatter(u['x'][1:-1], u['y'][1:-1], u['z'][1:-1], color=color, s=20, marker='^')
    ax3d.scatter(u['x'].iloc[0], u['y'].iloc[0], u['z'].iloc[0], color=color, marker='o', s=40)
    ax3d.scatter(u['x'].iloc[-1], u['y'].iloc[-1], u['z'].iloc[-1], color=color, marker='*', s=40)
ax3d.plot([], [], [], color=color, label='Users trajectories')

# === 5. UAV轨迹绘制，中间点用三角形 ===
uav_colors = plt.cm.Set2.colors
for i, uid in enumerate(sorted(uav_df['uav_id'].unique())):
    uav = uav_df[uav_df['uav_id'] == uid]
    color = uav_colors[i % len(uav_colors)]
    ax3d.plot(uav['x'], uav['y'], uav['z'], color=color, linewidth=2, alpha=0.8,
              label=f'UAV {uid + 1} trajectory')
    ax3d.scatter(uav['x'][1:-1], uav['y'][1:-1], uav['z'][1:-1], color=color, s=20, marker='^')
    ax3d.scatter(uav['x'].iloc[0], uav['y'].iloc[0], uav['z'].iloc[0], color=color, marker='o', s=60)
    ax3d.scatter(uav['x'].iloc[-1], uav['y'].iloc[-1], uav['z'].iloc[-1], color=color, marker='*', s=60)

# === 6. 美化3D图 ===
ax3d.set_xlabel('X position (m)', fontsize=18)
ax3d.tick_params(axis='both', which='major', labelsize=13)
ax3d.set_ylabel('Y position (m)', fontsize=18)
ax3d.set_zlabel('Z position (m)', fontsize=18)
ax3d.legend(loc='best', fontsize=18)
ax3d.view_init(elev=44, azim=40)

plt.tight_layout()
plt.show()
