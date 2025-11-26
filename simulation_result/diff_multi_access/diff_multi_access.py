import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams['font.family'] = 'Times New Roman'

# 读取CSV文件
df1 = pd.read_csv('heterogeneous_size.csv')
df2 = pd.read_csv('diff_multi_access.csv')

# 定义颜色方案
OMA_COLOR = '#4169E1'  # Royal Blue
NCONOMA_COLOR = '#FF6B6B'  # Coral Red
CONOMA_COLOR = "#0FB433"  # Coral Green
GRID_COLOR = '#E0E0E0'  # 柔和的网格线颜色

# 创建包含两个子图的图表（垂直排列）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# ============== 左图：heterogeneous_size.csv ==============
ax2.plot(df1.iloc[:, 0], df1['OMA'], marker='o', linestyle='--', markersize=8,
         label="OMA", color=OMA_COLOR, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)
ax2.plot(df1.iloc[:, 0], df1['CO-NOMA'], marker='^', linestyle='-.', markersize=8,
         label="CO-NOMA", color=CONOMA_COLOR, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)
ax2.plot(df1.iloc[:, 0], df1['NCO-NOMA'], marker='s', linestyle='-', markersize=8,
         label="NCO-NOMA", color=NCONOMA_COLOR, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)

ax2.set_xlabel("Data size difference within a user group (knats)", fontsize=24)
ax2.set_ylabel("Delay (s)", fontsize=24)
ax2.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.6, zorder=1)
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.set_ylim(0.4, 2.9)

# ============== 右图：diff_multi_access.csv ==============
ax1.plot(df2.iloc[:, 0], df2['OMA'], marker='o', linestyle='--', markersize=8,
         label="TDMA", color=OMA_COLOR, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)
ax1.plot(df2.iloc[:, 0], df2['CO-NOMA'], marker='^', linestyle='-.', markersize=8,
         label="CO-NOMA", color=CONOMA_COLOR, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)
ax1.plot(df2.iloc[:, 0], df2['NCO-NOMA'], marker='s', linestyle='-', markersize=8,
         label="NCO-NOMA", color=NCONOMA_COLOR, linewidth=2, markerfacecolor='white', markeredgewidth=1.5)

ax1.set_xlabel("Data size of each task (knats)", fontsize=24)
ax1.set_ylabel("Delay (s)", fontsize=24)
ax1.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.6, zorder=1)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.set_ylim(0.4, 2.9)
ax1.legend(fontsize=20)
ax1.set_xticks(df2.iloc[:, 0])
ax2.set_xticks(df1.iloc[:, 0])
# 调整子图间距
plt.tight_layout()
plt.show()