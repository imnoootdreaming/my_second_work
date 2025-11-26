import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'

# 读取数据
file_path = 'compare_CCCP_vs_ES.csv'
data = pd.read_csv(file_path)

# 提取数据
cases = data['Case']
delay_cccp = data['Delay_CCCP']
delay_es = data['Delay_ES']
time_cccp = data['Time_CCCP(s)']
time_es = data['Time_ES(s)']

bar_width = 0.35
x = np.arange(len(cases))

fig, ax1 = plt.subplots(figsize=(10, 8))

# 🎨 蓝柱 + 红线的配色方案（与图中风格接近）
colors = {
    'delay_cccp': '#6FA8DC',
    'delay_es':   '#2E6FBA',
    'time_cccp': '#FF8090',
    'time_es': '#CD5C5C'
}

# ---- 左轴 Delay（蓝色柱状图） ----
ax1.bar(x - bar_width/2, delay_cccp, bar_width, label="Delay of CCCP",
        color=colors['delay_cccp'], alpha=0.9, edgecolor='white', linewidth=1.0)
ax1.bar(x + bar_width/2, delay_es, bar_width, label="Delay of CFGS",
        color=colors['delay_es'], alpha=0.9, edgecolor='white', linewidth=1.0)

ax1.set_xlabel('Random cases', fontsize=24)
ax1.set_ylabel('Delay (s)', fontsize=24, color='#003049')
ax1.set_xticks(x)
ax1.set_xticklabels(cases)
ax1.tick_params(axis='y', labelcolor='#003049', colors='#003049', labelsize=18)
ax1.tick_params(axis='x', labelsize=18)

# ---- 右轴 Time（红色曲线） ----
ax2 = ax1.twinx()
ax2.plot(x, time_cccp, color=colors['time_cccp'], linestyle='--', marker='o',
         markersize=8, linewidth=2.5, label="Running time of CCCP", alpha=0.9)
ax2.plot(x, time_es, color=colors['time_es'], linestyle='--', marker='s',
         markersize=8, linewidth=2.5, label="Running time of CFGS", alpha=0.9)
ax2.set_ylabel('Running time (s)', fontsize=24, color='#780000')
ax2.tick_params(axis='y', labelcolor='#780000', colors='#780000', labelsize=18)

# ---- 轴线颜色（浅色） ----
ax1.spines['left'].set_color('#003049')
ax1.spines['left'].set_linewidth(1.6)
ax2.spines['right'].set_color('#780000')
ax2.spines['right'].set_linewidth(1.6)
ax2.spines['left'].set_visible(False)
ax1.spines['top'].set_color('#E0E0E0')
ax1.spines['bottom'].set_color('#E0E0E0')

# ---- 图例 ----
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,fontsize=24)

# ---- 坐标与样式 ----
ax1.set_ylim(0, 1.05)
ax2.set_ylim(-50, 5500)
ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
ax1.set_axisbelow(True)
ax1.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.show()
