import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'

# 读取三个CSV文件（空格分隔）
df_6_parallel = pd.read_csv('CCCP_group_NOMA_6_parallel.csv')
df_4_parallel = pd.read_csv('CCCP_group_NOMA_4_parallel.csv')
df_2_parallel = pd.read_csv('CCCP_group_NOMA_2_parallel.csv')

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8))

# 设置柱状图的位置
x = np.arange(len(df_6_parallel))
width = 0.25

# 画三组柱状图
bars3 = ax.bar(x - width - 0.015, df_2_parallel['Running_time'], width,
               label='4 users', color='#7DCFB6', alpha=0.9)
bars2 = ax.bar(x, df_4_parallel['Running_time'], width,
               label='8 users', color='#5B8FF9', alpha=0.9)
bars1 = ax.bar(x + width + 0.015, df_6_parallel['Running_time'], width,
               label='12 users', color='#F08BB4', alpha=0.9)

# 设置图表标签和标题
ax.set_xlabel('Random cases', fontsize=24)
ax.set_ylabel('Running time (s)', fontsize=24)
ax.set_xticks(x)
ax.set_xticklabels(df_6_parallel['ID'])
ax.legend(fontsize=24)
ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=18)
ax.set_ylim(0,0.7)

plt.tight_layout()
plt.show()