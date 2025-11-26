import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

# 读取Excel文件
df = pd.read_csv('STAR-RIS-Elements.csv')

DDPG_IPM = '#4169E1'  # Royal Blue  
TD3_IPM = '#FF6B6B'  # Coral Red
SAC_IPM = "#0FB433" # Coral Green   
RA_IPM = "#F39E00" # Coral Yellow
Pure_TD3_IPM = "#C910EE" 
GRID_COLOR = '#E0E0E0'  # 柔和的网格线颜色

# 绘制能耗对比图
plt.figure(figsize=(10, 8))
plt.plot(df['STAR-RIS Elements'], df['RA-IPM'], marker='s', linestyle=':', markersize = 8,
         label="RA-IPM", color = RA_IPM, linewidth = 2, markerfacecolor='white', markeredgewidth=1.5)

plt.plot(df['STAR-RIS Elements'], df['Pure TD3-BC-IPM'], marker='^', linestyle='-', markersize = 8,
         label="Pure TD3-BC", color = Pure_TD3_IPM, linewidth = 2, markerfacecolor='white', markeredgewidth=1.5)

plt.plot(df['STAR-RIS Elements'], df['TD3-IPM'], marker='o', linestyle='--', markersize = 8,
         label="TD3-IPM", color = DDPG_IPM, linewidth = 2, markerfacecolor='white', markeredgewidth=1.5)

plt.plot(df['STAR-RIS Elements'], df['SAC-IPM'], marker='v', linestyle='-.', markersize = 8,
         label="SAC-IPM", color = SAC_IPM, linewidth = 2, markerfacecolor='white', markeredgewidth=1.5)

plt.plot(df['STAR-RIS Elements'], df['TD3-BC-IPM'], marker='D', linestyle='-', markersize = 8,
         label="TD3-BC-IPM", color = TD3_IPM, linewidth = 2, markerfacecolor='white', markeredgewidth=1.5)

plt.xlabel("Number of STAR-RIS elements", fontsize=24)
plt.ylabel("Energy consumption (J)", fontsize=24)
plt.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8, zorder=1)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.ylim(0.035,0.52)
plt.legend(fontsize=24)
plt.tight_layout()
plt.savefig("STAR-RIS-Elements.pdf")
plt.show()
