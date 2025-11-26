import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import  ConnectionPatch
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'

def zone_and_linked(ax,axins,zone_left,zone_right,x,y,linked='bottom',
                    x_ratio=0.05,y_ratio=0.05):
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
    xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio - 0.3
    xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio + 0.3

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data)-(np.max(y_data)-np.min(y_data))*y_ratio - 0.5
    ylim_top = np.max(y_data)+(np.max(y_data)-np.min(y_data))*y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top+0.5)

    # 定义扩展后的边界
    rect_bottom = ylim_bottom - 425
    rect_top = ylim_top + 400
    rect_right = xlim_right - 1

    # 绘制外框（使用扩展后的上下边界）
    ax.plot([xlim_left, rect_right, rect_right, xlim_left, xlim_left],
            [rect_bottom, rect_bottom, rect_top, rect_top, rect_bottom],
            color='black', linestyle='--', linewidth=1.5, zorder=5)

    # 连接线端点位置同步更新
    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, rect_top)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (rect_right, rect_top)  # 注意这里改了
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, rect_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (rect_right, rect_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (rect_right, rect_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (rect_right, rect_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (rect_right, rect_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (rect_right, rect_bottom)

    # 添加连接线
    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)

# 读取Excel文件
df = pd.read_csv('CCCP_vs_ES.csv')

ES_COLOR = '#4169E1'  # Royal Blue
IPM_COLOR = '#FF6B6B'  # Coral Red
GRID_COLOR = '#E0E0E0'  # 柔和的网格线颜色

# 创建两个子图（上下排列）

plt.figure(figsize=(12, 8))
# --- 子图1：能耗对比 ---
plt.plot(df['ID'], df['CCCP_delay'], marker='s', linestyle='-', markersize=8,
         label="CCCP", color=IPM_COLOR, linewidth=2,
         markerfacecolor='white', markeredgewidth=1.5)
plt.plot(df['ID'], df['ES_delay'], marker='o', linestyle='--', markersize=8,
         label="Exhaustive search", color=ES_COLOR, linewidth=2,
         markerfacecolor='white', markeredgewidth=1.5)
plt.xlabel("Index of cases", fontsize=24)
plt.ylabel("Total Delay (s)", fontsize=24)
plt.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8, zorder=1)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.legend(fontsize=24)

# 布局与保存
plt.tight_layout()
plt.savefig("CCCP_vs_ES.pdf")
plt.show()
