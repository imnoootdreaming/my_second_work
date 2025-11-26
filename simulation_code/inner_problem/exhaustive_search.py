import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.special import log1p

from equivalent_channel import generate_equivalent_channel


def user1_lager_gain_ES(L_k1, L_k2, tau, D_k1, g_k1, g_k2, uav_pos_pre, uav_pos, case_id):
    # 类型转换
    uav_pos = np.array(uav_pos, dtype=float)
    uav_pos_pre = np.array(uav_pos_pre, dtype=float)

    # 参数定义
    B = 1e6  # 信道带宽 (单位：Hz)
    sigma2 = 3.98e-21 * B  # 噪声功率，单位瓦特
    P_max = 0.1995  # 用户最大功率（瓦特）23dB
    C = 800

    # UAV参数
    c1 = 0.00614
    c2 = 15.976
    kappa_uav = 1e-28  # UAV计算能量系数
    f_uav_max = 3e9  # UAV的最大计算频率
    E_uav_max = 5

    # 用户
    E_max = 0.2
    D_k1_max = np.random.uniform(tau - 0.3, tau)  # 每个用户最大可容忍时延从[tau-0.3,tau]之间随机生成
    D_k2_max = np.random.uniform(D_k1_max, tau)  # D_k2_max 必须大于 D_k1_max 但不能超过tau

    # UAV前后位置差
    uav_diff = np.linalg.norm(uav_pos - uav_pos_pre)

    # 给定上届
    z_max = log1p(g_k2 * P_max / sigma2)

    # 初始化每个变量的搜索范围
    # 1. 数值安全下界（防止 log(0) 或 除以 0）
    MIN_VAL = 1e-20
    # 2. 搜索终止精度（控制循环次数）
    tolerance = 1e-6
    original_ranges = {
        'p_k1_D1': (MIN_VAL, P_max),
        'p_k2_D1': (MIN_VAL, P_max),
        'z': (MIN_VAL, z_max),
        'D_k2': (MIN_VAL, D_k2_max),
        'f_k1_uav': (MIN_VAL, f_uav_max),
        'f_k2_uav': (MIN_VAL, f_uav_max)
    }

    # 初始化最优解
    optimal_value = float('inf')
    # 保守的初始化策略 将初始最优解设为搜索范围的中间值
    #   - 如果第一轮搜索没有找到可行解 至少有一个合理的初始值
    optimal_p_k1_D1 = (original_ranges['p_k1_D1'][0] + original_ranges['p_k1_D1'][1]) / 2
    optimal_p_k2_D1 = (original_ranges['p_k2_D1'][0] + original_ranges['p_k2_D1'][1]) / 2
    optimal_z = (original_ranges['z'][0] + original_ranges['z'][1]) / 2
    optimal_D_k2 = (original_ranges['D_k2'][0] + original_ranges['D_k2'][1]) / 2
    optimal_f_k1_uav = (original_ranges['f_k1_uav'][0] + original_ranges['f_k1_uav'][1]) / 2
    optimal_f_k2_uav = (original_ranges['f_k2_uav'][0] + original_ranges['f_k2_uav'][1]) / 2

    # 初始化搜索范围和步长
    step = 20  # 每轮搜索的步数

    # 初始搜索范围
    current_ranges = {
        'p_k1_D1': original_ranges['p_k1_D1'],
        'p_k2_D1': original_ranges['p_k2_D1'],
        'z': original_ranges['z'],
        'D_k2': original_ranges['D_k2'],
        'f_k1_uav': original_ranges['f_k1_uav'],
        'f_k2_uav': original_ranges['f_k2_uav']
    }

    # 计算初始步长
    step_sizes = {}
    for var_name, (min_val, max_val) in current_ranges.items():
        step_sizes[var_name] = (max_val - min_val) / step
    iteration = 1
    while (step_sizes['p_k1_D1'] >= tolerance or step_sizes['p_k2_D1'] >= tolerance or
           step_sizes['z'] >= tolerance or step_sizes['D_k2'] >= tolerance or
           step_sizes['f_k1_uav'] >= 1e3 or step_sizes['f_k2_uav'] >= 1e3):
        print(f"====== 第{iteration}次搜索 "
              f" step_size为 : "
              f"{step_sizes['p_k1_D1'], step_sizes['p_k2_D1'], step_sizes['z'], step_sizes['D_k2'], step_sizes['f_k1_uav'], step_sizes['f_k2_uav']}======")
        iteration = iteration + 1
        # 生成当前迭代的搜索点
        p_k1_D1_vals = np.arange(current_ranges['p_k1_D1'][0], current_ranges['p_k1_D1'][1], step_sizes['p_k1_D1'])
        p_k2_D1_vals = np.arange(current_ranges['p_k2_D1'][0], current_ranges['p_k2_D1'][1], step_sizes['p_k2_D1'])
        z_vals = np.arange(current_ranges['z'][0], current_ranges['z'][1], step_sizes['z'])
        D_k2_vals = np.arange(current_ranges['D_k2'][0], current_ranges['D_k2'][1], step_sizes['D_k2'])
        f_k1_uav_vals = np.arange(current_ranges['f_k1_uav'][0], current_ranges['f_k1_uav'][1], step_sizes['f_k1_uav'])
        f_k2_uav_vals = np.arange(current_ranges['f_k2_uav'][0], current_ranges['f_k2_uav'][1], step_sizes['f_k2_uav'])

        # 当前搜索范围
        for p_k1_D1_val in p_k1_D1_vals:
            for p_k2_D1_val in p_k2_D1_vals:
                for z_val in z_vals:
                    for D_k2_val in D_k2_vals:
                        for f_k1_uav_val in f_k1_uav_vals:
                            for f_k2_uav_val in f_k2_uav_vals:
                                # 计算目标函数值
                                obj_value = 2 * D_k1 + D_k2_val + L_k1 * C / f_k1_uav_val + L_k2 * C / f_k2_uav_val

                                # 所有约束检查
                                if (D_k1 * p_k1_D1_val - E_max <= 0 and
                                        p_k1_D1_val - P_max <= 0 and
                                        p_k2_D1_val - P_max <= 0 and
                                        D_k1 + L_k1 * C / f_k1_uav_val - D_k1_max <= 0 and
                                        D_k1 + D_k2_val + L_k2 * C / f_k2_uav_val - D_k2_max <= 0 and
                                        f_k1_uav_val + f_k2_uav_val - f_uav_max <= 0 and
                                        kappa_uav * f_k1_uav_val ** 2 * C * L_k1 + kappa_uav * f_k2_uav_val ** 2 * C * L_k2 + (
                                                c1 * (uav_diff ** 3) / (tau ** 2)) + (
                                                c2 * (tau ** 2) / uav_diff) - E_uav_max <= 0 and
                                        L_k1 - D_k1 * B * np.log(
                                            g_k2 * p_k2_D1_val + g_k1 * p_k1_D1_val + sigma2) + D_k1 * B * np.log(
                                            g_k2 * p_k2_D1_val + sigma2) <= 0 and
                                        D_k1 * p_k2_D1_val + sigma2 / g_k2 * D_k2_val * np.exp(
                                            z_val) - sigma2 / g_k2 * D_k2_val - E_max <= 0 and
                                        z_val - np.log1p(g_k2 * P_max / sigma2) <= 0 and
                                        L_k2 - D_k1 * B * np.log(sigma2 + g_k2 * p_k2_D1_val) + D_k1 * B * np.log(
                                            sigma2) - B * D_k2_val * z_val <= 0
                                ):
                                    # 如果满足约束条件，比较目标函数值
                                    if obj_value < optimal_value:
                                        optimal_value = obj_value
                                        optimal_p_k1_D1 = p_k1_D1_val
                                        optimal_p_k2_D1 = p_k2_D1_val
                                        optimal_z = z_val
                                        optimal_D_k2 = D_k2_val
                                        optimal_f_k1_uav = f_k1_uav_val
                                        optimal_f_k2_uav = f_k2_uav_val
                                        print(f"  Satisfy constraints, current optimal value equals {optimal_value}")

        # 更新搜索范围（围绕当前最优解）
        # 当前最优解：optimal_p_k1_D1
        # 当前步长：step_sizes['p_k1_D1']
        # 新范围：[最优解 - 步长, 最优解 + 步长]
        current_ranges['p_k1_D1'] = (max(MIN_VAL, optimal_p_k1_D1 - step_sizes['p_k1_D1']),
                                     min(P_max, optimal_p_k1_D1 + step_sizes['p_k1_D1']))
        current_ranges['p_k2_D1'] = (max(MIN_VAL, optimal_p_k2_D1 - step_sizes['p_k2_D1']),
                                     min(P_max, optimal_p_k2_D1 + step_sizes['p_k2_D1']))
        current_ranges['z'] = (max(MIN_VAL, optimal_z - step_sizes['z']),
                               min(z_max, optimal_z + step_sizes['z']))
        current_ranges['D_k2'] = (max(MIN_VAL, optimal_D_k2 - step_sizes['D_k2']),
                                  min(D_k2_max, optimal_D_k2 + step_sizes['D_k2']))
        current_ranges['f_k1_uav'] = (max(MIN_VAL, optimal_f_k1_uav - step_sizes['f_k1_uav']),
                                      min(f_uav_max, optimal_f_k1_uav + step_sizes['f_k1_uav']))
        current_ranges['f_k2_uav'] = (max(MIN_VAL, optimal_f_k2_uav - step_sizes['f_k2_uav']),
                                      min(f_uav_max, optimal_f_k2_uav + step_sizes['f_k2_uav']))

        # 更新步长（缩小搜索范围）
        for var_name in step_sizes.keys():
            step_sizes[var_name] = (current_ranges[var_name][1] - current_ranges[var_name][0]) / step

    # 输出结果
    print(f"Case {case_id + 1}:------------ES--------------")
    print(f"  总时延: {optimal_value:.6f} 瓦特秒")
    print("最优 p_k1_D1:", optimal_p_k1_D1)
    print("最优 p_k2_D1:", optimal_p_k2_D1)
    print("最优 z:", optimal_z)
    print("最优 D_k2:", optimal_D_k2)
    print("最优 f_k1_uav:", optimal_f_k1_uav)
    print("最优 f_k2_uav:", optimal_f_k2_uav)

    return optimal_value


def user2_lager_gain_ES(L_k1, L_k2, tau, D_k1, g_k1, g_k2, uav_pos_pre, uav_pos, case_id):
    # 类型转换
    uav_pos = np.array(uav_pos, dtype=float)
    uav_pos_pre = np.array(uav_pos_pre, dtype=float)

    # 参数定义
    B = 1e6  # 信道带宽 (单位：Hz)
    sigma2 = 3.98e-21 * B  # 噪声功率，单位瓦特
    P_max = 0.1995  # 用户最大功率（瓦特）23dB
    C = 800

    # UAV参数
    c1 = 0.00614
    c2 = 15.976
    kappa_uav = 1e-28  # UAV计算能量系数
    f_uav_max = 3e9  # UAV的最大计算频率
    E_uav_max = 5

    # 用户
    E_max = 0.2
    D_k1_max = np.random.uniform(tau - 0.3, tau)  # 每个用户最大可容忍时延从[tau-0.3,tau]之间随机生成
    D_k2_max = np.random.uniform(D_k1_max, tau)  # D_k2_max 必须大于 D_k1_max 但不能超过tau

    # UAV前后位置差
    uav_diff = np.linalg.norm(uav_pos - uav_pos_pre)

    # 给定上届
    z_max = log1p(g_k2 * P_max / sigma2)

    # 初始化每个变量的搜索范围
    # 1. 数值安全下界（防止 log(0) 或 除以 0）
    MIN_VAL = 1e-20
    # 2. 搜索终止精度（控制循环次数）
    tolerance = 1e-6
    original_ranges = {
        'p_k1_D1': (MIN_VAL, P_max),
        'p_k2_D1': (MIN_VAL, P_max),
        'z': (MIN_VAL, z_max),
        'D_k2': (MIN_VAL, D_k2_max),
        'f_k1_uav': (MIN_VAL, f_uav_max),
        'f_k2_uav': (MIN_VAL, f_uav_max)
    }

    # 初始化最优解
    optimal_value = float('inf')
    optimal_p_k1_D1 = (original_ranges['p_k1_D1'][0] + original_ranges['p_k1_D1'][1]) / 2
    optimal_p_k2_D1 = (original_ranges['p_k2_D1'][0] + original_ranges['p_k2_D1'][1]) / 2
    optimal_z = (original_ranges['z'][0] + original_ranges['z'][1]) / 2
    optimal_D_k2 = (original_ranges['D_k2'][0] + original_ranges['D_k2'][1]) / 2
    optimal_f_k1_uav = (original_ranges['f_k1_uav'][0] + original_ranges['f_k1_uav'][1]) / 2
    optimal_f_k2_uav = (original_ranges['f_k2_uav'][0] + original_ranges['f_k2_uav'][1]) / 2

    # 初始化搜索范围和步长
    step = 20  # 每轮搜索的步数

    # 初始搜索范围
    current_ranges = {
        'p_k1_D1': original_ranges['p_k1_D1'],
        'p_k2_D1': original_ranges['p_k2_D1'],
        'z': original_ranges['z'],
        'D_k2': original_ranges['D_k2'],
        'f_k1_uav': original_ranges['f_k1_uav'],
        'f_k2_uav': original_ranges['f_k2_uav']
    }

    # 计算初始步长
    step_sizes = {}
    for var_name, (min_val, max_val) in current_ranges.items():
        step_sizes[var_name] = (max_val - min_val) / step
    iteration = 1
    while (step_sizes['p_k1_D1'] >= tolerance or step_sizes['p_k2_D1'] >= tolerance or
           step_sizes['z'] >= tolerance or step_sizes['D_k2'] >= tolerance or
           step_sizes['f_k1_uav'] >= 1e3 or step_sizes['f_k2_uav'] >= 1e3):
        print(f"====== 第{iteration}次搜索 "
              f" step_size为 : "
              f"{step_sizes['p_k1_D1'], step_sizes['p_k2_D1'], step_sizes['z'], step_sizes['D_k2'], step_sizes['f_k1_uav'], step_sizes['f_k2_uav']}======")
        # 生成当前迭代的搜索点
        p_k1_D1_vals = np.arange(current_ranges['p_k1_D1'][0], current_ranges['p_k1_D1'][1], step_sizes['p_k1_D1'])
        p_k2_D1_vals = np.arange(current_ranges['p_k2_D1'][0], current_ranges['p_k2_D1'][1], step_sizes['p_k2_D1'])
        z_vals = np.arange(current_ranges['z'][0], current_ranges['z'][1], step_sizes['z'])
        D_k2_vals = np.arange(current_ranges['D_k2'][0], current_ranges['D_k2'][1], step_sizes['D_k2'])
        f_k1_uav_vals = np.arange(current_ranges['f_k1_uav'][0], current_ranges['f_k1_uav'][1], step_sizes['f_k1_uav'])
        f_k2_uav_vals = np.arange(current_ranges['f_k2_uav'][0], current_ranges['f_k2_uav'][1], step_sizes['f_k2_uav'])

        # 当前搜索范围
        for p_k1_D1_val in p_k1_D1_vals:
            for p_k2_D1_val in p_k2_D1_vals:
                for z_val in z_vals:
                    for D_k2_val in D_k2_vals:
                        for f_k1_uav_val in f_k1_uav_vals:
                            for f_k2_uav_val in f_k2_uav_vals:
                                # 计算目标函数值
                                obj_value = 2 * D_k1 + D_k2_val + L_k1 * C / f_k1_uav_val + L_k2 * C / f_k2_uav_val

                                # 所有约束检查
                                if (D_k1 * p_k1_D1_val - E_max <= 0 and
                                        p_k1_D1_val - P_max <= 0 and
                                        p_k2_D1_val - P_max <= 0 and
                                        D_k1 + L_k1 * C / f_k1_uav_val - D_k1_max <= 0 and
                                        D_k1 + D_k2_val + L_k2 * C / f_k2_uav_val - D_k2_max <= 0 and
                                        f_k1_uav_val + f_k2_uav_val - f_uav_max <= 0 and
                                        kappa_uav * f_k1_uav_val ** 2 * C * L_k1 + kappa_uav * f_k2_uav_val ** 2 * C * L_k2 + (
                                                c1 * (uav_diff ** 3) / (tau ** 2)) + (
                                                c2 * (tau ** 2) / uav_diff) - E_uav_max <= 0 and
                                        L_k1 - D_k1 * B * np.log(sigma2 + g_k1 * p_k1_D1_val) + D_k1 * B * np.log(
                                            sigma2) <= 0 and
                                        D_k1 * p_k2_D1_val + sigma2 / g_k2 * D_k2_val * np.exp(
                                            z_val) - sigma2 / g_k2 * D_k2_val - E_max <= 0 and
                                        z_val - np.log1p(g_k2 * P_max / sigma2) <= 0 and
                                        L_k2 - D_k1 * B * np.log(
                                            g_k1 * p_k1_D1_val + g_k2 * p_k2_D1_val + sigma2) + D_k1 * B * np.log(
                                            g_k1 * p_k1_D1_val + sigma2) - B * D_k2_val * z_val <= 0
                                ):
                                    # 如果满足约束条件，比较目标函数值
                                    if obj_value < optimal_value:
                                        optimal_value = obj_value
                                        optimal_p_k1_D1 = p_k1_D1_val
                                        optimal_p_k2_D1 = p_k2_D1_val
                                        optimal_z = z_val
                                        optimal_D_k2 = D_k2_val
                                        optimal_f_k1_uav = f_k1_uav_val
                                        optimal_f_k2_uav = f_k2_uav_val
                                        print(f"  Satisfy constraints, current optimal value equals {optimal_value}")

        # 更新搜索范围（围绕当前最优解）
        current_ranges['p_k1_D1'] = (max(MIN_VAL, optimal_p_k1_D1 - step_sizes['p_k1_D1']),
                                     min(P_max, optimal_p_k1_D1 + step_sizes['p_k1_D1']))
        current_ranges['p_k2_D1'] = (max(MIN_VAL, optimal_p_k2_D1 - step_sizes['p_k2_D1']),
                                     min(P_max, optimal_p_k2_D1 + step_sizes['p_k2_D1']))
        current_ranges['z'] = (max(MIN_VAL, optimal_z - step_sizes['z']),
                               min(z_max, optimal_z + step_sizes['z']))
        current_ranges['D_k2'] = (max(MIN_VAL, optimal_D_k2 - step_sizes['D_k2']),
                                  min(D_k2_max, optimal_D_k2 + step_sizes['D_k2']))
        current_ranges['f_k1_uav'] = (max(MIN_VAL, optimal_f_k1_uav - step_sizes['f_k1_uav']),
                                      min(f_uav_max, optimal_f_k1_uav + step_sizes['f_k1_uav']))
        current_ranges['f_k2_uav'] = (max(MIN_VAL, optimal_f_k2_uav - step_sizes['f_k2_uav']),
                                      min(f_uav_max, optimal_f_k2_uav + step_sizes['f_k2_uav']))

        # 更新步长（缩小搜索范围）
        for var_name in step_sizes.keys():
            step_sizes[var_name] = (current_ranges[var_name][1] - current_ranges[var_name][0]) / step

    # 输出结果
    print(f"Case {case_id + 1}:------------ES--------------")
    print(f"  总时延: {optimal_value:.6f} 瓦特秒")
    print("最优 p_k1_D1:", optimal_p_k1_D1)
    print("最优 p_k2_D1:", optimal_p_k2_D1)
    print("最优 z:", optimal_z)
    print("最优 D_k2:", optimal_D_k2)
    print("最优 f_k1_uav:", optimal_f_k1_uav)
    print("最优 f_k2_uav:", optimal_f_k2_uav)

    return optimal_value

if __name__ == "__main__":
    # 模拟参数
    num_cases = 1  # 随机案例数量
    iteration_result = []  # 存储迭代次数
    delay_result = []  # 存储每个案例的总时延结果

    np.random.seed(42)  # 确保仿真结果

    # 坐标参数
    center = (0, 100)  # 圆心
    radius = 60        # 半径
    uav_pos_pre = [0,0,50]
    ris_pos = [0,0,20]

    D_k1 = 0.2  # 初始化第一段传输时间时长
    num_users = 2  # 只考虑一个NOMA组
    for case_id in range(num_cases):
        try:
            # 随机生成 [100,200] knats 数据 [500,650] ms 的任务持续时间
            tau = (0.65 - 0.5) * np.random.rand() + 0.5
            uav_pos = [0, 0 + 10 * tau, 50]  # UAV最大移动速度为10m/s 假设沿着y轴移动
            L_k1 = ((200 - 100) * np.random.rand() + 100) * 1000
            L_k2 = ((200 - 100) * np.random.rand() + 100) * 1000

            # 用户位置
            angles = np.random.uniform(0, 2 * np.pi, num_users)
            r = radius * np.sqrt(np.random.uniform(0, 1, num_users))
            users_x = center[0] + r * np.cos(angles)
            users_y = center[1] + r * np.sin(angles)
            users_z = np.zeros(num_users)
            users_pos = np.stack([users_x, users_y, users_z], axis=1)

            # 生成等效信道
            g_k1, g_k2 = generate_equivalent_channel(uav_pos, ris_pos, users_pos)

            # 判断情况
            if g_k1 >= g_k2:
                optimal_value = user1_lager_gain_ES(
                    L_k1, L_k2, tau, D_k1, g_k1 * 1e15, g_k2 * 1e20, uav_pos_pre, uav_pos, case_id
                )
            else:
                optimal_value = user2_lager_gain_ES(
                    L_k1, L_k2, tau, D_k1, g_k1 * 1e15, g_k2 * 1e20, uav_pos_pre, uav_pos, case_id
                )

            # 保存结果
            delay_result.append(optimal_value)

        except Exception as e:
            print(f"案例 {case_id} 出错，跳过。错误信息: {e}")
            continue

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(delay_result) + 1), delay_result, marker='o', linestyle='-')
    plt.xlabel("Index of cases", fontsize=12)
    plt.ylabel("Total delay (s)", fontsize=12)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.show()