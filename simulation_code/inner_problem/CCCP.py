import os
import time

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
from equivalent_channel import generate_equivalent_channel
import warnings
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('error', category=RuntimeWarning)  # 把 RuntimeWarning 当作异常处理
GRID_COLOR = '#E0E0E0'  # 柔和的网格线颜色


def user1_lager_gain_CCCP(L_k1, L_k2, tau, D_k1, g_k1, g_k2,
                          uav_pos_pre, uav_pos, case_id, id_noma_group):
    """
    当NOMA组中用户1的信道增益大于用户2时执行该函数
    :param L_k1: 用户1的总任务量
    :param L_k2: 用户2的总任务量
    :param tau: 时隙持续时长
    :param D_k1: 第一段卸载时间
    :param g_k1: 用户1至UAV的信道增益
    :param g_k2: 用户2至UAV的信道增益
    :return:
    """
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
    f_norm = 1e9
    kappa_uav = 1e-28  # UAV计算能量系数
    f_uav_max = 3e9  # UAV的最大计算频率
    f_uav_max_norm = f_uav_max / f_norm
    E_uav_max = 5

    # 用户
    E_max = 0.2
    # NOTE - 这里 D_k1_max 肯定是要小于 D_k2_max
    D_k1_max = np.random.uniform(tau - 0.2, tau)  # 每个用户最大可容忍时延从[tau-0.2,tau]之间随机生成
    D_k2_max = np.random.uniform(D_k1_max, tau)  # D_k2_max 必须大于 D_k1_max 但不能超过tau

    max_iterations = 10
    tolerance = 3e-3  # 收敛阈值

    # UAV前后迭代位置差
    uav_diff = np.linalg.norm(uav_pos - uav_pos_pre)

    # 最优目标函数值初始化
    optimal_value = float('inf')
    pre_objective_value = float('inf')  # 设置一个初始的前一次目标函数值 对于时延而言足够大就行

    # 声明优化变量
    p_k1_D1_var = cp.Variable(nonneg=True)
    p_k2_D1_var = cp.Variable(nonneg=True)
    z_var = cp.Variable(nonneg=True)
    D_k2_var = cp.Variable(nonneg=True)
    f_k1_uav_var = cp.Variable(nonneg=True)
    f_k2_uav_var = cp.Variable(nonneg=True)
    # ---------- 定义参数 (hat 值) ----------
    hat_p_k2_D1_param = cp.Parameter(nonneg=True, value=0.05)
    hat_z_param = cp.Parameter(nonneg=True, value=2e-13)
    hat_D_k2_param = cp.Parameter(nonneg=True, value=2e-15)
    hat_f_k1_uav_param = cp.Parameter(nonneg=True, value=2e8)
    hat_f_k2_uav_param = cp.Parameter(nonneg=True, value=2e8)

    # 目标函数
    objective = cp.Minimize(
        2 * D_k1 + D_k2_var + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var) + (L_k2 * C / f_norm) * cp.inv_pos(
            f_k2_uav_var)
    )

    # 约束条件
    constraints = []

    # 27d
    constraints.append(
        D_k1 * p_k1_D1_var - E_max <= 0
    )

    # 27e
    constraints.append(
        p_k1_D1_var - P_max <= 0
    )

    # 27f
    constraints.append(
        p_k2_D1_var - P_max <= 0
    )

    # 27h
    constraints.append(
        D_k1 + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var) - D_k1_max <= 0
    )

    # 27i
    constraints.append(
        D_k1 + D_k2_var + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var) - D_k2_max <= 0
    )

    # 27j
    constraints.append(
        f_k1_uav_var + f_k2_uav_var - f_uav_max_norm <= 0
    )

    # 27k
    constraints.append(
        kappa_uav * ((hat_f_k1_uav_param * f_norm) ** 2 + 2 * hat_f_k1_uav_param * f_norm * (
                f_k1_uav_var * f_norm - hat_f_k1_uav_param * f_norm)) * C * L_k1
        + kappa_uav * ((hat_f_k2_uav_param * f_norm) ** 2 + 2 * hat_f_k2_uav_param * f_norm * (
                f_k2_uav_var * f_norm - hat_f_k2_uav_param * f_norm)) * C * L_k2
        + (c1 * (uav_diff ** 3) / (tau ** 2)) + (c2 * (tau ** 2) / uav_diff) - E_uav_max <= 0
    )

    # 31
    constraints.append(
        z_var - cp.log(1 + g_k2 * P_max / sigma2) <= 0
    )

    # 46
    constraints.append(
        D_k1 * p_k2_D1_var - sigma2 / g_k2 * D_k2_var - E_max
        + sigma2 / (2 * g_k2) * (D_k2_var + cp.exp(z_var)) ** 2
        - sigma2 / (2 * g_k2) * (hat_D_k2_param ** 2 + (cp.exp(hat_z_param)) ** 2
                                 + 2 * hat_D_k2_param * (D_k2_var - hat_D_k2_param)
                                 + 2 * cp.exp(2 * hat_z_param) * (z_var - hat_z_param)) <= 0
    )

    # 47
    constraints.append(
        L_k1 - D_k1 * B * cp.log(g_k2 * p_k2_D1_var + g_k1 * p_k1_D1_var + sigma2)
        + D_k1 * B * cp.log(g_k2 * hat_p_k2_D1_param + sigma2)
        + D_k1 * B * g_k2 / (g_k2 * hat_p_k2_D1_param + sigma2) * (p_k2_D1_var - hat_p_k2_D1_param) <= 0
    )

    # 48
    constraints.append(
        L_k2 - D_k1 * B * cp.log(sigma2 + g_k2 * p_k2_D1_var)
        + D_k1 * B * cp.log(sigma2) + B / 2 * (D_k2_var ** 2 + z_var ** 2)
        - B / 2 * (hat_D_k2_param + hat_z_param) ** 2
        - B * (hat_D_k2_param + hat_z_param) * (D_k2_var - hat_D_k2_param + z_var - hat_z_param) <= 0
    )

    # 变量约束
    constraints += [
        p_k1_D1_var >= 0,
        p_k2_D1_var >= 0,
        z_var >= 0,
        D_k2_var >= 0,
        f_k1_uav_var >= 0,
        f_k2_uav_var >= 0
    ]

    # 开始迭代
    print("-------------------------------------------------------- ")
    print(f"-------- {case_id + 1}次案例 -> 第{id_noma_group + 1}个NOMA组 -> CCCP算法开始迭代 -------- ")
    print("-------------------------------------------------------- ")
    for iteration in range(max_iterations):
        # 创建问题
        problem = cp.Problem(objective, constraints)
        # 求解优化问题
        problem.solve(solver=cp.SCS, warm_start=True)
        # print("status:", problem.status)

        # 计算当前目标函数值
        current_objective_value = problem.value

        # 检查收敛性
        if abs(current_objective_value - pre_objective_value) < tolerance:
            print(f"CCCP算法迭代过程收敛在第 {iteration + 1} 轮")
            iteration_nums = iteration + 1
            optimal_value = current_objective_value
            # 输出当前优化结果和目标函数值
            # print(f"p_k1_D1 = {p_k1_D1_opt}")
            # print(f"p_k2_D1 = {p_k2_D1_opt}")
            # print(f"z_var = {z_var_opt}")
            # print(f"D_k2_var = {D_k2_var_opt}")
            # print(f"f_k1_uav_var = {f_k1_uav_var_opt * f_norm}")
            # print(f"f_k2_uav_var = {f_k2_uav_var_opt * f_norm}")
            print(f"最优目标函数值: {optimal_value}")
            break

        # 更新为当前CCCP求解出来的最优解
        hat_p_k2_D1_param.value = p_k2_D1_var.value
        hat_z_param.value = z_var.value
        hat_D_k2_param.value = D_k2_var.value
        hat_f_k1_uav_param.value = f_k1_uav_var.value
        hat_f_k2_uav_param.value = f_k2_uav_var.value

        # 更新最优目标函数值
        pre_objective_value = current_objective_value

    return optimal_value


def user2_lager_gain_CCCP(L_k1, L_k2, tau, D_k1, g_k1, g_k2,
                          uav_pos_pre, uav_pos, case_id, id_noma_group):
    """
    当NOMA组中用户2的信道增益大于用户1时执行该函数
    :param L_k1: 用户1的总任务量
    :param L_k2: 用户2的总任务量
    :param tau: 时隙持续时长
    :param D_k1: 第一段卸载时间
    :param g_k1: 用户1至UAV的信道增益
    :param g_k2: 用户2至UAV的信道增益
    :return:
    """
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
    f_norm = 1e9
    kappa_uav = 1e-28  # UAV计算能量系数
    f_uav_max = 3e9  # 保留原来的真实最大频率（Hz）
    f_uav_max_norm = f_uav_max / f_norm
    E_uav_max = 5

    # 用户
    E_max = 0.2
    D_k1_max = np.random.uniform(tau - 0.2, tau)  # 每个用户最大可容忍时延从[tau-0.3,tau]之间随机生成
    D_k2_max = np.random.uniform(D_k1_max, tau)  # D_k2_max 必须大于 D_k1_max 但不能超过tau

    # UAV前后迭代位置差
    uav_diff = np.linalg.norm(uav_pos - uav_pos_pre)

    # CCCP算法的最大迭代次数
    max_iterations = 10
    tolerance = 3e-3  # 收敛阈值

    # 最优目标函数值初始化
    optimal_value = float('inf')
    pre_objective_value = float('inf')  # 设置一个初始的前一次目标函数值 对于能耗而言足够大就行
    # 声明优化变量
    p_k1_D1_var = cp.Variable(nonneg=True)
    p_k2_D1_var = cp.Variable(nonneg=True)
    z_var = cp.Variable(nonneg=True)
    D_k2_var = cp.Variable(nonneg=True)
    f_k1_uav_var = cp.Variable(nonneg=True)
    f_k2_uav_var = cp.Variable(nonneg=True)
    # ---------- 定义参数 (hat 值) ----------
    hat_p_k1_D1_param = cp.Parameter(nonneg=True, value=0.12)
    hat_z_param = cp.Parameter(nonneg=True, value=2e-13)
    hat_D_k2_param = cp.Parameter(nonneg=True, value=2e-15)
    hat_f_k1_uav_param = cp.Parameter(nonneg=True, value=2e8)
    hat_f_k2_uav_param = cp.Parameter(nonneg=True, value=2e8)

    # 目标函数
    objective = cp.Minimize(
        2 * D_k1 + D_k2_var + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var) + (L_k2 * C / f_norm) * cp.inv_pos(
            f_k2_uav_var)
    )

    # 约束条件
    constraints = []

    # 27d
    constraints.append(
        D_k1 * p_k1_D1_var - E_max <= 0
    )

    # 27e
    constraints.append(
        p_k1_D1_var - P_max <= 0
    )

    # 27f
    constraints.append(
        p_k2_D1_var - P_max <= 0
    )

    # 27h
    constraints.append(
        D_k1 + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var) - D_k1_max <= 0
    )

    # 27i
    constraints.append(
        D_k1 + D_k2_var + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var) - D_k2_max <= 0
    )

    # 27j
    constraints.append(
        f_k1_uav_var + f_k2_uav_var - f_uav_max_norm <= 0
    )

    # 27k
    constraints.append(
        kappa_uav * ((hat_f_k1_uav_param * f_norm) ** 2 + 2 * hat_f_k1_uav_param * f_norm * (
                    f_k1_uav_var * f_norm - hat_f_k1_uav_param * f_norm)) * C * L_k1
        + kappa_uav * ((hat_f_k2_uav_param * f_norm) ** 2 + 2 * hat_f_k2_uav_param * f_norm * (
                    f_k2_uav_var * f_norm - hat_f_k2_uav_param * f_norm)) * C * L_k2
        + (c1 * (uav_diff ** 3) / (tau ** 2)) + (c2 * (tau ** 2) / uav_diff) - E_uav_max <= 0
    )

    # 31
    constraints.append(
        z_var - cp.log(1 + g_k2 * P_max / sigma2) <= 0
    )

    # 44
    constraints.append(
        L_k1 - D_k1 * B * cp.log(sigma2 + g_k1 * p_k1_D1_var) + D_k1 * B * cp.log(sigma2) <= 0
    )

    # 46
    constraints.append(
        D_k1 * p_k2_D1_var - sigma2 / g_k2 * D_k2_var - E_max
        + sigma2 / (2 * g_k2) * (D_k2_var + cp.exp(z_var)) ** 2
        - sigma2 / (2 * g_k2) * (hat_D_k2_param ** 2 + (cp.exp(hat_z_param)) ** 2
                                 + 2 * hat_D_k2_param * (D_k2_var - hat_D_k2_param)
                                 + 2 * cp.exp(2 * hat_z_param) * (z_var - hat_z_param)) <= 0
    )

    # 49
    constraints.append(
        L_k2 - D_k1 * B * cp.log(g_k1 * p_k1_D1_var + g_k2 * p_k2_D1_var + sigma2)
        + B / 2 * (D_k2_var ** 2 + z_var ** 2)
        - B / 2 * (hat_D_k2_param + hat_z_param) ** 2
        + D_k1 * B * cp.log(g_k1 * hat_p_k1_D1_param + sigma2)
        + D_k1 * B * g_k1 / (g_k1 * hat_p_k1_D1_param + sigma2) * (p_k1_D1_var - hat_p_k1_D1_param)
        - B * (hat_D_k2_param + hat_z_param) * (D_k2_var - hat_D_k2_param + z_var - hat_z_param) <= 0
    )

    # 变量约束
    constraints += [
        p_k1_D1_var >= 0,
        p_k2_D1_var >= 0,
        z_var >= 0,
        D_k2_var >= 0,
        f_k1_uav_var >= 0,
        f_k2_uav_var >= 0
    ]

    # 开始迭代
    iteration_nums = 0
    print("-------------------------------------------------------- ")
    print(f"-------- {case_id + 1}次案例 -> 第{id_noma_group + 1}个NOMA组 -> CCCP算法开始迭代 -------- ")
    print("-------------------------------------------------------- ")
    for iteration in range(max_iterations):
        # 创建问题
        problem = cp.Problem(objective, constraints)
        # 求解优化问题
        problem.solve(solver=cp.SCS, warm_start=True)
        # print("status:", problem.status)

        # 计算当前目标函数值
        current_objective_value = problem.value

        # 检查收敛性
        if abs(current_objective_value - pre_objective_value) < tolerance:
            print(f"CCCP算法迭代过程收敛在第 {iteration + 1} 轮")
            iteration_nums = iteration + 1
            optimal_value = current_objective_value
            # 输出当前优化结果和目标函数值
            # print(f"p_k1_D1 = {p_k1_D1_var_opt}")
            # print(f"p_k2_D1 = {p_k2_D1_var_opt}")
            # print(f"z_var = {z_var_opt}")
            # print(f"D_k2_var = {D_k2_var_opt}")
            # print(f"f_k1_uav_var = {f_k1_uav_var_opt * f_norm}")
            # print(f"f_k2_uav_var = {f_k2_uav_var_opt * f_norm}")
            print(f"最优目标函数值: {optimal_value}")
            break

        # 更新为当前CCCP求解出来的最优解
        hat_p_k1_D1_param.value = p_k1_D1_var.value
        hat_z_param.value = z_var.value
        hat_D_k2_param.value = D_k2_var.value
        hat_f_k1_uav_param.value = f_k1_uav_var.value
        hat_f_k2_uav_param.value = f_k2_uav_var.value

        # 更新最优目标函数值
        pre_objective_value = current_objective_value

    return optimal_value


# 并行执行 CCCP
def run_one_group(id_noma_group, case_id, D_k1, uav_pos_pre, ris_pos, center, radius, num_users):
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

    # 判断情况并调用对应CCCP
    if g_k1 >= g_k2:
        return user1_lager_gain_CCCP(L_k1, L_k2, tau, D_k1,
                                     g_k1 * 1e15, g_k2 * 1e15, uav_pos_pre, uav_pos, case_id, id_noma_group)
    else:
        return user2_lager_gain_CCCP(L_k1, L_k2, tau, D_k1,
                                     g_k1 * 1e15, g_k2 * 1e15, uav_pos_pre, uav_pos, case_id, id_noma_group)


def main():
    # 模拟参数
    num_cases = 30  # 随机案例数量
    iteration_result = []  # 存储迭代次数
    delay_result = []  # 存储每个案例的总时延结果
    time_result = []  # 存储每个案例的CCCP执行时长

    np.random.seed(42)  # 确保仿真结果可复现

    # 坐标参数
    center = (0, 100)  # 圆心
    radius = 60  # 半径
    uav_pos_pre = [0, 0, 50]
    ris_pos = [0, 0, 20]

    D_k1 = 0.2  # 初始化第一段传输时间时长
    num_users = 2  # NOMA组中两个用户 一定是2个 多个的话建模因为建模是按照2个用户建模的 上面的建模就错了
    total_noma_group = 1  # total_noma_group个NOMA组 每个NOMA组中有num_users个用户
    case_id = 0  # 初始案例编号

    # 在最外层创建进程池
    with ProcessPoolExecutor(max_workers=min(total_noma_group, os.cpu_count())) as executor:
        while case_id < num_cases:
            start_time = time.time()
            try:
                futures = [executor.submit(run_one_group, id_noma_group, case_id, D_k1,
                                           uav_pos_pre, ris_pos, center, radius, num_users)
                           for id_noma_group in range(total_noma_group)]
                results = [f.result() for f in futures]

                optimal_value = sum(results)
                if np.isinf(optimal_value) or np.isnan(optimal_value):
                    raise RuntimeError("CCCP算法返回 inf 或 nan 求解失败")

                elapsed_time = time.time() - start_time

                delay_result.append(optimal_value)
                time_result.append(elapsed_time)
                case_id += 1

            except Exception as e:
                print(f"-------- 案例 {case_id + 1} 出错跳过  重新生成 --------")
                print(f" Error message:{e}")

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(delay_result) + 1), delay_result, marker='o', linestyle='-')
    plt.xlabel("Index of cases", fontsize=24)
    plt.ylabel("Total delay (s)", fontsize=24)
    plt.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8, zorder=1)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True)
    plt.show()

    # 构造结果字典  同样跳过第一个构造图函数案例
    results_dict = {
        'ID': list(range(1, num_cases + 1)),
        'Delay': delay_result,
        'Running_time': time_result
    }

    # 转换为DataFrame
    results_df = pd.DataFrame(results_dict)

    # 保存为Excel文件
    results_df.to_csv(f'CCCP_group_NOMA_{total_noma_group}_parallel.csv', index=False)
    print(f"✅ CCCP_group_NOMA_{total_noma_group} excel output.xlsx")


if __name__ == "__main__":
    main()
