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
    B = 1e6
    sigma2 = 3.98e-21 * B
    P_max = 0.1995
    C = 800

    # UAV参数
    c1 = 0.00614
    c2 = 15.976
    kappa_uav = 1e-28
    f_uav_max = 3e9
    E_uav_max = 5

    # 用户
    E_max = 0.2
    D_k1_max = np.random.uniform(tau - 0.3, tau)
    D_k2_max = np.random.uniform(D_k1_max, tau)

    uav_diff = np.linalg.norm(uav_pos - uav_pos_pre)
    z_max = log1p(g_k2 * P_max / sigma2)

    MIN_VAL = 1e-20
    original_ranges = {
        'p_k1_D1': (MIN_VAL, P_max),
        'p_k2_D1': (MIN_VAL, P_max),
        'z': (MIN_VAL, z_max),
        'D_k2': (MIN_VAL, D_k2_max),
        'f_k1_uav': (MIN_VAL, f_uav_max),
        'f_k2_uav': (MIN_VAL, f_uav_max)
    }

    # 初始最优
    optimal_value = float('inf')
    # coarse default initial guess (中点)
    optimal_p_k1_D1 = (original_ranges['p_k1_D1'][0] + original_ranges['p_k1_D1'][1]) / 2
    optimal_p_k2_D1 = (original_ranges['p_k2_D1'][0] + original_ranges['p_k2_D1'][1]) / 2
    optimal_z = (original_ranges['z'][0] + original_ranges['z'][1]) / 2
    optimal_D_k2 = (original_ranges['D_k2'][0] + original_ranges['D_k2'][1]) / 2
    optimal_f_k1_uav = (original_ranges['f_k1_uav'][0] + original_ranges['f_k1_uav'][1]) / 2
    optimal_f_k2_uav = (original_ranges['f_k2_uav'][0] + original_ranges['f_k2_uav'][1]) / 2

    # ---------------------------
    # 论文指定的 coarse 步长（直接使用）
    # ---------------------------
    coarse_steps = {
        'p': 0.01,            # transmit power (W) for p_k1_D1 and p_k2_D1
        'D_k2': 0.02,         # transmission time step (s)
        'z': 0.1,
        'f': 0.05e9           # frequency step: 0.05 GHz -> 0.05e9 Hz
    }

    # ---------- COARSE SEARCH ----------
    print("====== Starting coarse search ======")
    p_vals = np.arange(original_ranges['p_k1_D1'][0], original_ranges['p_k1_D1'][1], coarse_steps['p'])
    p2_vals = np.arange(original_ranges['p_k2_D1'][0], original_ranges['p_k2_D1'][1], coarse_steps['p'])
    z_vals = np.arange(original_ranges['z'][0], original_ranges['z'][1], coarse_steps['z'])
    D_k2_vals = np.arange(original_ranges['D_k2'][0], original_ranges['D_k2'][1], coarse_steps['D_k2'])
    f1_vals = np.arange(original_ranges['f_k1_uav'][0], original_ranges['f_k1_uav'][1], coarse_steps['f'])
    f2_vals = np.arange(original_ranges['f_k2_uav'][0], original_ranges['f_k2_uav'][1], coarse_steps['f'])

    for p_k1_D1_val in p_vals:
        for p_k2_D1_val in p2_vals:
            for z_val in z_vals:
                for D_k2_val in D_k2_vals:
                    for f_k1_uav_val in f1_vals:
                        for f_k2_uav_val in f2_vals:
                            # 目标函数
                            obj_value = 2 * D_k1 + D_k2_val + L_k1 * C / f_k1_uav_val + L_k2 * C / f_k2_uav_val

                            # 约束检查（保持与你原代码一致）
                            if (D_k1 * p_k1_D1_val - E_max <= 0 and
                                    p_k1_D1_val - P_max <= 0 and
                                    p_k2_D1_val - P_max <= 0 and
                                    D_k1 + L_k1 * C / f_k1_uav_val - D_k1_max <= 0 and
                                    D_k1 + D_k2_val + L_k2 * C / f_k2_uav_val - D_k2_max <= 0 and
                                    f_k1_uav_val + f_k2_uav_val - f_uav_max <= 0 and
                                    kappa_uav * f_k1_uav_val ** 2 * C * L_k1 + kappa_uav * f_k2_uav_val ** 2 * C * L_k2 + (
                                            c1 * (uav_diff ** 3) / (tau ** 2)) + (
                                            c2 * (tau ** 2) / uav_diff) - E_uav_max <= 0 and
                                    L_k1 - D_k1 * B * np.log(g_k2 * p_k2_D1_val + g_k1 * p_k1_D1_val + sigma2) + D_k1 * B * np.log(g_k2 * p_k2_D1_val + sigma2) <= 0 and
                                    D_k1 * p_k2_D1_val + sigma2 / g_k2 * D_k2_val * np.exp(z_val) - sigma2 / g_k2 * D_k2_val - E_max <= 0 and
                                    z_val - np.log1p(g_k2 * P_max / sigma2) <= 0 and
                                    L_k2 - D_k1 * B * np.log(sigma2 + g_k2 * p_k2_D1_val) + D_k1 * B * np.log(sigma2) - B * D_k2_val * z_val <= 0
                            ):
                                if obj_value < optimal_value:
                                    optimal_value = obj_value
                                    optimal_p_k1_D1 = p_k1_D1_val
                                    optimal_p_k2_D1 = p_k2_D1_val
                                    optimal_z = z_val
                                    optimal_D_k2 = D_k2_val
                                    optimal_f_k1_uav = f_k1_uav_val
                                    optimal_f_k2_uav = f_k2_uav_val

    print(" Coarse-optimal found:", optimal_value)

    # ---------- FINE SEARCH ----------
    fine_steps = {
        'p': coarse_steps['p'] / 10.0,
        'D_k2': coarse_steps['D_k2'] / 10.0,
        'z': coarse_steps['z'] / 10.0,
        'f': coarse_steps['f'] / 10.0
    }

    # fine range: coarse_opt +/- 10 * fine_step  (which equals +/- coarse_step)
    def clamp(a, lo, hi):
        return max(lo, min(hi, a))

    fine_ranges = {
        'p_k1_D1': (clamp(optimal_p_k1_D1 - coarse_steps['p'], MIN_VAL, P_max),
                    clamp(optimal_p_k1_D1 + coarse_steps['p'], MIN_VAL, P_max)),
        'p_k2_D1': (clamp(optimal_p_k2_D1 - coarse_steps['p'], MIN_VAL, P_max),
                    clamp(optimal_p_k2_D1 + coarse_steps['p'], MIN_VAL, P_max)),
        'z': (clamp(optimal_z - coarse_steps['z'], MIN_VAL, z_max),
              clamp(optimal_z + coarse_steps['z'], MIN_VAL, z_max)),
        'D_k2': (clamp(optimal_D_k2 - coarse_steps['D_k2'], MIN_VAL, D_k2_max),
                 clamp(optimal_D_k2 + coarse_steps['D_k2'], MIN_VAL, D_k2_max)),
        'f_k1_uav': (clamp(optimal_f_k1_uav - coarse_steps['f'], MIN_VAL, f_uav_max),
                     clamp(optimal_f_k1_uav + coarse_steps['f'], MIN_VAL, f_uav_max)),
        'f_k2_uav': (clamp(optimal_f_k2_uav - coarse_steps['f'], MIN_VAL, f_uav_max),
                     clamp(optimal_f_k2_uav + coarse_steps['f'], MIN_VAL, f_uav_max))
    }

    print("====== Starting fine search around coarse-optimal (one pass) ======")
    # reset optimal for fine pass (keep global best)
    # (we keep the previous optimal_value so fine search can improve it)
    p_vals = np.arange(fine_ranges['p_k1_D1'][0], fine_ranges['p_k1_D1'][1], fine_steps['p'])
    p2_vals = np.arange(fine_ranges['p_k2_D1'][0], fine_ranges['p_k2_D1'][1], fine_steps['p'])
    z_vals = np.arange(fine_ranges['z'][0], fine_ranges['z'][1], fine_steps['z'])
    D_k2_vals = np.arange(fine_ranges['D_k2'][0], fine_ranges['D_k2'][1], fine_steps['D_k2'])
    f1_vals = np.arange(fine_ranges['f_k1_uav'][0], fine_ranges['f_k1_uav'][1], fine_steps['f'])
    f2_vals = np.arange(fine_ranges['f_k2_uav'][0], fine_ranges['f_k2_uav'][1], fine_steps['f'])

    for p_k1_D1_val in p_vals:
        for p_k2_D1_val in p2_vals:
            for z_val in z_vals:
                for D_k2_val in D_k2_vals:
                    for f_k1_uav_val in f1_vals:
                        for f_k2_uav_val in f2_vals:
                            obj_value = 2 * D_k1 + D_k2_val + L_k1 * C / f_k1_uav_val + L_k2 * C / f_k2_uav_val
                            if (D_k1 * p_k1_D1_val - E_max <= 0 and
                                    p_k1_D1_val - P_max <= 0 and
                                    p_k2_D1_val - P_max <= 0 and
                                    D_k1 + L_k1 * C / f_k1_uav_val - D_k1_max <= 0 and
                                    D_k1 + D_k2_val + L_k2 * C / f_k2_uav_val - D_k2_max <= 0 and
                                    f_k1_uav_val + f_k2_uav_val - f_uav_max <= 0 and
                                    kappa_uav * f_k1_uav_val ** 2 * C * L_k1 + kappa_uav * f_k2_uav_val ** 2 * C * L_k2 + (
                                            c1 * (uav_diff ** 3) / (tau ** 2)) + (
                                            c2 * (tau ** 2) / uav_diff) - E_uav_max <= 0 and
                                    L_k1 - D_k1 * B * np.log(g_k2 * p_k2_D1_val + g_k1 * p_k1_D1_val + sigma2) + D_k1 * B * np.log(g_k2 * p_k2_D1_val + sigma2) <= 0 and
                                    D_k1 * p_k2_D1_val + sigma2 / g_k2 * D_k2_val * np.exp(z_val) - sigma2 / g_k2 * D_k2_val - E_max <= 0 and
                                    z_val - np.log1p(g_k2 * P_max / sigma2) <= 0 and
                                    L_k2 - D_k1 * B * np.log(sigma2 + g_k2 * p_k2_D1_val) + D_k1 * B * np.log(sigma2) - B * D_k2_val * z_val <= 0
                            ):
                                if obj_value < optimal_value:
                                    optimal_value = obj_value
                                    optimal_p_k1_D1 = p_k1_D1_val
                                    optimal_p_k2_D1 = p_k2_D1_val
                                    optimal_z = z_val
                                    optimal_D_k2 = D_k2_val
                                    optimal_f_k1_uav = f_k1_uav_val
                                    optimal_f_k2_uav = f_k2_uav_val

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


# 对 user2_lager_gain_ES 做同样的最小修改（流程一致），下面只是直接给出改动后的函数体：
def user2_lager_gain_ES(L_k1, L_k2, tau, D_k1, g_k1, g_k2, uav_pos_pre, uav_pos, case_id):
    # 与 user1 函数中相同的转换与参数（为节省篇幅，保持实现与上面一致）
    uav_pos = np.array(uav_pos, dtype=float)
    uav_pos_pre = np.array(uav_pos_pre, dtype=float)

    B = 1e6
    sigma2 = 3.98e-21 * B
    P_max = 0.1995
    C = 800

    c1 = 0.00614
    c2 = 15.976
    kappa_uav = 1e-28
    f_uav_max = 3e9
    E_uav_max = 5

    E_max = 0.2
    D_k1_max = np.random.uniform(tau - 0.3, tau)
    D_k2_max = np.random.uniform(D_k1_max, tau)

    uav_diff = np.linalg.norm(uav_pos - uav_pos_pre)
    z_max = log1p(g_k2 * P_max / sigma2)

    MIN_VAL = 1e-20
    original_ranges = {
        'p_k1_D1': (MIN_VAL, P_max),
        'p_k2_D1': (MIN_VAL, P_max),
        'z': (MIN_VAL, z_max),
        'D_k2': (MIN_VAL, D_k2_max),
        'f_k1_uav': (MIN_VAL, f_uav_max),
        'f_k2_uav': (MIN_VAL, f_uav_max)
    }

    optimal_value = float('inf')
    optimal_p_k1_D1 = (original_ranges['p_k1_D1'][0] + original_ranges['p_k1_D1'][1]) / 2
    optimal_p_k2_D1 = (original_ranges['p_k2_D1'][0] + original_ranges['p_k2_D1'][1]) / 2
    optimal_z = (original_ranges['z'][0] + original_ranges['z'][1]) / 2
    optimal_D_k2 = (original_ranges['D_k2'][0] + original_ranges['D_k2'][1]) / 2
    optimal_f_k1_uav = (original_ranges['f_k1_uav'][0] + original_ranges['f_k1_uav'][1]) / 2
    optimal_f_k2_uav = (original_ranges['f_k2_uav'][0] + original_ranges['f_k2_uav'][1]) / 2

    # coarse steps same as paper
    coarse_steps = {'p': 0.01, 'D_k2': 0.02, 'z': 0.1, 'f': 0.05e9}

    # COARSE SEARCH
    print("====== Starting coarse search (user2) ======")
    p_vals = np.arange(original_ranges['p_k1_D1'][0], original_ranges['p_k1_D1'][1], coarse_steps['p'])
    p2_vals = np.arange(original_ranges['p_k2_D1'][0], original_ranges['p_k2_D1'][1], coarse_steps['p'])
    z_vals = np.arange(original_ranges['z'][0], original_ranges['z'][1], coarse_steps['z'])
    D_k2_vals = np.arange(original_ranges['D_k2'][0], original_ranges['D_k2'][1], coarse_steps['D_k2'])
    f1_vals = np.arange(original_ranges['f_k1_uav'][0], original_ranges['f_k1_uav'][1], coarse_steps['f'])
    f2_vals = np.arange(original_ranges['f_k2_uav'][0], original_ranges['f_k2_uav'][1], coarse_steps['f'])

    for p_k1_D1_val in p_vals:
        for p_k2_D1_val in p2_vals:
            for z_val in z_vals:
                for D_k2_val in D_k2_vals:
                    for f_k1_uav_val in f1_vals:
                        for f_k2_uav_val in f2_vals:
                            obj_value = 2 * D_k1 + D_k2_val + L_k1 * C / f_k1_uav_val + L_k2 * C / f_k2_uav_val
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
                                    D_k1 * p_k2_D1_val + sigma2 / g_k2 * D_k2_val * np.exp(z_val) - sigma2 / g_k2 * D_k2_val - E_max <= 0 and
                                    z_val - np.log1p(g_k2 * P_max / sigma2) <= 0 and
                                    L_k2 - D_k1 * B * np.log(
                                        g_k1 * p_k1_D1_val + g_k2 * p_k2_D1_val + sigma2) + D_k1 * B * np.log(
                                        g_k1 * p_k1_D1_val + sigma2) - B * D_k2_val * z_val <= 0
                            ):
                                if obj_value < optimal_value:
                                    optimal_value = obj_value
                                    optimal_p_k1_D1 = p_k1_D1_val
                                    optimal_p_k2_D1 = p_k2_D1_val
                                    optimal_z = z_val
                                    optimal_D_k2 = D_k2_val
                                    optimal_f_k1_uav = f_k1_uav_val
                                    optimal_f_k2_uav = f_k2_uav_val

    print(" Coarse-optimal found (user2):", optimal_value)

    # FINE SEARCH 准备（同上）
    fine_steps = {'p': coarse_steps['p'] / 10.0, 'D_k2': coarse_steps['D_k2'] / 10.0, 'z': coarse_steps['z'] / 10.0, 'f': coarse_steps['f'] / 10.0}
    def clamp(a, lo, hi): return max(lo, min(hi, a))

    fine_ranges = {
        'p_k1_D1': (clamp(optimal_p_k1_D1 - coarse_steps['p'], MIN_VAL, P_max),
                    clamp(optimal_p_k1_D1 + coarse_steps['p'], MIN_VAL, P_max)),
        'p_k2_D1': (clamp(optimal_p_k2_D1 - coarse_steps['p'], MIN_VAL, P_max),
                    clamp(optimal_p_k2_D1 + coarse_steps['p'], MIN_VAL, P_max)),
        'z': (clamp(optimal_z - coarse_steps['z'], MIN_VAL, z_max),
              clamp(optimal_z + coarse_steps['z'], MIN_VAL, z_max)),
        'D_k2': (clamp(optimal_D_k2 - coarse_steps['D_k2'], MIN_VAL, D_k2_max),
                 clamp(optimal_D_k2 + coarse_steps['D_k2'], MIN_VAL, D_k2_max)),
        'f_k1_uav': (clamp(optimal_f_k1_uav - coarse_steps['f'], MIN_VAL, f_uav_max),
                     clamp(optimal_f_k1_uav + coarse_steps['f'], MIN_VAL, f_uav_max)),
        'f_k2_uav': (clamp(optimal_f_k2_uav - coarse_steps['f'], MIN_VAL, f_uav_max),
                     clamp(optimal_f_k2_uav + coarse_steps['f'], MIN_VAL, f_uav_max))
    }

    print("====== Starting fine search (user2) ======")
    p_vals = np.arange(fine_ranges['p_k1_D1'][0], fine_ranges['p_k1_D1'][1], fine_steps['p'])
    p2_vals = np.arange(fine_ranges['p_k2_D1'][0], fine_ranges['p_k2_D1'][1], fine_steps['p'])
    z_vals = np.arange(fine_ranges['z'][0], fine_ranges['z'][1], fine_steps['z'])
    D_k2_vals = np.arange(fine_ranges['D_k2'][0], fine_ranges['D_k2'][1], fine_steps['D_k2'])
    f1_vals = np.arange(fine_ranges['f_k1_uav'][0], fine_ranges['f_k1_uav'][1], fine_steps['f'])
    f2_vals = np.arange(fine_ranges['f_k2_uav'][0], fine_ranges['f_k2_uav'][1], fine_steps['f'])

    for p_k1_D1_val in p_vals:
        for p_k2_D1_val in p2_vals:
            for z_val in z_vals:
                for D_k2_val in D_k2_vals:
                    for f_k1_uav_val in f1_vals:
                        for f_k2_uav_val in f2_vals:
                            obj_value = 2 * D_k1 + D_k2_val + L_k1 * C / f_k1_uav_val + L_k2 * C / f_k2_uav_val
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
                                    D_k1 * p_k2_D1_val + sigma2 / g_k2 * D_k2_val * np.exp(z_val) - sigma2 / g_k2 * D_k2_val - E_max <= 0 and
                                    z_val - np.log1p(g_k2 * P_max / sigma2) <= 0 and
                                    L_k2 - D_k1 * B * np.log(
                                        g_k1 * p_k1_D1_val + g_k2 * p_k2_D1_val + sigma2) + D_k1 * B * np.log(
                                        g_k1 * p_k1_D1_val + sigma2) - B * D_k2_val * z_val <= 0
                            ):
                                if obj_value < optimal_value:
                                    optimal_value = obj_value
                                    optimal_p_k1_D1 = p_k1_D1_val
                                    optimal_p_k2_D1 = p_k2_D1_val
                                    optimal_z = z_val
                                    optimal_D_k2 = D_k2_val
                                    optimal_f_k1_uav = f_k1_uav_val
                                    optimal_f_k2_uav = f_k2_uav_val

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
                    L_k1, L_k2, tau, D_k1, g_k1 * 1e15, g_k2 * 1e15, uav_pos_pre, uav_pos, case_id
                )
            else:
                optimal_value = user2_lager_gain_ES(
                    L_k1, L_k2, tau, D_k1, g_k1 * 1e15, g_k2 * 1e15, uav_pos_pre, uav_pos, case_id
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