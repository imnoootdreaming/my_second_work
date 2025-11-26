import time
import numpy as np
import pandas as pd
from equivalent_channel import generate_equivalent_channel
import sys
# 导入两种算法
from CCCP import user1_lager_gain_CCCP, user2_lager_gain_CCCP
from exhaustive_search import user1_lager_gain_ES, user2_lager_gain_ES


def compare_algorithms_per_type(num_cases_each=10):
    """分别生成 g1>g2 和 g2>g1 各 10 个案例，并保存结果"""
    np.random.seed(42)

    # 基础参数
    center = (0, 200)
    radius = 60
    uav_pos_pre = [0, 0, 50]
    ris_pos = [0, 0, 20]
    D_k1 = 0.2
    num_users = 2

    results_g1_greater = []
    results_g2_greater = []

    # 分别统计数量
    count_g1 = 0
    count_g2 = 0

    while count_g1 < num_cases_each and count_g2 < num_cases_each:
        try:
            tau = (0.65 - 0.5) * np.random.rand() + 0.5
            uav_pos = [0, 0 + 10 * tau, 50]
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
            g_k1 *= 1e15
            g_k2 *= 1e15

            # ===== g1 > g2 =====
            if g_k1 > g_k2 and count_g1 < num_cases_each:
                case_type = "g1>g2"
                start_cccp = time.time()
                optimal_cccp = user1_lager_gain_CCCP(L_k1, L_k2, tau, D_k1, g_k1, g_k2,
                                                     uav_pos_pre, uav_pos, count_g1 + 1, 0)
                if optimal_cccp == float("inf"):
                    continue
                time_cccp = time.time() - start_cccp

                start_es = time.time()
                optimal_es = user1_lager_gain_ES(L_k1, L_k2, tau, D_k1, g_k1, g_k2,
                                                 uav_pos_pre, uav_pos, count_g1 + 1)
                if optimal_es == float("inf"):
                    continue
                time_es = time.time() - start_es

                count_g1 += 1
                print(f"\n案例 {count_g1} ({case_type}) 结果：")
                print(f"  CCCP 最优时延: {optimal_cccp:.6f}, 运行时间: {time_cccp:.3f}s")
                print(f"  ES   最优时延: {optimal_es:.6f}, 运行时间: {time_es:.3f}s")

                results_g1_greater.append({
                    "Case": count_g1,
                    "Type": case_type,
                    "Delay_CCCP": optimal_cccp,
                    "Delay_ES": optimal_es,
                    "Time_CCCP(s)": time_cccp,
                    "Time_ES(s)": time_es
                })

            # ===== g2 > g1 =====
            elif g_k2 > g_k1 and count_g2 < num_cases_each:
                case_type = "g2>g1"
                start_cccp = time.time()
                optimal_cccp = user2_lager_gain_CCCP(L_k1, L_k2, tau, D_k1, g_k1, g_k2,
                                                     uav_pos_pre, uav_pos, count_g2 + 1, 0)
                if optimal_cccp == float("inf"):
                    continue
                time_cccp = time.time() - start_cccp

                start_es = time.time()
                optimal_es = user2_lager_gain_ES(L_k1, L_k2, tau, D_k1, g_k1, g_k2,
                                                 uav_pos_pre, uav_pos, count_g2 + 1)
                if optimal_es == float("inf"):
                    continue
                time_es = time.time() - start_es

                count_g2 += 1
                print(f"\n案例 {count_g2} ({case_type}) 结果：")
                print(f"  CCCP 最优时延: {optimal_cccp:.6f}, 运行时间: {time_cccp:.3f}s")
                print(f"  ES   最优时延: {optimal_es:.6f}, 运行时间: {time_es:.3f}s")

                results_g2_greater.append({
                    "Case": count_g2,
                    "Type": case_type,
                    "Delay_CCCP": optimal_cccp,
                    "Delay_ES": optimal_es,
                    "Time_CCCP(s)": time_cccp,
                    "Time_ES(s)": time_es
                })

        except Exception as e:
            print(f"⚠️ 出错：{e}")
            continue

    # ===== 保存结果 =====
    df_g1 = pd.DataFrame(results_g1_greater)
    df_g2 = pd.DataFrame(results_g2_greater)
    df_g1.to_csv("compare_CCCP_vs_ES_g1_greater.csv", index=False)
    df_g2.to_csv("compare_CCCP_vs_ES_g2_greater.csv", index=False)

    print("\n✅ 结果已保存：")
    print(" - compare_CCCP_vs_ES_g1_greater.csv")
    print(" - compare_CCCP_vs_ES_g2_greater.csv")

    return df_g1, df_g2


if __name__ == "__main__":
    compare_algorithms_per_type(num_cases_each=10)
