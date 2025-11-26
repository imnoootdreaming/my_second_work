import time
import numpy as np
import pandas as pd
from equivalent_channel import generate_equivalent_channel
import sys
# 导入两种算法
from CCCP import user1_lager_gain_CCCP, user2_lager_gain_CCCP
from exhaustive_search import user1_lager_gain_ES, user2_lager_gain_ES


def compare_algorithms(num_cases=10):
    np.random.seed(42)

    center = (0, 200)
    radius = 60
    uav_pos_pre = [0, 0, 50]
    ris_pos = [0, 0, 20]
    D_k1 = 0.2
    num_users = 2

    results = []
    count = 0

    while count < num_cases:
        try:
            tau = (0.65 - 0.5) * np.random.rand() + 0.5
            uav_pos = [0, 10 * tau, 50]
            L_k1 = (100 + 100 * np.random.rand()) * 1000
            L_k2 = (100 + 100 * np.random.rand()) * 1000

            # 用户位置
            angles = np.random.uniform(0, 2 * np.pi, num_users)
            r = radius * np.sqrt(np.random.uniform(0, 1, num_users))
            users_x = center[0] + r * np.cos(angles)
            users_y = center[1] + r * np.sin(angles)
            users_z = np.zeros(num_users)
            users_pos = np.stack([users_x, users_y, users_z], axis=1)

            # 等效信道
            g_k1, g_k2 = generate_equivalent_channel(uav_pos, ris_pos, users_pos)
            g_k1 *= 1e15
            g_k2 *= 1e15

            # 根据信道选择对应用户
            if g_k1 > g_k2:
                start_c = time.time()
                delay_cccp = user1_lager_gain_CCCP(L_k1, L_k2, tau, D_k1,
                                                   g_k1, g_k2, uav_pos_pre, uav_pos, count + 1, 0)
                t_cccp = time.time() - start_c
                print(f"CCCP delay:{delay_cccp} time:{t_cccp}")
                if delay_cccp == float("inf"):
                    continue
                start_e = time.time()
                delay_es = user1_lager_gain_ES(L_k1, L_k2, tau, D_k1,
                                               g_k1, g_k2, uav_pos_pre, uav_pos, count + 1)
                t_es = time.time() - start_e
                print(f"C2F delay:{delay_es} time:{t_es}")
                case_type = "g1>g2"

            else:
                start_c = time.time()
                delay_cccp = user2_lager_gain_CCCP(L_k1, L_k2, tau, D_k1,
                                                   g_k1, g_k2, uav_pos_pre, uav_pos, count + 1, 0)
                t_cccp = time.time() - start_c
                print(f"CCCP delay:{delay_cccp} time:{t_cccp}")
                if delay_cccp == float("inf"):
                    continue
                start_e = time.time()
                delay_es = user2_lager_gain_ES(L_k1, L_k2, tau, D_k1,
                                               g_k1, g_k2, uav_pos_pre, uav_pos, count + 1)
                t_es = time.time() - start_e
                print(f"C2F delay:{delay_es} time:{t_es}")
                case_type = "g2>g1"

            if delay_es == float("inf"):
                continue

            count += 1
            print(f"\n案例 {count} ({case_type})")
            print(f"  CCCP : {delay_cccp:.6f}, time={t_cccp:.3f}s")
            print(f"  ES   : {delay_es:.6f}, time={t_es:.3f}s")

            results.append({
                "Case": count,
                "Type": case_type,
                "Delay_CCCP": delay_cccp,
                "Delay_ES": delay_es,
                "Time_CCCP(s)": t_cccp,
                "Time_ES(s)": t_es,
            })

        except Exception as e:
            print("出错：", e)
            continue

    df = pd.DataFrame(results)
    df.to_csv("compare_CCCP_vs_C2F_total10.csv", index=False)
    return df


if __name__ == "__main__":
    compare_algorithms(num_cases=10)
