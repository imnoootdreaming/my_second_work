import os
import warnings
import numpy as np
import cvxpy as cp
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings('error', category=RuntimeWarning)  # 把 RuntimeWarning 当作异常处理


def generate_fixed_nlos(num_user, K, num_inner_iter, seed):
    """
    生成固定 NLoS 分量，类属性共享
    """
    rng = np.random.default_rng(seed)
    nlos_real = rng.normal(0, 1, (num_inner_iter, num_user, K))
    nlos_imag = rng.normal(0, 1, (num_inner_iter, num_user, K))
    return nlos_real, nlos_imag


class RIS:
    # 类属性，用于所有实例共享 NLoS 分量
    nlos_real = None
    nlos_imag = None

    def __init__(self, K, coordinate_users, coordinate_ris, coordinate_uav):
        """
        :param K: RIS 元件数
        :param coordinate_users: 用户坐标 (num_user, 3)
        :param coordinate_ris: RIS 坐标 (3,)
        :param coordinate_uav: UAV 坐标 (num_uav, 3)
        """
        self.K = K
        self.coordinate_users = np.array(coordinate_users)
        self.coordinate_ris = np.array(coordinate_ris)
        self.coordinate_uav = np.array(coordinate_uav)
        self.num_user = self.coordinate_users.shape[0]
        self.num_uav = self.coordinate_uav.shape[0]

        # 大尺度衰落
        self.loss_large_user_2_ris = np.zeros(self.coordinate_users.shape[0])
        self.loss_large_ris_2_uav = np.zeros(self.coordinate_uav.shape[0])

        # RIS 参数矩阵
        self.theta = np.eye(K, dtype=np.complex128)

        # 用户->RIS 和 RIS->UAV 相位响应矩阵
        self.user_2_ris_NLoS = np.zeros((self.coordinate_users.shape[0], K), dtype=np.complex128)
        self.ris_2_uav_NLoS = np.zeros((self.coordinate_uav.shape[0], K), dtype=np.complex128)

    def loss_large(self, alpha_ris_2_uav=2.3, alpha_user_2_ris=3, g0_dB=-30):
        """
        大尺度衰落模型
            用户 -> RIS
            RIS -> UAV
        """
        g0 = 10 ** (g0_dB / 10)  # 线性值 1e-3

        # 大尺度衰落模型 lambda 表达式
        g = lambda d, alpha: g0 * d ** (-alpha)

        # 用户 -> UAV (直射链路不存在)
        loss_large_d = np.zeros(self.num_user)

        # 算出所有 User 到 RIS 的信道增益
        for n in range(self.num_user):
            d_RU = np.linalg.norm(self.coordinate_users[n] - self.coordinate_ris)
            self.loss_large_user_2_ris[n] = g(d_RU, alpha_user_2_ris)

        # 算出所有 RIS 到 UAV 的信道增益
        for n in range(self.num_uav):
            d_RB = np.linalg.norm(self.coordinate_ris - self.coordinate_uav[n])
            self.loss_large_ris_2_uav[n] = g(d_RB, alpha_ris_2_uav)

    def set_user_ris_phase_response(self, d_over_lambda=0.5):
        """
        设置 User -> RIS LoS 相位响应
        """
        for n in range(self.num_user):
            cos_phi = (self.coordinate_users[n, 0] - self.coordinate_ris[0]) / \
                      np.linalg.norm(self.coordinate_users[n] - self.coordinate_ris)
            phase_vector = np.exp(1j * -2 * np.pi * d_over_lambda * np.arange(self.K) * cos_phi)
            self.user_2_ris_NLoS[n, :] = phase_vector

    def set_ris_uav_phase_response(self, next_uav_pos, d_over_lambda=0.5):
        """
        设置 RIS -> UAV LoS 相位响应
        :param next_uav_pos: UAV 坐标 (num_uav, 3)
        """
        self.coordinate_uav = np.array(next_uav_pos)
        for u in range(self.num_uav):
            cos_phi = (self.coordinate_uav[u, 0] - self.coordinate_ris[0]) / \
                      np.linalg.norm(self.coordinate_uav[u] - self.coordinate_ris)
            phase_vector = np.exp(1j * 2 * np.pi * d_over_lambda * np.arange(self.K) * cos_phi)
            self.ris_2_uav_NLoS[u, :] = phase_vector

    def set_ris_parameters(self, phase):
        """
        设置 RIS 的反射面参数
        :param phase: K维相位向量 (单位: 弧度)，长度 = self.K
        """
        self.theta = np.diag(np.exp(1j * phase))

    def set_random_ris_parameters(self):
        random_phase = np.random.uniform(0, 2 * np.pi, self.K)  # 随机相位
        self.theta = np.diag(np.exp(1j * random_phase))  # 随机反射对角矩阵

    def channel_compute(self, num_user, num_uav, num_noma_group,
                        inner_iter, total_time_slot, seed, beta_dB=3):
        """
        计算 User -> RIS -> UAV 信道增益
        :return: h_user_2_uav (num_uav, num_noma_group, 2),
                 h_user_2_ris_all (num_user, K),
                 h_ris_2_uav (num_uav, K)
        """
        beta = 10 ** (beta_dB / 10)
        h_user_2_uav = np.zeros((num_uav, num_user), dtype=np.float64)
        h_ris_2_uav = np.zeros((num_uav, self.K), dtype=np.complex128)
        h_user_2_ris_all = np.zeros((num_user, self.K), dtype=np.complex128)

        # 延迟生成 NLoS 分量，只生成一次，类属性共享
        if RIS.nlos_real is None or RIS.nlos_imag is None:
            RIS.nlos_real, RIS.nlos_imag = generate_fixed_nlos(self.num_user, self.K, total_time_slot, seed)

        # 当前 inner_iter 时隙对应的 NLoS 分量
        nlos_real_slice = RIS.nlos_real[inner_iter, :, :]  # shape: (num_user, K)
        nlos_imag_slice = RIS.nlos_imag[inner_iter, :, :]
        # print(f"========== NLoS 实部分量: {nlos_real_slice} ==========")
        # print(f"========== NLoS 虚部分量: {nlos_imag_slice} ==========")
        for u in range(num_uav):
            # UAV LoS 信道 (K,)
            h_ris_2_uav[u, :] = self.loss_large_ris_2_uav[u] * self.ris_2_uav_NLoS[u, :]

            for n in range(num_user):
                # User -> RIS
                # LoS + NLoS 信道
                h_user_2_ris = self.loss_large_user_2_ris[n] * (
                        np.sqrt(beta / (1 + beta)) * self.user_2_ris_NLoS[n, :].reshape(1, -1) +
                        np.sqrt(1 / (1 + beta)) *
                        (nlos_real_slice[n, :].reshape(1, -1) + 1j * nlos_imag_slice[n, :].reshape(1, -1))
                )
                h_user_2_ris_all[n, :] = h_user_2_ris.flatten()

                composite_channel = h_user_2_ris @ self.theta @ h_ris_2_uav[u, :].reshape(-1, 1)
                # User -> RIS -> UAV 信道增益取模后平方
                h_user_2_uav[u, n] = np.abs(composite_channel[0, 0]) ** 2

        # 把复数信道转为功率增益
        h_user_2_ris_all = np.abs(h_user_2_ris_all) ** 2
        h_ris_2_uav = np.abs(h_ris_2_uav) ** 2

        return h_user_2_uav, h_user_2_ris_all, h_ris_2_uav


def compute_one_group(uav_idx, g1, g2,
                      L_k1, L_k2, tau, D_k1,
                      user1_lager_gain_CCCP, user2_lager_gain_CCCP,
                      uav_pos_pre, uav_pos):
    """计算单个 UAV-NOMA组的 delay"""
    vio_no_feasible_solution = 0
    try:
        if g1 >= g2:
            delay = user1_lager_gain_CCCP(
                L_k1, L_k2, tau, D_k1,
                g1, g2, uav_pos_pre,
                uav_pos, uav_idx
            )
        else:
            delay = user2_lager_gain_CCCP(
                L_k1, L_k2, tau, D_k1,
                g1, g2, uav_pos_pre,
                uav_pos, uav_idx
            )
    except Exception as e:
        print(f"[Warning] CCCP computation failed for UAV {uav_idx + 1}: {e}")
        vio_no_feasible_solution = 1
        delay = float("inf")

    if delay == float("inf"):
        vio_no_feasible_solution = 1  # 如果 CCCP 求解没有求解出来值 同样是无解情况

    return delay, vio_no_feasible_solution


def compute_one_group_DRL(uav_idx, g1, g2,
                          L_k1, L_k2, tau, D_k1, D_k2,
                          p_k1_D1, p_k2_D1, p_k2_D2,
                          f_uav_k1, f_uav_k2,
                          uav_pos_pre, uav_pos):
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
    # NOTE - 这里 D_k1_max 肯定是要小于 D_k2_max
    D_k1_max = tau - 0.1
    D_k2_max = tau  # 用户 2 的最大可容忍时延大于用户 1
    # UAV前后迭代位置差
    uav_diff = np.linalg.norm(uav_pos - uav_pos_pre)
    uav_diff = max(uav_diff, 1e-8)  # 防止除零
    """计算单个 UAV-NOMA组的 delay"""
    vio_constraint = 0
    vio_uav_constraint = 0  # UAV 动作导致的
    delay = float("inf")
    z = np.log1p(g2 * p_k2_D2 / sigma2)  # 替换成 z
    if g1 >= g2:
        # (27d)
        if D_k1 * p_k1_D1 - E_max > 0:
            vio_constraint += 1
        # (27e)
        if p_k1_D1 - P_max > 0:
            vio_constraint += 1
        # (27f)
        if p_k2_D1 - P_max > 0:
            vio_constraint += 1
        # (27h)
        if D_k1 + L_k1 * C / f_uav_k1 - D_k1_max > 0:
            vio_constraint += 1
            vio_uav_constraint += 1
        # (27i)
        if D_k1 + D_k2 + L_k2 * C / f_uav_k2 - D_k2_max > 0:
            vio_constraint += 1
            vio_uav_constraint += 1
        # (27j)
        if f_uav_k1 + f_uav_k2 - f_uav_max > 0:
            vio_uav_constraint += (f_uav_k1 + f_uav_k2 - f_uav_max) / f_uav_max
        # (27k)
        if kappa_uav * f_uav_k1 ** 2 * C * L_k1 + kappa_uav * f_uav_k2 ** 2 * C * L_k2 + (c1 * (uav_diff ** 3) / (tau ** 2)) + (c2 * (tau ** 2) / uav_diff) - E_uav_max > 0:
            vio_uav_constraint += 1
        # (28)
        if L_k1 - D_k1 * B * np.log(g2 * p_k2_D1 + g1 * p_k1_D1 + sigma2) + D_k1 * B * np.log(g2 * p_k2_D1 + sigma2) > 0:
            vio_constraint += 1
            print("约束28")
        # (30)
        if D_k1 * p_k2_D1 + (sigma2 / g2) * D_k2 * np.exp(z) - (sigma2 / g2) * D_k2 - E_max > 0:
            vio_constraint += 1
        # (31)
        if z - np.log1p(g2 * P_max / sigma2) > 0:
            vio_constraint += 1
        # (32)
        if L_k2 - D_k1 * B * np.log(sigma2 + g2 * p_k2_D1) + D_k1 * B * np.log(sigma2) - B * D_k2 * z > 0:
            vio_constraint += 1
            print("约束32")
        if vio_constraint == 0:
            delay = 2 * D_k1 + D_k2 + L_k1 * C / f_uav_k1 + L_k2 * C / f_uav_k2
    else:
        if D_k1 * p_k1_D1 - E_max > 0:
            vio_constraint += 1
        if p_k1_D1 - P_max > 0:
            vio_constraint += 1
        if p_k2_D1 - P_max > 0:
            vio_constraint += 1
        if D_k1 + L_k1 * C / f_uav_k1 - D_k1_max > 0:
            vio_constraint += 1
            vio_uav_constraint += 1
        if D_k1 + D_k2 + L_k2 * C / f_uav_k2 - D_k2_max > 0:
            vio_constraint += 1
            vio_uav_constraint += 1
        if f_uav_k1 + f_uav_k2 - f_uav_max > 0:
            vio_uav_constraint += (f_uav_k1 + f_uav_k2 - f_uav_max) / f_uav_max
        if (kappa_uav * f_uav_k1 ** 2 * C * L_k1 +
            kappa_uav * f_uav_k2 ** 2 * C * L_k2 +
            (c1 * (uav_diff ** 3) / (tau ** 2)) +
            (c2 * (tau ** 2) / uav_diff) - E_uav_max) > 0:
            vio_uav_constraint += 1
        if L_k1 - D_k1 * B * np.log(sigma2 + g1 * p_k1_D1) + D_k1 * B * np.log(sigma2) > 0:
            vio_constraint += 1
            print("约束28")
        if (D_k1 * p_k2_D1 + (sigma2 / g2) * D_k2 * np.exp(z) - (sigma2 / g2) * D_k2 - E_max) > 0:
            vio_constraint += 1
        if z - np.log1p(g2 * P_max / sigma2) > 0:
            vio_constraint += 1
        if (L_k2 - D_k1 * B * np.log(g1 * p_k1_D1 + g2 * p_k2_D1 + sigma2)
            + D_k1 * B * np.log(g1 * p_k1_D1 + sigma2)
            - B * D_k2 * z) > 0:
            vio_constraint += 1
            print("约束32")
        if vio_constraint == 0:
            delay = 2 * D_k1 + D_k2 + L_k1 * C / f_uav_k1 + L_k2 * C / f_uav_k2
    return delay, vio_constraint / 9, vio_uav_constraint / 4  # 总共11个约束


def wrapper(args):
    return compute_one_group(*args)


def wrapper_DRL(args):
    return compute_one_group_DRL(*args)


class Reward:
    def __init__(self, K, ris_pos, uav_num, x_max, y_max, safe_distance, max_workers=None):
        self.K = K
        self.ris_pos = ris_pos
        self.ris_num = 1
        self.uav_num = uav_num
        self.x_max = x_max
        self.y_max = y_max
        self.safe_distance = safe_distance
        # 创建并复用进程池（在整个 Reward 生命周期中）
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)
        self._executor = ProcessPoolExecutor(max_workers=max_workers)

    def reward_compute(self, L_k1, L_k2, tau, D_k1, channel_gain,
                       uav_pos_pre, uav_pos, choose, noma_group_num,
                       uav_fixed, access_approache='NCO-NOMA'):
        uav_collision_penalty = np.zeros(self.uav_num)  # 每个 UAV 的碰撞惩罚
        uav_exceed_boundary_penalty = np.zeros(self.uav_num)  # 每个 UAV 的越界惩罚
        reward = {"uav": [[] for _ in range(self.uav_num)], "ris": []}

        # ===== ✅ 边界检查 + 碰撞检测 =====
        for i in range(self.uav_num):
            x, y, z = uav_pos[i]
            if x < -self.x_max or x > self.x_max or y < -self.y_max or y > self.y_max:
                uav_exceed_boundary_penalty[i] = 1
                # 拉回边界
                uav_pos[i][0] = np.clip(x, 0, self.x_max)
                uav_pos[i][1] = np.clip(y, 0, self.y_max)

            for j in range(i + 1, self.uav_num):
                dist = np.linalg.norm(uav_pos[i] - uav_pos[j])
                if dist < self.safe_distance:  # 小于5米判定为碰撞或过近
                    uav_collision_penalty[i] = 1
                    uav_collision_penalty[j] = 1

        args_list = []  # 存储参数
        choose = np.array(choose)  # 确保是 numpy array
        used_uavs = np.unique(choose)  # 只考虑 choose 中实际存在的 UAV
        total_vio_choose_single_uav = 0
        for uav_idx in used_uavs:
            # 找出选择该 UAV 的用户
            selected_users = np.where(choose == uav_idx)[0]
            # 违反约束(2) -> 每个用户只能选择一个UAV UAV必须处理两个用户
            penalty = abs(len(selected_users) - 2)
            total_vio_choose_single_uav += penalty
            # 不满足分组条件，不计算该 UAV 的基础 reward
            if penalty > 0:
                continue

            user1, user2 = selected_users[:2]

            # UAV 的索引就代表了第几个 NOMA 组 取对应 NOMA 组分配的 D_k1
            if access_approache == 'NCO-NOMA' and not uav_fixed:
                args_list.append((
                    uav_idx, channel_gain[uav_idx, user1], channel_gain[uav_idx, user2],
                    L_k1, L_k2, tau, D_k1[uav_idx],
                    self.user1_lager_gain_CCCP, self.user2_lager_gain_CCCP,
                    uav_pos_pre[uav_idx], uav_pos[uav_idx]
                ))
            elif access_approache == 'NCO-NOMA' and uav_fixed:
                args_list.append((
                    uav_idx, channel_gain[uav_idx, user1], channel_gain[uav_idx, user2],
                    L_k1, L_k2, tau, D_k1[uav_idx],
                    self.user1_lager_gain_fixedUAV_CCCP, self.user2_lager_gain_fixedUAV_CCCP,
                    uav_pos_pre[uav_idx], uav_pos[uav_idx]
                ))
            elif access_approache == 'NOMA':
                args_list.append((
                    uav_idx, channel_gain[uav_idx, user1], channel_gain[uav_idx, user2],
                    L_k1, L_k2, tau, D_k1[uav_idx],
                    self.user1_lager_gain_NOMA_CCCP, self.user2_lager_gain_NOMA_CCCP,
                    uav_pos_pre[uav_idx], uav_pos[uav_idx]
                ))
            else:
                args_list.append((
                    uav_idx, channel_gain[uav_idx, user1], channel_gain[uav_idx, user2],
                    L_k1, L_k2, tau, D_k1[uav_idx],
                    self.OMA_CCCP, self.OMA_CCCP,
                    uav_pos_pre[uav_idx], uav_pos[uav_idx]
                ))

        # 如果所有选择系数都无效 List 中将不会存储任何数据 :
        # 因此 reward 组成为：
        # 1. 选择 UAV 错误
        # 2. UAV 距离过近
        if len(args_list) == 0:
            ris_reward = - total_vio_choose_single_uav
            reward["ris"].append(ris_reward)

            for i in range(self.uav_num):
                r = - uav_collision_penalty[i] - uav_exceed_boundary_penalty[i]
                reward["uav"][i].append(r)

            total_reward = np.sum(reward["uav"]) - total_vio_choose_single_uav

            return total_reward, reward, np.inf

        # 并行执行满足约束(2)下 UAV 对应的NOMA组
        results = list(self._executor.map(wrapper, args_list))
        delays, vio_no_feasible_solution = zip(*results)  # 拆开结果

        # 分组计算奖励
        group_rewards = np.zeros(self.uav_num)
        valid_delays = []
        for idx, (args, d) in enumerate(zip(args_list, delays)):
            uav_idx = args[0]
            group_rewards[uav_idx] = np.exp(-d)  # 基础 reward
            valid_delays.append(d)

        total_delay = np.sum(valid_delays) if len(valid_delays) > 0 else np.inf

        # ===== ✅ 组合 UAV 和 RIS 的奖励 =====
        base_uav_reward = 0
        for i in range(self.uav_num):
            base_reward = group_rewards[i]
            r_uav = base_reward - uav_collision_penalty[i] - uav_exceed_boundary_penalty[i]
            reward["uav"][i].append(r_uav)
            base_uav_reward += base_reward

        ris_reward = (
                base_uav_reward
                - np.sum(vio_no_feasible_solution)
                - total_vio_choose_single_uav
        )
        reward["ris"].append(ris_reward)

        total_reward = (
                base_uav_reward
                - np.sum(uav_collision_penalty)
                - np.sum(uav_exceed_boundary_penalty)
                - np.sum(vio_no_feasible_solution)
                - total_vio_choose_single_uav
        )

        return total_reward, reward, total_delay

    def DRL_reward(self, L_k1, L_k2, tau, D_k1, D_k2, p_k1_D1, p_k2_D1, p_k2_D2, f_uav,
                   channel_gain, uav_pos_pre, uav_pos, choose, noma_group_num):
        uav_collision_penalty = np.zeros(self.uav_num)  # 每个 UAV 的碰撞惩罚
        uav_exceed_boundary_penalty = np.zeros(self.uav_num)  # 每个 UAV 的越界惩罚
        reward = {"uav": [[] for _ in range(self.uav_num)], "ris": []}

        # ===== ✅ 边界检查 + 碰撞检测=====
        for i in range(self.uav_num):
            x, y, z = uav_pos[i]
            if x < -self.x_max or x > self.x_max or y < -self.y_max or y > self.y_max:
                uav_exceed_boundary_penalty[i] = 1
                # 拉回边界
                uav_pos[i][0] = np.clip(x, 0, self.x_max)
                uav_pos[i][1] = np.clip(y, 0, self.y_max)

            for j in range(i + 1, self.uav_num):
                dist = np.linalg.norm(uav_pos[i] - uav_pos[j])
                if dist < self.safe_distance:  # 小于5米判定为碰撞或过近
                    uav_collision_penalty[i] = 1
                    uav_collision_penalty[j] = 1

        args_list = []  # 存储参数
        choose = np.array(choose)  # 确保是 numpy array
        used_uavs = np.unique(choose)  # 只考虑 choose 中实际存在的 UAV
        total_vio_choose_single_uav = 0
        for uav_idx in used_uavs:
            # 找出选择该 UAV 的用户
            selected_users = np.where(choose == uav_idx)[0]
            # 违反约束(2) -> 每个用户只能选择一个UAV UAV必须处理两个用户
            penalty = abs(len(selected_users) - 2)
            total_vio_choose_single_uav += penalty
            # 不满足分组条件，不计算该 UAV 的基础 reward
            if penalty > 0:
                continue

            user1, user2 = selected_users[:2]
            # UAV 的索引就代表了第几个 NOMA 组 取对应 NOMA 组分配的 D_k1
            args_list.append((
                uav_idx, channel_gain[uav_idx, user1] * 1e28, channel_gain[uav_idx, user2] * 1e28,
                L_k1, L_k2, tau, D_k1[uav_idx], D_k2[uav_idx],
                p_k1_D1[uav_idx], p_k2_D1[uav_idx], p_k2_D2[uav_idx],
                f_uav[uav_idx * 2], f_uav[uav_idx * 2 + 1], uav_pos_pre[uav_idx], uav_pos[uav_idx]
            ))

        # 如果所有选择系数都无效 List 中将不会存储任何数据 :
        # 因此 reward 组成为：
        # 1. 选择 UAV 错误
        # 2. UAV 距离过近
        if len(args_list) == 0:
            ris_reward = - total_vio_choose_single_uav
            reward["ris"].append(ris_reward)

            for i in range(self.uav_num):
                r = - uav_collision_penalty[i] - uav_exceed_boundary_penalty[i]
                reward["uav"][i].append(r)

            total_reward = np.sum(reward["uav"]) - total_vio_choose_single_uav

            return total_reward, reward, np.inf

        # 并行执行满足约束(2)下 UAV 对应的NOMA组
        results = list(self._executor.map(wrapper_DRL, args_list))
        delays, vio_constraint, vio_uav_constraint = zip(*results)  # 拆开结果

        # 分组计算奖励
        group_rewards = np.zeros(self.uav_num)
        valid_delays = []
        uav_vio_constraint = np.zeros(self.uav_num)  # 用来按 UAV 存 vio_uav_constraint
        for idx, (args, d, vio_uav) in enumerate(zip(args_list, delays, vio_uav_constraint)):
            uav_idx = args[0]
            group_rewards[uav_idx] = np.exp(-d)  # 基础 reward
            uav_vio_constraint[uav_idx] = vio_uav  # 把对应的 vio 存到该 UAV 的位置 其余没存储的位置都是0
            valid_delays.append(d)

        total_delay = np.sum(valid_delays) if len(valid_delays) > 0 else np.inf

        # ===== ✅ 组合 UAV 和 RIS 的奖励 =====
        base_uav_reward = 0
        for i in range(self.uav_num):
            base_reward = group_rewards[i]
            r_uav = base_reward - uav_collision_penalty[i] - uav_exceed_boundary_penalty[i] - uav_vio_constraint[i]
            reward["uav"][i].append(r_uav)
            base_uav_reward += base_reward

        ris_reward = (
                base_uav_reward
                - np.sum(vio_constraint)
                - total_vio_choose_single_uav
        )
        reward["ris"].append(ris_reward)

        total_reward = (
                base_uav_reward
                - np.sum(uav_collision_penalty)
                - np.sum(uav_exceed_boundary_penalty)
                - np.sum(vio_constraint)
                - np.sum(vio_uav_constraint)
                - total_vio_choose_single_uav
        )

        return total_reward, reward, total_delay

    @staticmethod
    def user1_lager_gain_CCCP(L_k1, L_k2, tau, D_k1,
                              g_k1, g_k2, uav_pos_pre,
                              uav_pos, id_noma_group):
        # print("-------------------------------------------------------- ")
        # print(f"--------- 第{id_noma_group + 1}个NOMA组 -> case1 -> CCCP算法开始迭代 --------- ")
        # print("-------------------------------------------------------- ")
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
        D_k1_max = tau - 0.1
        D_k2_max = tau  # 用户 2 的最大可容忍时延大于用户 1

        max_iterations = 8
        tolerance = 1e-2  # 收敛阈值

        # UAV前后迭代位置差
        uav_diff = np.linalg.norm(uav_pos - uav_pos_pre)
        uav_diff = max(uav_diff, 1e-8)  # 防止除零

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

        epsilon = 1e-8

        # 目标函数
        objective = cp.Minimize(
            2 * D_k1 + D_k2_var
            + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon)
            + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon)
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
            D_k1 + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon) - D_k1_max <= 0
        )

        # 27i
        constraints.append(
            D_k1 + D_k2_var + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon) - D_k2_max <= 0
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
                # print(f"CCCP算法迭代过程收敛在第 {iteration + 1} 轮")
                iteration_nums = iteration + 1
                optimal_value = current_objective_value
                # 输出当前优化结果和目标函数值
                # print(f"p_k1_D1 = {p_k1_D1_opt}")
                # print(f"p_k2_D1 = {p_k2_D1_opt}")
                # print(f"z_var = {z_var_opt}")
                # print(f"D_k2_var = {D_k2_var_opt}")
                # print(f"f_k1_uav_var = {f_k1_uav_var_opt * f_norm}")
                # print(f"f_k2_uav_var = {f_k2_uav_var_opt * f_norm}")
                # print(f"第{id_noma_group + 1}个NOMA组 -> Case1 -> 最小延迟为: : {optimal_value}", flush=True)
                break

            # 更新为当前CCCP求解出来的最优解
            if p_k2_D1_var.value is not None:
                hat_p_k2_D1_param.value = p_k2_D1_var.value
            if z_var.value is not None:
                hat_z_param.value = z_var.value
            if D_k2_var.value is not None:
                hat_D_k2_param.value = D_k2_var.value
            if f_k1_uav_var.value is not None:
                hat_f_k1_uav_param.value = f_k1_uav_var.value
            if f_k2_uav_var.value is not None:
                hat_f_k2_uav_param.value = f_k2_uav_var.value

            # 更新最优目标函数值
            pre_objective_value = current_objective_value

        return optimal_value

    @staticmethod
    def user2_lager_gain_CCCP(L_k1, L_k2, tau, D_k1,
                              g_k1, g_k2, uav_pos_pre,
                              uav_pos, id_noma_group):
        # print("-------------------------------------------------------- ")
        # print(f"--------- 第{id_noma_group + 1}个NOMA组 -> case2 -> CCCP算法开始迭代 --------- ")
        # print("-------------------------------------------------------- ")
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
        D_k1_max = tau - 0.1
        D_k2_max = tau  # 用户 2 的最大可容忍时延大于用户 1

        # UAV前后迭代位置差
        uav_diff = np.linalg.norm(uav_pos - uav_pos_pre)
        uav_diff = max(uav_diff, 1e-8)  # 防止除零

        # CCCP算法的最大迭代次数
        max_iterations = 8
        tolerance = 1e-2  # 收敛阈值

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

        epsilon = 1e-8

        # 目标函数
        objective = cp.Minimize(
            2 * D_k1 + D_k2_var
            + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon)
            + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon)
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
            D_k1 + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon) - D_k1_max <= 0
        )

        # 27i
        constraints.append(
            D_k1 + D_k2_var + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon) - D_k2_max <= 0
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
                # print(f"CCCP算法迭代过程收敛在第 {iteration + 1} 轮")
                iteration_nums = iteration + 1
                optimal_value = current_objective_value
                # 输出当前优化结果和目标函数值
                # print(f"p_k1_D1 = {p_k1_D1_var_opt}")
                # print(f"p_k2_D1 = {p_k2_D1_var_opt}")
                # print(f"z_var = {z_var_opt}")
                # print(f"D_k2_var = {D_k2_var_opt}")
                # print(f"f_k1_uav_var = {f_k1_uav_var_opt * f_norm}")
                # print(f"f_k2_uav_var = {f_k2_uav_var_opt * f_norm}")
                # print(f"第{id_noma_group + 1}个NOMA组 -> Case2 -> 最小延迟为: : {optimal_value}", flush=True)
                break

            # # 更新为当前CCCP求解出来的最优解
            # hat_p_k1_D1_param.value = p_k1_D1_var.value
            # hat_z_param.value = z_var.value
            # hat_D_k2_param.value = D_k2_var.value
            # hat_f_k1_uav_param.value = f_k1_uav_var.value
            # hat_f_k2_uav_param.value = f_k2_uav_var.value
            # 更新为当前CCCP求解出来的最优解
            if p_k1_D1_var.value is not None:
                hat_p_k1_D1_param.value = p_k1_D1_var.value
            if z_var.value is not None:
                hat_z_param.value = z_var.value
            if D_k2_var.value is not None:
                hat_D_k2_param.value = D_k2_var.value
            if f_k1_uav_var.value is not None:
                hat_f_k1_uav_param.value = f_k1_uav_var.value
            if f_k2_uav_var.value is not None:
                hat_f_k2_uav_param.value = f_k2_uav_var.value

            # 更新最优目标函数值
            pre_objective_value = current_objective_value

        return optimal_value

    @staticmethod
    def user1_lager_gain_fixedUAV_CCCP(L_k1, L_k2, tau, D_k1,
                                       g_k1, g_k2, uav_pos_pre,
                                       uav_pos, id_noma_group):
        # 类型转换
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
        D_k1_max = tau - 0.1
        D_k2_max = tau  # 用户 2 的最大可容忍时延大于用户 1

        max_iterations = 8
        tolerance = 1e-2  # 收敛阈值

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

        epsilon = 1e-8

        # 目标函数
        objective = cp.Minimize(
            2 * D_k1 + D_k2_var
            + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon)
            + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon)
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
            D_k1 + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon) - D_k1_max <= 0
        )

        # 27i
        constraints.append(
            D_k1 + D_k2_var + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon) - D_k2_max <= 0
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
                    f_k2_uav_var * f_norm - hat_f_k2_uav_param * f_norm)) * C * L_k2 - E_uav_max <= 0
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
                # print(f"CCCP算法迭代过程收敛在第 {iteration + 1} 轮")
                iteration_nums = iteration + 1
                optimal_value = current_objective_value
                # 输出当前优化结果和目标函数值
                # print(f"p_k1_D1 = {p_k1_D1_opt}")
                # print(f"p_k2_D1 = {p_k2_D1_opt}")
                # print(f"z_var = {z_var_opt}")
                # print(f"D_k2_var = {D_k2_var_opt}")
                # print(f"f_k1_uav_var = {f_k1_uav_var_opt * f_norm}")
                # print(f"f_k2_uav_var = {f_k2_uav_var_opt * f_norm}")
                # print(f"第{id_noma_group + 1}个NOMA组 -> Case1 -> 最小延迟为: : {optimal_value}", flush=True)
                break

            # 更新为当前CCCP求解出来的最优解
            if p_k2_D1_var.value is not None:
                hat_p_k2_D1_param.value = p_k2_D1_var.value
            if z_var.value is not None:
                hat_z_param.value = z_var.value
            if D_k2_var.value is not None:
                hat_D_k2_param.value = D_k2_var.value
            if f_k1_uav_var.value is not None:
                hat_f_k1_uav_param.value = f_k1_uav_var.value
            if f_k2_uav_var.value is not None:
                hat_f_k2_uav_param.value = f_k2_uav_var.value

            # 更新最优目标函数值
            pre_objective_value = current_objective_value

        return optimal_value

    @staticmethod
    def user2_lager_gain_fixedUAV_CCCP(L_k1, L_k2, tau, D_k1,
                                       g_k1, g_k2, uav_pos_pre,
                                       uav_pos, id_noma_group):
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
        D_k1_max = tau - 0.1
        D_k2_max = tau  # 用户 2 的最大可容忍时延大于用户 1

        # CCCP算法的最大迭代次数
        max_iterations = 8
        tolerance = 1e-2  # 收敛阈值

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

        epsilon = 1e-8

        # 目标函数
        objective = cp.Minimize(
            2 * D_k1 + D_k2_var
            + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon)
            + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon)
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
            D_k1 + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon) - D_k1_max <= 0
        )

        # 27i
        constraints.append(
            D_k1 + D_k2_var + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon) - D_k2_max <= 0
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
                    f_k2_uav_var * f_norm - hat_f_k2_uav_param * f_norm)) * C * L_k2 - E_uav_max <= 0
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
                # print(f"CCCP算法迭代过程收敛在第 {iteration + 1} 轮")
                iteration_nums = iteration + 1
                optimal_value = current_objective_value
                # 输出当前优化结果和目标函数值
                # print(f"p_k1_D1 = {p_k1_D1_var_opt}")
                # print(f"p_k2_D1 = {p_k2_D1_var_opt}")
                # print(f"z_var = {z_var_opt}")
                # print(f"D_k2_var = {D_k2_var_opt}")
                # print(f"f_k1_uav_var = {f_k1_uav_var_opt * f_norm}")
                # print(f"f_k2_uav_var = {f_k2_uav_var_opt * f_norm}")
                # print(f"第{id_noma_group + 1}个NOMA组 -> Case2 -> 最小延迟为: : {optimal_value}", flush=True)
                break

            # # 更新为当前CCCP求解出来的最优解
            # hat_p_k1_D1_param.value = p_k1_D1_var.value
            # hat_z_param.value = z_var.value
            # hat_D_k2_param.value = D_k2_var.value
            # hat_f_k1_uav_param.value = f_k1_uav_var.value
            # hat_f_k2_uav_param.value = f_k2_uav_var.value
            # 更新为当前CCCP求解出来的最优解
            if p_k1_D1_var.value is not None:
                hat_p_k1_D1_param.value = p_k1_D1_var.value
            if z_var.value is not None:
                hat_z_param.value = z_var.value
            if D_k2_var.value is not None:
                hat_D_k2_param.value = D_k2_var.value
            if f_k1_uav_var.value is not None:
                hat_f_k1_uav_param.value = f_k1_uav_var.value
            if f_k2_uav_var.value is not None:
                hat_f_k2_uav_param.value = f_k2_uav_var.value

            # 更新最优目标函数值
            pre_objective_value = current_objective_value

        return optimal_value

    @staticmethod
    def user1_lager_gain_NOMA_CCCP(L_k1, L_k2, tau, D_k1,
                                   g_k1, g_k2, uav_pos_pre,
                                   uav_pos, id_noma_group):
        # print("-------------------------------------------------------- ")
        # print(f"--------- 第{id_noma_group + 1}个NOMA组 -> case1 -> CCCP算法开始迭代 --------- ")
        # print("-------------------------------------------------------- ")
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
        D_k1_max = tau - 0.1
        D_k2_max = tau  # 用户 2 的最大可容忍时延大于用户 1

        max_iterations = 8
        tolerance = 1e-2  # 收敛阈值

        # UAV前后迭代位置差
        uav_diff = np.linalg.norm(uav_pos - uav_pos_pre)
        uav_diff = max(uav_diff, 1e-8)  # 防止除零

        # 最优目标函数值初始化
        optimal_value = float('inf')
        pre_objective_value = float('inf')  # 设置一个初始的前一次目标函数值 对于时延而言足够大就行

        # 声明优化变量
        p_k2_D1_var = cp.Variable(nonneg=True)
        p_k1_D1_var = cp.Variable(nonneg=True)
        f_k1_uav_var = cp.Variable(nonneg=True)
        f_k2_uav_var = cp.Variable(nonneg=True)
        # ---------- 定义参数 (hat 值) ----------
        hat_p_k2_D1_param = cp.Parameter(nonneg=True, value=0.05)
        hat_f_k1_uav_param = cp.Parameter(nonneg=True, value=2e8)
        hat_f_k2_uav_param = cp.Parameter(nonneg=True, value=2e8)

        epsilon = 1e-8

        # 目标函数
        objective = cp.Minimize(
            2 * D_k1
            + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon)
            + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon)
        )

        # 约束条件
        constraints = []

        constraints.append(
            p_k1_D1_var - P_max <= 0
        )

        constraints.append(
            p_k2_D1_var - P_max <= 0
        )

        constraints.append(
            D_k1 * p_k1_D1_var - E_max <= 0
        )

        constraints.append(
            D_k1 * p_k2_D1_var - E_max <= 0
        )

        constraints.append(
            D_k1 + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon) - D_k1_max <= 0
        )

        constraints.append(
            D_k1 + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon) - D_k2_max <= 0
        )

        constraints.append(
            f_k1_uav_var + f_k2_uav_var - f_uav_max_norm <= 0
        )

        constraints.append(
            kappa_uav * ((hat_f_k1_uav_param * f_norm) ** 2 + 2 * hat_f_k1_uav_param * f_norm * (
                    f_k1_uav_var * f_norm - hat_f_k1_uav_param * f_norm)) * C * L_k1
            + kappa_uav * ((hat_f_k2_uav_param * f_norm) ** 2 + 2 * hat_f_k2_uav_param * f_norm * (
                    f_k2_uav_var * f_norm - hat_f_k2_uav_param * f_norm)) * C * L_k2
            + (c1 * (uav_diff ** 3) / (tau ** 2)) + (c2 * (tau ** 2) / uav_diff) - E_uav_max <= 0
        )

        constraints.append(
            L_k1 - D_k1 * B * cp.log(g_k2 * p_k2_D1_var + g_k1 * p_k1_D1_var + sigma2)
            + D_k1 * B * cp.log(g_k2 * hat_p_k2_D1_param + sigma2)
            + D_k1 * B * g_k2 / (g_k2 * hat_p_k2_D1_param + sigma2) * (p_k2_D1_var - hat_p_k2_D1_param) <= 0
        )

        constraints.append(
            L_k2 - D_k1 * B * cp.log(sigma2 + g_k2 * p_k2_D1_var)
            + D_k1 * B * cp.log(sigma2) <= 0
        )

        # 变量约束
        constraints += [
            p_k1_D1_var >= 0,
            p_k2_D1_var >= 0,
            f_k1_uav_var >= 0,
            f_k2_uav_var >= 0
        ]

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
                # print(f"CCCP算法迭代过程收敛在第 {iteration + 1} 轮")
                iteration_nums = iteration + 1
                optimal_value = current_objective_value
                # 输出当前优化结果和目标函数值
                # print(f"p_k1_D1 = {p_k1_D1_opt}")
                # print(f"p_k2_D1 = {p_k2_D1_opt}")
                # print(f"z_var = {z_var_opt}")
                # print(f"D_k2_var = {D_k2_var_opt}")
                # print(f"f_k1_uav_var = {f_k1_uav_var_opt * f_norm}")
                # print(f"f_k2_uav_var = {f_k2_uav_var_opt * f_norm}")
                # print(f"第{id_noma_group + 1}个NOMA组 -> Case1 -> 最小延迟为: : {optimal_value}", flush=True)
                break

            # 更新为当前CCCP求解出来的最优解
            if p_k2_D1_var.value is not None:
                hat_p_k2_D1_param.value = p_k2_D1_var.value
            if f_k1_uav_var.value is not None:
                hat_f_k1_uav_param.value = f_k1_uav_var.value
            if f_k2_uav_var.value is not None:
                hat_f_k2_uav_param.value = f_k2_uav_var.value

            # 更新最优目标函数值
            pre_objective_value = current_objective_value

        return optimal_value

    @staticmethod
    def user2_lager_gain_NOMA_CCCP(L_k1, L_k2, tau, D_k1,
                                   g_k1, g_k2, uav_pos_pre,
                                   uav_pos, id_noma_group):
        # print("-------------------------------------------------------- ")
        # print(f"--------- 第{id_noma_group + 1}个NOMA组 -> case2 -> CCCP算法开始迭代 --------- ")
        # print("-------------------------------------------------------- ")
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
        D_k1_max = tau - 0.1
        D_k2_max = tau  # 用户 2 的最大可容忍时延大于用户 1

        # UAV前后迭代位置差
        uav_diff = np.linalg.norm(uav_pos - uav_pos_pre)
        uav_diff = max(uav_diff, 1e-8)  # 防止除零

        # CCCP算法的最大迭代次数
        max_iterations = 8
        tolerance = 1e-2  # 收敛阈值

        # 最优目标函数值初始化
        optimal_value = float('inf')
        pre_objective_value = float('inf')  # 设置一个初始的前一次目标函数值 对于能耗而言足够大就行
        # 声明优化变量
        p_k1_D1_var = cp.Variable(nonneg=True)
        p_k2_D1_var = cp.Variable(nonneg=True)
        f_k1_uav_var = cp.Variable(nonneg=True)
        f_k2_uav_var = cp.Variable(nonneg=True)
        # ---------- 定义参数 (hat 值) ----------
        hat_p_k1_D1_param = cp.Parameter(nonneg=True, value=0.12)
        hat_f_k1_uav_param = cp.Parameter(nonneg=True, value=2e8)
        hat_f_k2_uav_param = cp.Parameter(nonneg=True, value=2e8)

        epsilon = 1e-8

        # 目标函数
        objective = cp.Minimize(
            2 * D_k1
            + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon)
            + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon)
        )

        # 约束条件
        constraints = []

        constraints.append(
            p_k1_D1_var - P_max <= 0
        )

        constraints.append(
            p_k2_D1_var - P_max <= 0
        )

        constraints.append(
            D_k1 * p_k1_D1_var - E_max <= 0
        )

        constraints.append(
            D_k1 * p_k2_D1_var - E_max <= 0
        )

        constraints.append(
            D_k1 + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon) - D_k1_max <= 0
        )

        constraints.append(
            D_k1 + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon) - D_k2_max <= 0
        )

        constraints.append(
            f_k1_uav_var + f_k2_uav_var - f_uav_max_norm <= 0
        )

        constraints.append(
            kappa_uav * ((hat_f_k1_uav_param * f_norm) ** 2 + 2 * hat_f_k1_uav_param * f_norm * (
                    f_k1_uav_var * f_norm - hat_f_k1_uav_param * f_norm)) * C * L_k1
            + kappa_uav * ((hat_f_k2_uav_param * f_norm) ** 2 + 2 * hat_f_k2_uav_param * f_norm * (
                    f_k2_uav_var * f_norm - hat_f_k2_uav_param * f_norm)) * C * L_k2
            + (c1 * (uav_diff ** 3) / (tau ** 2)) + (c2 * (tau ** 2) / uav_diff) - E_uav_max <= 0
        )

        constraints.append(
            L_k1 - D_k1 * B * cp.log(sigma2 + g_k1 * p_k1_D1_var) + D_k1 * B * cp.log(sigma2) <= 0
        )

        constraints.append(
            L_k2 - D_k1 * B * cp.log(g_k1 * p_k1_D1_var + g_k2 * p_k2_D1_var + sigma2)
            + D_k1 * B * cp.log(g_k1 * hat_p_k1_D1_param + sigma2)
            + D_k1 * B * g_k1 / (g_k1 * hat_p_k1_D1_param + sigma2) * (p_k1_D1_var - hat_p_k1_D1_param) <= 0
        )

        # 变量约束
        constraints += [
            p_k1_D1_var >= 0,
            p_k2_D1_var >= 0,
            f_k1_uav_var >= 0,
            f_k2_uav_var >= 0
        ]

        # 开始迭代
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
                # print(f"CCCP算法迭代过程收敛在第 {iteration + 1} 轮")
                iteration_nums = iteration + 1
                optimal_value = current_objective_value
                # 输出当前优化结果和目标函数值
                # print(f"p_k1_D1 = {p_k1_D1_var_opt}")
                # print(f"p_k2_D1 = {p_k2_D1_var_opt}")
                # print(f"z_var = {z_var_opt}")
                # print(f"D_k2_var = {D_k2_var_opt}")
                # print(f"f_k1_uav_var = {f_k1_uav_var_opt * f_norm}")
                # print(f"f_k2_uav_var = {f_k2_uav_var_opt * f_norm}")
                # print(f"第{id_noma_group + 1}个NOMA组 -> Case2 -> 最小延迟为: : {optimal_value}", flush=True)
                break

            # # 更新为当前CCCP求解出来的最优解
            # hat_p_k1_D1_param.value = p_k1_D1_var.value
            # hat_z_param.value = z_var.value
            # hat_D_k2_param.value = D_k2_var.value
            # hat_f_k1_uav_param.value = f_k1_uav_var.value
            # hat_f_k2_uav_param.value = f_k2_uav_var.value
            # 更新为当前CCCP求解出来的最优解
            if p_k1_D1_var.value is not None:
                hat_p_k1_D1_param.value = p_k1_D1_var.value
            if f_k1_uav_var.value is not None:
                hat_f_k1_uav_param.value = f_k1_uav_var.value
            if f_k2_uav_var.value is not None:
                hat_f_k2_uav_param.value = f_k2_uav_var.value

            # 更新最优目标函数值
            pre_objective_value = current_objective_value

        return optimal_value

    @staticmethod
    def OMA_CCCP(L_k1, L_k2, tau, D_k1,
                 g_k1, g_k2, uav_pos_pre,
                 uav_pos, id_noma_group):
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
        D_k1_max = tau - 0.1
        D_k2_max = tau  # 用户 2 的最大可容忍时延大于用户 1
        D_k2 = D_k2_max - D_k1  # 简化用户 2 所需的传输时间计算

        max_iterations = 8
        tolerance = 1e-2  # 收敛阈值

        # UAV前后迭代位置差
        uav_diff = np.linalg.norm(uav_pos - uav_pos_pre)
        uav_diff = max(uav_diff, 1e-8)  # 防止除零

        # 最优目标函数值初始化
        optimal_value = float('inf')
        pre_objective_value = float('inf')  # 设置一个初始的前一次目标函数值 对于时延而言足够大就行

        # 声明优化变量
        p_k1_D1_var = cp.Variable(nonneg=True)
        p_k2_D2_var = cp.Variable(nonneg=True)
        f_k1_uav_var = cp.Variable(nonneg=True)
        f_k2_uav_var = cp.Variable(nonneg=True)
        # ---------- 定义参数 (hat 值) ----------
        hat_f_k1_uav_param = cp.Parameter(nonneg=True, value=2e8)
        hat_f_k2_uav_param = cp.Parameter(nonneg=True, value=2e8)

        epsilon = 1e-8

        # 目标函数
        objective = cp.Minimize(
            2 * D_k1 + D_k2
            + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon)
            + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon)
        )

        # 约束条件
        constraints = []

        constraints.append(
            p_k1_D1_var - P_max <= 0
        )

        constraints.append(
            p_k2_D2_var - P_max <= 0
        )

        constraints.append(
            D_k1 * p_k1_D1_var - E_max <= 0
        )

        constraints.append(
            D_k2 * p_k2_D2_var - E_max <= 0
        )

        # 27h
        constraints.append(
            D_k1 + (L_k1 * C / f_norm) * cp.inv_pos(f_k1_uav_var + epsilon) - D_k1_max <= 0
        )

        # 27i
        constraints.append(
            D_k2 + (L_k2 * C / f_norm) * cp.inv_pos(f_k2_uav_var + epsilon) - D_k2_max <= 0
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

        # 47
        constraints.append(
            L_k1 - D_k1 * B * cp.log(g_k1 * p_k1_D1_var + sigma2) + D_k1 * B * cp.log(sigma2) <= 0
        )

        # 48
        constraints.append(
            L_k2 - D_k2 * B * cp.log(sigma2 + g_k2 * p_k2_D2_var) + D_k2 * B * cp.log(sigma2) <= 0
        )

        # 变量约束
        constraints += [
            p_k1_D1_var >= 0,
            p_k2_D2_var >= 0,
            f_k1_uav_var >= 0,
            f_k2_uav_var >= 0
        ]

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
                # print(f"CCCP算法迭代过程收敛在第 {iteration + 1} 轮")
                iteration_nums = iteration + 1
                optimal_value = current_objective_value
                # 输出当前优化结果和目标函数值
                # print(f"p_k1_D1 = {p_k1_D1_opt}")
                # print(f"p_k2_D1 = {p_k2_D1_opt}")
                # print(f"z_var = {z_var_opt}")
                # print(f"D_k2_var = {D_k2_var_opt}")
                # print(f"f_k1_uav_var = {f_k1_uav_var_opt * f_norm}")
                # print(f"f_k2_uav_var = {f_k2_uav_var_opt * f_norm}")
                # print(f"第{id_noma_group + 1}个NOMA组 -> Case1 -> 最小延迟为: : {optimal_value}", flush=True)
                break

            # 更新为当前CCCP求解出来的最优解
            if f_k1_uav_var.value is not None:
                hat_f_k1_uav_param.value = f_k1_uav_var.value
            if f_k2_uav_var.value is not None:
                hat_f_k2_uav_param.value = f_k2_uav_var.value

            # 更新最优目标函数值
            pre_objective_value = current_objective_value

        return optimal_value
