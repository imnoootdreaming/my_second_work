import gym
from gym import spaces
import numpy as np
import torch
import my_class_new
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


# 根据自己的场景搭建的通信环境
class MyEnv(gym.Env):
    def __init__(self, K, noma_group_num, uav_num, users_num_per_noma_group,
                 uav_pos, ris_pos, users_pos,
                 users_center, users_radius,
                 total_time_slots, seed, uav_fixed=False):
        super(MyEnv, self).__init__()
        self.t = 0  # 初始化当前读取时隙
        self.K = K
        self.noma_group_num = noma_group_num
        self.uav_num = uav_num
        self.users_num_per_noma_group = users_num_per_noma_group
        self.num_users = users_num_per_noma_group * noma_group_num
        self.ris_pos = np.array(ris_pos, dtype=float)
        self.total_time_slots = total_time_slots
        self.L_k1 = 150 * 1e3
        self.L_k2 = 150 * 1e3
        self.tau = 0.5
        self.speed_uav_max = 10
        self.x_max = 500
        self.y_max = 500  # 所考虑的仿真范围大小为 500*500m
        self.safe_distance = 5  # UAV 之间的安全距离为 safe_distance
        # UAV & 用户初始位置
        self.init_cur_uav_pos = np.array(uav_pos, dtype=float)
        self.init_users_pos = np.array(users_pos, dtype=float)
        self.uav_fixed = uav_fixed
        # 当前状态位置
        self.cur_uav_pos = self.init_cur_uav_pos.copy()
        self.users_pos = self.init_users_pos.copy()
        self.users_center = users_center
        self.users_radius = users_radius

        # 固定随机种子，保证用户的移动轨迹在每次 reset 后可复现
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # --- Gauss-Markov 移动模型参数 ---
        # Parameters taken from:
        # Online Trajectory and Resource Optimization for Stochastic UAV-Enabled MEC Systems
        self.alpha = 0.4  # 记忆水平/记忆系数
        self.v_bar = np.array([1.0, 0.0])  # 渐近速度均值（地面用户的二维向量）
        self.sigma_bar = 2.0  # 渐近标准差
        self.init_user_velocity = np.array([1.0, 0.0])  # 初始速度
        # --- Gauss-Markov 移动模型参数 ---

        # 预生成用户轨迹 (t, num_users, 3)
        self.precomputed_users_traj = self._generate_user_trajectory()

        # 定义 Agent 数量
        self.n = self.uav_num + 1  # 所有的 UAV agent + 一个 RIS agent

        # 初始化奖励函数 和 线程池
        self.Reward = my_class_new.Reward(self.K, self.ris_pos, self.uav_num, self.x_max, self.y_max,
                                          self.safe_distance)

        # 动作空间
        self.action_space = {}
        self.epsilon = 1e-8
        # 每个UAV单独动作空间
        self.action_space["uav"] = {
            f"uav_{i}": spaces.Box(
                low=np.array([self.epsilon, self.epsilon]),  # [角度, 距离]
                high=np.array([2 * np.pi, self.tau * self.speed_uav_max]),
                dtype=np.float32
            )
            for i in range(self.uav_num)
        }

        # RIS agent
        self.action_space["ris"] = spaces.Box(
            low=np.concatenate([
                np.zeros(self.K) + self.epsilon,  # RIS 相位
                np.zeros(self.uav_num) + self.epsilon,  # first_trans_period 最小 0 加上保护边界 防止除以0现象
                np.zeros(self.users_num_per_noma_group * self.noma_group_num) + self.epsilon  # UAV 选择 最小 0
            ]),
            high=np.concatenate([
                np.ones(self.K) * 2 * np.pi,  # RIS 相位最大 2π
                np.ones(self.uav_num) * self.tau,  # first_trans_period 上界
                np.ones(self.users_num_per_noma_group * self.noma_group_num) * self.uav_num  # UAV 选择上界
            ]),
            dtype=np.float32
        )

        # 观测空间
        self.observation_space = {}

        # UAV 观测
        # ① UAV自己的坐标 ② RIS -> 当前 UAV 的信道 ③ 所有用户的坐标
        self.observation_space["uav"] = {
            f"uav_{i}": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(3 +
                       self.K +
                       self.users_num_per_noma_group * self.noma_group_num * 3,),
                dtype=np.float32
            )
            for i in range(self.uav_num)
        }

        # RIS 观测
        # ① 用户 -> RIS 信道 ② RIS -> UAV 信道
        self.observation_space["ris"] = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.users_num_per_noma_group * self.noma_group_num * self.K +
                   self.uav_num * self.K,),
            dtype=np.float32
        )

    def _generate_user_trajectory(self):
        """
        Gauss-Markov 模型轨迹
        """
        num_users = self.users_num_per_noma_group * self.noma_group_num
        traj = np.zeros((self.total_time_slots, num_users, 3), dtype=float)
        # 使用二维向量表示速度，因为用户在地面移动
        velocities = np.zeros((self.total_time_slots, num_users, 2), dtype=float)

        # 初始化起始位置和速度
        traj[0] = self.init_users_pos.copy()
        velocities[0] = np.tile(self.init_user_velocity, (num_users, 1))

        # 循环为所有时间片生成轨迹
        for t in range(self.total_time_slots - 1):
            # 为所有用户生成随机分量
            # 对应于 Gauss-Markov 模型中的随机过程 w_k[n]
            random_component = self.rng.normal(size=(num_users, 2))

            # 根据高斯-马尔可夫模型方程更新速度:
            # v_k[n+1] = α*v_k[n] + (1-α)*v_bar + sqrt(1-α^2)*σ_bar*w_k[n]
            velocities[t + 1] = (self.alpha * velocities[t] +  # self.alpha: α (记忆水平); velocities[t]: v_k[n] (当前速度)
                                 (1 - self.alpha) * self.v_bar +  # self.v_bar: v_bar (渐进平均速度)
                                 np.sqrt(
                                     1 - self.alpha ** 2) * self.sigma_bar * random_component)  # self.sigma_bar: σ_bar (渐进标准差)

            # 根据上一个时间片的速度更新位置: p_k[n+1] = p_k[n] + v_k[n]*Δt
            # traj[t, :, :2]: p_k[n] (当前位置)
            # velocities[t]: v_k[n] (当前速度)
            # self.tau: Δt (时间片时长)
            traj[t + 1, :, :2] = traj[t, :, :2] + velocities[t] * self.tau

            # 保持z坐标不变，因为用户在地面上
            traj[t + 1, :, 2] = traj[t, :, 2]

        return traj

    def update_users_pos(self):
        """
        用户移动：直接取预生成轨迹
            如果用户移动超出边界 则拉回边界
        """
        self.users_pos = self.precomputed_users_traj[self.t]
        # 限制用户在给定圆形区域内
        for i in range(self.users_pos.shape[0]):
            pos_2d = self.users_pos[i, :2]  # 只取 x, y
            vec = pos_2d - self.users_center[:2]
            dist = np.linalg.norm(vec)
            if dist > self.users_radius:
                # 拉回边界点
                clipped = self.users_center[:2] + vec / dist * self.users_radius
                self.users_pos[i, 0] = clipped[0]
                self.users_pos[i, 1] = clipped[1]

    # 用于执行动作
    def step(self, actions, outer_iter):
        # numpy 版本映射
        # === UAV 动作映射 ===
        uav_actions_dict = actions["uav"]
        diff_thetas = []
        diff_distances = []

        for i in range(self.uav_num):
            act = np.array(uav_actions_dict[f"uav_{i}"])
            low = self.action_space["uav"][f"uav_{i}"].low
            high = self.action_space["uav"][f"uav_{i}"].high
            act = act * (high - low) + low
            # act = np.clip(act, low, high)
            diff_thetas.append(act[0])
            diff_distances.append(act[1])
        # ------------------ 随机 UAV 轨迹 ------------------
        # 随机生成 UAV 的角度与距离动作
        # diff_thetas = np.random.uniform(
        #     low=self.epsilon,
        #     high=2 * np.pi,
        #     size=self.uav_num
        # )
        #
        # diff_distances = np.random.uniform(
        #     low=self.epsilon,
        #     high=self.tau * self.speed_uav_max,
        #     size=self.uav_num
        # )
        # # ------------------ 随机 UAV 轨迹 ------------------
        if self.uav_fixed:
            # ------------------ 固定 UAV 轨迹 ------------------
            diff_theta = np.zeros(self.uav_num)
            diff_distance = np.zeros(self.uav_num)
            # ------------------ 固定 UAV 轨迹 ------------------
        else:
            diff_theta = np.array(diff_thetas)
            diff_distance = np.array(diff_distances)

        # === RIS 动作映射 ===
        ris_action = actions["ris"]
        ris_low = self.action_space["ris"].low
        ris_high = self.action_space["ris"].high
        # 直接 clip 动作到 [low, high] 范围
        ris_action = ris_action * (ris_high - ris_low) + ris_low
        # RIS agent
        phase_ris = ris_action[:self.K]  # 相位
        # 每个 NOMA 组的首个调度时隙
        first_trans_period = ris_action[self.K:self.K + self.noma_group_num]
        # ------------------ 随机传输时隙 ------------------
        # first_trans_period = np.random.uniform(low=self.epsilon,
        #                                        high=self.tau,
        #                                        size=self.uav_num)
        # ------------------ 随机传输时隙 ------------------
        # 每个用户选择的 UAV (长度 = num_users)
        # ------------------ 用户选择 ------------------
        uav_selection = np.ceil(ris_action[self.K + self.noma_group_num:]).astype(int) - 1
        # print(f"============= UAV 选择系数为: {uav_selection} =============")
        uav_selection = np.clip(uav_selection, 0, self.uav_num - 1)  # DRL 优化选择 UAV
        # ------------------ 随机选择 ------------------
        # uav_selection = np.random.randint(0, self.uav_num, size=self.uav_num)
        # ------------------ 用户选择 ------------------

        RIS = my_class_new.RIS(self.K, self.users_pos, self.ris_pos, self.cur_uav_pos)  # 初始化RIS对象

        # 下一时刻 UAV 位置
        next_uav_pos = self.cur_uav_pos + np.stack([diff_distance * np.cos(diff_theta),
                                                    diff_distance * np.sin(diff_theta),
                                                    np.zeros_like(diff_distance * np.cos(diff_theta))], axis=1)

        # ------------------ 设置 RIS 信道增益 ------------------
        RIS.set_ris_parameters(phase_ris)  # DRL 优化 RIS 相位
        # RIS.set_random_ris_parameters()  # 随机 RIS 相位
        # ------------------ 设置 RIS 信道增益 ------------------

        # 设置大尺度衰落
        RIS.loss_large()
        # 设置 User -> RIS 的响应
        RIS.set_user_ris_phase_response()
        # 设置 RIS -> UAV 的响应
        RIS.set_ris_uav_phase_response(next_uav_pos=next_uav_pos)

        # NOTE - 信道增益计算
        # 返回形状
        # h_user_2_uav -> (num_uav, num_user)
        # h_user_2_ris -> (num_user, K)
        # h_ris_2_uav -> (num_uav, K)
        h_user_2_uav, h_user_2_ris, h_ris_2_uav = \
            RIS.channel_compute(self.users_num_per_noma_group * self.noma_group_num, self.uav_num,
                                self.noma_group_num, self.t, self.total_time_slots, self.seed)

        # NOTE - 利用CCCP求解以计算奖励
        total_reward, reward, total_delay = self.Reward.reward_compute(L_k1=self.L_k1,
                                                                       L_k2=self.L_k2,
                                                                       tau=self.tau,
                                                                       D_k1=first_trans_period,
                                                                       channel_gain=h_user_2_uav,
                                                                       uav_pos_pre=self.cur_uav_pos,
                                                                       uav_pos=next_uav_pos,
                                                                       choose=uav_selection,
                                                                       noma_group_num=self.noma_group_num,
                                                                       uav_fixed=self.uav_fixed,
                                                                       access_approache='NCO-NOMA')

        self.t = self.t + 1  # 更新到下一个时隙

        BOLD_PURPLE = "\033[1;95m"  # 加粗的紫色
        RESET = "\033[0m"  # 重置颜色
        print(
            f"\n{BOLD_PURPLE}——————————————————— 第{outer_iter + 1}次迭代 "
            f"-> 第{self.t}个时隙 "
            f"-> 奖励：{total_reward:.3f} "
            f"-> 时延 ： {total_delay:.3f} ———————————————————{RESET} \n"
        )

        if self.t < self.total_time_slots:
            # 更新UAV位置
            self.cur_uav_pos = next_uav_pos
            # 更新用户位置 (用户移动)
            self.update_users_pos()
            # 重新计算 用户->RIS 和 RIS->UAV 信道
            RIS = my_class_new.RIS(self.K, self.users_pos, self.ris_pos, self.cur_uav_pos)
            # ------------------ 设置 RIS 信道增益 ------------------
            RIS.set_ris_parameters(phase_ris)  # DRL 优化 RIS 相位
            # RIS.set_random_ris_parameters()  # 随机 RIS 相位
            # ------------------ 设置 RIS 信道增益 ------------------

            # 设置大尺度衰落
            RIS.loss_large()
            # 设置 User -> RIS 的响应
            RIS.set_user_ris_phase_response()
            # 设置 RIS -> UAV 的响应
            RIS.set_ris_uav_phase_response(next_uav_pos=next_uav_pos)
            # NOTE - 下一时刻信道增益计算
            # 返回形状
            # next_h_user_2_uav -> (num_uav, num_user)
            # next_h_user_2_ris -> (num_user, K)
            # next_h_ris_2_uav -> (num_uav, K)
            next_h_user_2_uav, next_h_user_2_ris, next_h_ris_2_uav = \
                RIS.channel_compute(self.users_num_per_noma_group * self.noma_group_num, self.uav_num,
                                    self.noma_group_num, self.t, self.total_time_slots, self.seed)
            done = 0
        else:
            # 终止状态的下一个状态就是当前状态
            next_h_user_2_ris = h_user_2_ris
            next_h_ris_2_uav = h_ris_2_uav
            done = 1

        next_state_dict = {"uav": {}, "ris": None}

        for i in range(self.uav_num):
            uav_coord = self.cur_uav_pos[i]
            ris_to_uav = next_h_ris_2_uav[i]
            users_coord = self.users_pos.flatten()
            uav_obs = np.concatenate([uav_coord, ris_to_uav, users_coord]).astype(np.float32)
            next_state_dict["uav"][f"uav_{i}"] = uav_obs

        ris_obs = np.concatenate([next_h_user_2_ris.flatten(), next_h_ris_2_uav.flatten()]).astype(np.float32)
        next_state_dict["ris"] = ris_obs

        return next_state_dict, float(total_reward), reward, done

    # 用于重置环境
    def reset(self):
        self.t = 0  # 时隙初始化
        # 初始化到最初状态
        # init_cur_uav_pos: 初始 UAV 坐标
        # init_user_pos: 初始化用户坐标
        #
        # 恢复初始 UAV 和用户位置
        self.cur_uav_pos = self.init_cur_uav_pos.copy()
        self.users_pos = self.precomputed_users_traj[self.t].copy()  # 读取轨迹的初始点
        # 初始化 RIS 对象
        RIS = my_class_new.RIS(self.K, self.users_pos, self.ris_pos, self.cur_uav_pos)
        # 初始化 RIS 相位为 0
        # 没有用到这个值 只是为了保证计算得到h_user_2_ris h_ris_2_uav 而不会使用 h_user_2_uav
        phase_ris = np.zeros(self.K)
        # ------------------ 设置 RIS 信道增益 ------------------
        RIS.set_ris_parameters(phase_ris)
        # RIS.set_random_ris_parameters()  # 随机 RIS 相位
        # ------------------ 设置 RIS 信道增益 ------------------
        # 设置大尺度衰落
        RIS.loss_large()
        # 设置 User -> RIS 的相位响应
        RIS.set_user_ris_phase_response()
        # 这里直接用当前 UAV 位置（初始位置），不是下一步的
        RIS.set_ris_uav_phase_response(next_uav_pos=self.cur_uav_pos)

        h_user_2_uav, h_user_2_ris, h_ris_2_uav = \
            RIS.channel_compute(self.users_num_per_noma_group * self.noma_group_num, self.uav_num,
                                self.noma_group_num, self.t, self.total_time_slots, self.seed)

        # ===== 修改后的 init_state_dict =====
        init_state_dict = {"uav": {}, "ris": None}
        for i in range(self.uav_num):
            uav_coord = self.cur_uav_pos[i]
            ris_to_uav = h_ris_2_uav[i]
            users_coord = self.users_pos.flatten()
            uav_obs = np.concatenate([uav_coord, ris_to_uav, users_coord]).astype(np.float32)
            init_state_dict["uav"][f"uav_{i}"] = uav_obs

        ris_obs = np.concatenate([h_user_2_ris.flatten(), h_ris_2_uav.flatten()]).astype(np.float32)
        init_state_dict["ris"] = ris_obs

        return init_state_dict  # reward, done, info can't be included

    # 用于获取当前时隙下的 UAV 坐标
    def getPosUAV(self):
        return self.cur_uav_pos

    def getPosUser(self):
        return self.users_pos
