import numpy as np
import pandas as pd


class UserTask:
    # 用户的任务信息，包括卸载数据量、计算密度、信道增益、任务时延要求
    def __init__(self, L=None, C=None, g=None, T_max=None):
        if T_max is None:
            T_max = []
        if g is None:
            g = []
        if C is None:
            C = []
        if L is None:
            L = []

        self.L = L
        self.C = C
        self.g = g
        self.T_max = T_max

    def update(self, L, C, g, T_max):

        # 按照信道增益从低到高排序
        sorted_q = np.argsort(g)

        self.L = np.take(L, sorted_q)
        self.C = C
        self.g = np.take(g, sorted_q)
        self.T_max = np.take(T_max, sorted_q)

    def display_Info(self):
        print('系统中用户信息如下(按信道增益从小到大排列)：\n')
        n = len(self.L)
        print('用户数量：%d ' % n)
        print('用户数据量：', self.L)
        print('用户信道增益：', self.g)
        print('用户延迟：', self.T_max)


class MecSystem:
    # MEC系统参数
    def __init__(self, B, sigma2, k1, k2, fl_max, fm_max, P_max):
        self.B = B
        self.sigma2 = sigma2
        self.k1 = k1
        self.k2 = k2
        self.fl_max = fl_max
        self.fm_max = fm_max
        self.P_max = P_max

    def display_Info(self):
        print('系统参数如下：\n')

    def set(self, B, sigma2, k1, k2, fl_max, fm_max, P_max):
        self.B = B
        self.sigma2 = sigma2
        self.k1 = k1
        self.k2 = k2
        self.fl_max = fl_max
        self.fm_max = fm_max
        self.P_max = P_max


class RIS:
    """
        RIS类：
    """
    def __init__(self, K, coordinate_users, coordinate_RIS, coordinate_UAV,
                 alpha_RUAV=2.3, alpha_UR=3, beta=2):
        """

        :param K:RIS元件个数
        :param coordinate_users:用户坐标
        :param coordinate_RIS: RIS坐标
        :param coordinate_UAV: UAV坐标
        :param alpha_RUAV: 路径损耗因子 RIS -> UAV
        :param alpha_UR: 路径损耗因子 User -> RIS
        :param beta: 莱斯银子3dB = 2
        """
        self.loss_large_dc = None
        self.loss_large_Rc = None
        self.loss_large_d = None
        self.loss_large_Ru = None
        self.loss_large_RB = None
        self.K = K
        self.beta = beta
        self.alpha_RUAV = alpha_RUAV      # RIS -> UAV
        self.alpha_UR = alpha_UR      # User -> RIS
        self.coordinate_RIS = np.array(coordinate_RIS, dtype=float)
        self.coordinate_UAV = np.array(coordinate_UAV, dtype=float)
        self.coordinate_users = np.array(coordinate_users, dtype=float)
        self.Theta_ref = None

    def loss_large(self):
        """
        大尺度衰落模型
            用户 -> RIS
            RIS -> UAV
        :return:
        """
        g0 = 1e-3
        Num_user = self.coordinate_users.shape[0]
        # 大尺度衰落模型
        g = lambda d, alpha: g0 * d ** (-alpha)

        # RIS → UAV 的距离
        d_RB = np.linalg.norm(self.coordinate_RIS - self.coordinate_UAV)
        # 论文中指定 RIS -> UAV 的大尺度衰落
        loss_large_RB = g(d_RB, self.alpha_RUAV)

        loss_large_Ru = np.zeros(Num_user)  # 用户 -> RIS
        loss_large_d = np.zeros(Num_user)  # 用户 -> UAV (直射链路不存在)

        # 算出所有 User 到 RIS 的信道增益
        for n in range(Num_user):
            d_RU = np.linalg.norm(self.coordinate_users[n] - self.coordinate_RIS)
            loss_large_Ru[n] = g(d_RU, self.alpha_RUAV)

        # 保存成员变量（可选）
        self.loss_large_RB = loss_large_RB
        self.loss_large_Ru = loss_large_Ru

        return loss_large_RB, loss_large_Ru

    def set_ris_parameters(self, cos_phi):
        """
        设置RIS的信道增益
        :param cos_phi:公式中\cos\varphi_{l}(t)参数
        :return:void
        """
        d_over_lambda = 0.5  # 元素间距与波长的比值为半波长0.5
        # 计算相位响应向量
        phase_vector = np.zeros(self.K, dtype=np.complex128)

        for k in range(self.K):
            # 根据公式计算每个RIS元素的相位响应
            phase_shift = 2 * np.pi * d_over_lambda * k * cos_phi
            phase_vector[k] = np.exp(1j * phase_shift)

        # 创建对角相位矩阵 (反射方向)
        self.Theta_ref = np.diag(phase_vector)

    def channel_random(self, Num_user, Num_collab):
        g_k = np.zeros(Num_user)  # 用户信道增益

        for n in range(Num_user):
            # User n → RIS 信道 (K x 1 向量)
            h_RU = self.loss_large_Ru[n] * np.ones((self.K, 1))

            # RIS -> UAV (1 x K 向量)
            h_RB = self.loss_large_RB * (
                    np.sqrt(self.beta / (1 + self.beta)) +
                    np.sqrt(0.5 / (1 + self.beta)) * (
                            np.random.normal(0, 1, (1, self.K)) + 1j * np.random.normal(0, 1, (1, self.K))
                    )
            )

            # User n -> RIS -> UAV
            # 使用反射方向的相位矩阵
            composite_channel = h_RB @ self.Theta_ref @ h_RU
            g_k[n] = np.abs(composite_channel[0, 0]) ** 2  # 信道增益 = 对复数h取模的平方|h|^2

        return g_k[0], g_k[1]

    def channel_simulation(self, Theta, Num_user, g_state):
        # print(g_state)

        g = np.zeros(Num_user)  # 信道增益

        # 不考虑直连信道
        # gd_real = np.sqrt(0.final) * g_state[3 * Num_user: 4 * Num_user]  # 直连信道增益 实部
        # gd_imaginary = np.sqrt(0.final) * g_state[4 * Num_user: final * Num_user]  # 直连信道增益 虚部
        # gd_real = np.zeros(Num_user)
        # gd_imaginary = np.zeros(Num_user)

        gRB_real = np.sqrt(0.5) * g_state[0: 1 * self.K]  # 莱斯信道增益 实部
        gRB_imaginary = np.sqrt(0.5) * g_state[1 * self.K:  2 * self.K]  # 莱斯信道增益 实部

        # 计算信道增益
        h_RB = self.loss_large_RB * (np.sqrt(self.beta / (1 + self.beta)) + np.sqrt(1 / (1 + self.beta)) * (gRB_real + 1j * gRB_imaginary))  # Los信道 + Nlos信道

        # print(self.loss_large_Ru)

        for n in range(0, Num_user):
            h_RU = self.loss_large_Ru[n] * np.ones((self.K, 1))  # Los信道
            # hd = self.loss_large_d[n] * (gd_real[n] + 1j * gd_imaginary[n])  # 瑞利信道

            # 不考虑直连信道
            hd = 0
            g[n] = np.linalg.norm(h_RB @ Theta @ h_RU + hd)

        return g


class DATA:
    def __init__(self, N, K, MAX, L=[], C=[], T_max=[]):
        # 数据集类
        self.N = N  # 用户数量
        self.K = K  # 反射面数量
        self.L = L  # 数据量
        self.C = C  # 计算密度
        self.T_max = T_max  # 任务时延
        self.MAX = MAX  # 仿真时隙数量

    def generate_data(self):
        print("正在生成仿真数据...")
        State_data = np.zeros((self.MAX, 5 * self.N + 2 * self.K))

        for T in range(0, self.MAX):
            State_data[T, 0:self.N] = np.ones(self.N) * self.L  # 单位为Mbits
            State_data[T, self.N:2 * self.N] = np.ones(self.N) * self.C  # 计算密度
            State_data[T, 2 * self.N:3 * self.N] = np.random.uniform(self.T_max[0], self.T_max[1],
                                                                     self.N)  # 单位为s # 任务最大时延

            # K + N 个信道，2 *（K+N）个状态
            # 生成2 *（K+N）个高斯随机变量，前K＋N个为实部，后K+N个为虚部
            State_data[T, 3 * self.N: 5 * self.N + 2 * self.K] = np.random.normal(0, 1, 2 * (self.K + self.N))

        # print(State_data)

        df = pd.DataFrame(State_data)
        # 将数据框写入 CSV 文件
        df.to_csv('State_data.csv', index=False, header=False)
        print("仿真数据已写入文件.")

        def generate_data2(self):
            print("正在生成仿真数据...")
            State_data = np.zeros((self.MAX, 5 * self.N + 2 * self.K))

            for T in range(0, self.MAX):
                State_data[T, 0:self.N] = np.ones(self.N) * self.L  # 单位为Mbits
                State_data[T, self.N:2 * self.N] = np.ones(self.N) * self.C  # 计算密度
                State_data[T, 2 * self.N:3 * self.N] = np.random.uniform(self.T_max[0], self.T_max[1],
                                                                         self.N)  # 单位为s # 任务最大时延

                # K + N 个信道，2 *（K+N）个状态
                # 生成2 *（K+N）个高斯随机变量，前K＋N个为实部，后K+N个为虚部
                State_data[T, 3 * self.N: 5 * self.N + 2 * self.K] = np.random.normal(0, 1, 2 * (self.K + self.N))

            # print(State_data)

            df = pd.DataFrame(State_data)
            # 将数据框写入 CSV 文件
            df.to_csv('State_data.csv', index=False, header=False)
            print("仿真数据已写入文件.")

    def read_data(self):
        # 指定要读取的文件名
        file_name = 'State_data.csv'

        # 使用 read_csv 函数读取 CSV 文件
        df = pd.read_csv(file_name, header=None, index_col=None)
        data = df.to_numpy()

        # 打印读取的数据
        print("仿真数据已读取")
        # print(data)
        return data



