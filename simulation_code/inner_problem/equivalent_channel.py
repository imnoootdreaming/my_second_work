import numpy as np
import myClass

def generate_equivalent_channel(uav_pos, ris_pos, users_pos):
    """

    Returns:等效透射信道和反射信道

    """
    K = 20  # RIS元件个数
    RIS = myClass.RIS(K, users_pos, ris_pos, uav_pos)  # 创建实例

    # 数据类型转换
    uav_pos = np.array(uav_pos, dtype=float)
    ris_pos = np.array(ris_pos, dtype=float)
    users_pos = np.array(users_pos, dtype=float)

    # 相移计算：
    # 计算RIS到UAV的距离向量以使用np.linalg.norm求解距离
    distance = np.linalg.norm(uav_pos - ris_pos)
    # 计算cos(φ_l(t)) = (x_UAV - x_RIS) / d_l(t)
    # 根据公式，这里只考虑x方向的分量
    cos_phi = (uav_pos[0] - ris_pos[0]) / distance

    # 设置RIS参数
    RIS.set_ris_parameters(cos_phi)
    # 生成信道
    RIS.loss_large()  # 首先生成大尺度衰弱
    g_k1, g_k2 = RIS.channel_random(Num_user=2, Num_collab=1)  # 等效信道
    # 信道增益是Numpy数组 进行转换
    if isinstance(g_k1, np.ndarray):
        g_k1 = g_k1.item()  # 只提取一个标量值

    if isinstance(g_k2, np.ndarray):
        g_k2 = g_k2.item()

    return g_k1, g_k2