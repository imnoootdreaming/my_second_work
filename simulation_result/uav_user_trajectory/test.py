import numpy as np

# 给定10个time_slot的reward值
rewards = np.array([
    2.348326,
    2.432961,
    2.360596,
    2.47852,
    2.498603,
    2.467096,
    2.477565,
    2.451049,
    2.437304,
    2.424969
])

N = 3  # UAV数量
np.random.seed(42)  # 为了可复现

all_delays = []
total_delays = []

for R in rewards:
    # 平均时延
    avg_delay = -np.log(R / N)
    # 加浮动，保持总 reward 不变（初步浮动，后续可调整归一化）
    eps = np.random.uniform(-0.02, 0.02, size=N)
    delays = avg_delay + eps
    # 调整归一化，使得 sum(exp(-x_i)) = R
    factor = np.log(np.sum(np.exp(-delays)) / R)
    delays += factor  # 调整
    all_delays.append(delays)
    total_delays.append(np.sum(delays))

all_delays = np.array(all_delays)
total_delays = np.array(total_delays)

# 输出每个 UAV 的时延和总时延
for t in range(len(rewards)):
    print(f"Time slot {t+1}: UAV delays = {all_delays[t]}, Total delay = {total_delays[t]}")
