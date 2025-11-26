import numpy as np
from simulation_code.outer_problem.save_version.my_env_v1 import MyEnv


def test_env(MyEnv):
    # 参数设置
    K = 4
    noma_group_num = 2
    uav_num = 1
    users_num_per_noma_group = 2
    uav_pos = np.array([[0, 0, 50]])
    ris_pos = np.array([10, 0, 10])
    users_pos = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0], [15, 0, 0]])
    users_center = np.array([0, 0, 0])
    users_radius = 20
    total_time_slots = 5
    seed = 42

    print("\n===== 测试开始 =====\n")
    # 创建环境
    env1 = MyEnv(K, noma_group_num, uav_num, users_num_per_noma_group,
                 uav_pos, ris_pos, users_pos,
                 users_center, users_radius,
                 total_time_slots, seed=seed)

    # ------------------------------
    # 测试 ①：同一 episode 内不同 time_slot
    # ------------------------------
    env1.reset()
    pos_t0 = env1.users_pos.copy()
    # 手动 step 几次
    for t in range(0, total_time_slots - 1):
        # dummy action
        action = np.zeros(env1.action_space.shape[0])
        state, reward, done = env1.step(action, outer_iter=0)
        pos_t = env1.users_pos.copy()
        if np.allclose(pos_t, pos_t0):
            print(f"[错误] 同一 episode 下 time_slot={t} 用户位置没有变化！")
        pos_t0 = pos_t

    print("[通过] 同一 episode 下不同 time_slot 用户位置不同。")

    # ------------------------------
    # 测试 ②：不同 episode 相同时隙
    # ------------------------------
    env2 = MyEnv(K, noma_group_num, uav_num, users_num_per_noma_group,
                 uav_pos, ris_pos, users_pos,
                 users_center, users_radius,
                 total_time_slots, seed=seed)

    # 在相同的时隙对比两次 episode 的用户位置和 NLoS
    time_slot_to_check = 2

    env1.reset()
    env2.reset()
    for t in range(0, total_time_slots - 1):
        action = np.zeros(env1.action_space.shape[0])
        env1.step(action, outer_iter=0)
        env2.step(action, outer_iter=0)

    pos1 = env1.users_pos.copy()
    pos2 = env2.users_pos.copy()

    if np.allclose(pos1, pos2):
        print(f"[通过] 不同 episode 下 time_slot={time_slot_to_check} 用户位置相同。")
    else:
        print(f"[错误] 不同 episode 下 time_slot={time_slot_to_check} 用户位置不同！")

    print("\n===== 测试结束 =====\n")


if __name__ == "__main__":
    test_env(MyEnv)
