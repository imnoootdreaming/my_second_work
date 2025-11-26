from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

# 替换为你的 TensorBoard 日志路径
log_path = r"C:\Users\atr\Desktop\MyHardResearchTime\InnovationOne\simulation_codes\outer_problem\new_outer_problem\TD3_result_1119\run1119\TD3_User4_Collab8_20250715-144919"  # 注意：请改成你自己的log路径

# 加载日志文件
ea = event_accumulator.EventAccumulator(log_path)
ea.Reload()

# 指定要导出的 tag
tag_name = 'Reward/Average_Episode_Reward'

# 提取指定 tag 的 scalar 数据
if tag_name in ea.Tags()['scalars']:
    events = ea.Scalars(tag_name)

    # 将数据组织为 DataFrame
    df = pd.DataFrame({
        'step': [e.step for e in events],
        'value': [e.value for e in events],
        'wall_time': [e.wall_time for e in events]
    })

    # 保存为 CSV
    df.to_csv('Average_Episode_Reward.csv', index=False)
    print('保存成功：Average_Episode_Reward.csv')
else:
    print(f'未找到指定的 tag: {tag_name}')
