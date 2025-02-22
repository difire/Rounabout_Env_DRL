from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os
dir_name = 'roundabout_v1_1_1x50_DDPG3_174014446'
dir_path = f'{os.path.dirname(os.path.dirname(__file__))}/Runs/{dir_name}'

event_file_name = [f for f in os.listdir(dir_path) if f.startswith('events.out.tfevents')]
# 指定事件文件路径
event_file_path = f'{dir_path}/{event_file_name[0]}'

# 创建EventAccumulator对象
event_acc = EventAccumulator(event_file_path)

# 读取事件文件中的所有事件
event_acc.Reload()

print(event_acc.Tags())

# 获取特定标签的标量数据
steps_total = event_acc.Scalars('Steps/Total')
steps_average100 = event_acc.Scalars('Steps/Average100')

# 提取步骤和总步数
steps = [event.step for event in steps_total]
total_steps = [event.value for event in steps_total]

# 提取步骤和平均步数
average_steps = [event.value for event in steps_average100]

# 使用matplotlib绘图
plt.figure(figsize=(10, 5))
plt.plot(steps, total_steps, label='Total Steps')
plt.plot(steps, average_steps, label='Average Steps (Last 100 Episodes)')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.legend()
plt.show()

from Utils.EventsDataLoader import EventsDataLoader
data_loader = EventsDataLoader(dir_path)
data_loader.save_img('Reward/Total')
data_loader.save_img('Reward/Average100')
data_loader.save_img('Average/Speed')
data_loader.save_img('Average/Acceleration')
data_loader.save_img('Steps/Total')
data_loader.save_img('Steps/Average100')
data_loader.save_img('Success Rate/Rate')
data_loader.save_img('Success Rate/Average100')
