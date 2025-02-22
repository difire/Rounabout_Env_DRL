import os
import gym
import highway_env
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from collections import deque, namedtuple
import random


# 训练函数
def train(env, agent, _abs_dir_path, episodes=1000, writer=None):
    episode_rewards = []
    step_counts = []
    success_rates = []
    max_success_rate = 0
    Start_lane_index_list = ['ser', 'eer', 'ner', 'wer']
    End_lane_index_list = ['nx', 'ex', 'sx', 'wx']
    if env.config['save_training_avi']:
        from Utils.frames2video import Frames2Video
        frames_save_path = os.path.join(_abs_dir_path, f'{str(time.time_ns())[1:9]}')
        print(f'GIF save path: {frames_save_path}')
        video_writer = Frames2Video(f'{frames_save_path}', frame_width=env.config['screen_width'],
                                    frame_height=env.config['screen_height'], fps=env.config['policy_frequency'])
    for episode in range(episodes):
        env.configure(
            {
                'offscreen_rendering': True,    # 显示画面开关。
                'start_lane_index': (random.choice(Start_lane_index_list), None, 0),
                'goal_lane_index': (None, random.choice(End_lane_index_list), 0)
            }
        )
        state = env.reset()
        total_speed = 0
        total_acceleration = 0
        step_count = 0
        successed = 0
        episode_reward = 0
        done = False

        while True:
            if not env.config['offscreen_rendering']:
                frame = env.render(mode='rgb_array')
                if env.config['save_training_avi']:
                    video_writer.add_frame(frame)
            if done:
                episode_reward = env.total_reward
                break
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            step_count += 1
            # Collect statistics
            total_speed += info['speed']
            total_acceleration += abs(action[0])
            agent.train()
        if episode % 20 == 0:
            print(f"Episode: {episode + 1}, Reward: {episode_reward}, Max Success Rate: {max_success_rate}")
        episode_rewards.append(episode_reward)
        step_counts.append(step_count)

        if env.terminate_flag == 'REACHEND':
            successed = 1
        success_rates.append(successed)
        success_rate = np.mean(success_rates[-100:])
        if success_rate > max_success_rate:
            max_success_rate = success_rate
            print(f"{episode}: New max success rate: {max_success_rate}")
            agent.save(f'{_abs_dir_path}\\best_weight')
        if writer is not None:
            avg_speed = total_speed / step_count
            avg_acceleration = total_acceleration / step_count

            # Log metrics to TensorBoard
            writer.add_scalar('Reward/Total', episode_reward, episode)
            writer.add_scalar('Reward/Average100', np.mean(episode_rewards[-100:]), episode)
            writer.add_scalar('Average/Speed', avg_speed, episode)
            writer.add_scalar('Average/Acceleration', avg_acceleration, episode)
            writer.add_scalar('Steps/Total', step_count, episode)
            writer.add_scalar('Steps/Average100', np.mean(step_counts[-100:]), episode)
            writer.add_scalar('Success Rate/Rate', np.mean(success_rates[-1]), episode)
            writer.add_scalar('Success Rate/Average100', np.mean(success_rates[-100:]), episode)


def Train0(env, agent, device):
    # build save path
    _dir_name = f'Runs\\{env.name}_{agent.name}_{str(time.time_ns())[:9]}'
    _abs_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), _dir_name)
    # Initialize TensorBoard,this will auto make dirs
    writer = SummaryWriter(log_dir=f"{_abs_dir_path}")
    # Log network structures
    writer.add_graph(agent.actor, input_to_model=torch.zeros(1, env.observation_space.shape[1], device=device).to(device))
    # 开始训练
    train(env, agent, _abs_dir_path, episodes=1000, writer=writer)
    agent.save(f'{_abs_dir_path}\\last_weight')
    writer.close()
    env.close()
    print(f'training success. saved as {_abs_dir_path}\\*.pth')
    from Utils.EventsDataLoader import EventsDataLoader
    data_loader = EventsDataLoader(_abs_dir_path)
    data_loader.save_img('Reward/Total')
    data_loader.save_img('Reward/Average100')
    data_loader.save_img('Average/Speed')
    data_loader.save_img('Average/Acceleration')
    data_loader.save_img('Steps/Total')
    data_loader.save_img('Steps/Average100')
    data_loader.save_img('Success Rate/Rate')
    data_loader.save_img('Success Rate/Average100')



def Test0(env, agent, _dir_name, file_name):
    env.configure(
        {
            "show_trajectories": False,
            "render_agent": False,
            "manual_control": False,
            "offscreen_rendering": False,
            "show_global_paths": True,
            "real_time_rendering": False,
            "save_experience": False,
            "duration": 20,
            "vehicles_count": 5,
        }
    )
    env.reset()
    _abs_dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f'Runs\\{_dir_name}')
    agent.load(f'{_abs_dir_path}\\{file_name}_weight')
    frames = []
    Start_lane_index_list = ['ser', 'eer', 'ner', 'wer']
    End_lane_index_list = ['nx', 'ex', 'sx', 'wx']
    from Utils.frames2video import Frames2Video
    save_path = os.path.join(_abs_dir_path, f'{str(time.time_ns())[:9]}')
    print(f'GIF save path: {save_path}')
    video_writer = Frames2Video(f'{save_path}', frame_width=env.config['screen_width'],
                                frame_height=env.config['screen_height'], fps=env.config['policy_frequency'])
    for episode in range(10):
        env.configure(
            {
                'start_lane_index': (random.choice(Start_lane_index_list), None, 0),
                'goal_lane_index': (None, random.choice(End_lane_index_list), 0)
            }
        )
        state = env.reset()
        done = False
        while not done:
            frame = env.render(mode='rgb_array')
            video_writer.add_frame(frame)
            frames.append(frame)
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                frame = env.render(mode='rgb_array')
                video_writer.add_frame(frame)
                frames.append(frame)
    from Utils.Array2GIF import array2gif2
    array2gif2(frames, f'{save_path}.gif', duration=env.config['duration'] * env.config['policy_frequency'] / len(frames))
    env.close()