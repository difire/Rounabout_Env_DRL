from Envs.roundabout_v1_1_1x50 import RoundaboutEnv
from Algorithms.SAC.attention_sac import SAC
from Algorithms.Buffer.buffer import ReplayBuffer

from Scripts.base_runing import *


if __name__ == '__main__':
    env = RoundaboutEnv()
    print(env.name)
    env.configure(
        {
            "simulation_frequency": 20,
            "policy_frequency": 5,
            "screen_width": 400,
            "screen_height": 300,
            "centering_position": [0.3, 0.5],
            "scaling": 4,
            "show_trajectories": False,
            "render_agent": False,
            "manual_control": False,
            "offscreen_rendering": True,
            "show_global_paths": True,
            "real_time_rendering": False,
            "duration": 20,
            "vehicles_count": 5,
            "collision_reward": -10,
            "high_speed_reward": 0.2,
            "lane_center_reward": 0.1,
            "jerk_penalty": -1,
            "lane_change_penalty": -0.2,
            "heading_reward": -0.1,
            "overspeed_penalty": -10,
            "off_road_reward": -10,
            "negative_speed": -10,
            "reach_goal_reward": 10,
            "save_experience": False,
            "save_training_avi": False,
        }
    )
    env.reset()
    state_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 定义设备（CPU或GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    replay_buffer = ReplayBuffer(state_dim, action_dim, device, int(1e6))
    # 初始化智能体
    agent = SAC(state_dim, action_dim, max_action, replay_buffer, device)

    # Train0(env, agent, device)
    Test0(env, agent, 'roundabout_v1_1_1x50_Attention_SAC_174015524', 'best')