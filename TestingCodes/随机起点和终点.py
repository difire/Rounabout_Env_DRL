import random
from Envs.roundabout_0_v1_2_1x40 import RoundaboutEnv
env = RoundaboutEnv()
env.configure(
    {
        "simulation_frequency": 20,
        "policy_frequency": 5,
        "screen_width": 600,
        "screen_height": 600,
        "centering_position": [0.3, 0.5],
        "scaling": 6,
        "show_trajectories": False,
        "render_agent": False,
        "manual_control": False,
        "offscreen_rendering": True,
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
    }
)
env.configure(
    {
        "show_trajectories": False,
        "render_agent": False,
        "manual_control": False,
        "offscreen_rendering": False,
        "show_global_paths": True,
        "real_time_rendering": True,
        "save_experience": False,
        "duration": 20,
        "vehicles_count": 5,
    }
)
print(env.road.network.graph)
Start_lane_index_list = ['ser', 'eer', 'ner', 'wer']
End_lane_index_list = ['nx', 'ex', 'sx', 'wx']
for e in range(5):
    env.configure(
        {
            'start_lane_index': (random.choice(Start_lane_index_list), None, 0),
            'goal_lane_index': (None, random.choice(End_lane_index_list), 0)
        }
    )
    env.reset()
    print(env.routes)
    done = False
    while not done:
        env.render(mode='rgb_array')
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        state = next_state


