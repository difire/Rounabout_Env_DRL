LOG = '''
2025年2月18日：
1. 在保存图像的格式上，考虑了透明度通道，使保存的GIF图像包括全局路线和局部目标点。
2. 优化局部目标点的选取，之前由速度大小决定，出现低速行驶时，局部目标点不更新，导致车辆减速甚至倒车，加入局部目标点选取的最小距离，必须大于1米。
3. 转角的变化太大了，加大转角突变惩罚。
4. 在输入数据中加入最大和最小限速，同时设置超出速度范围则回合结束。

2025年2月19日：
1. 输入中数据加入上时刻的加速度和转角，让智能体学习输出与历史输出的差异。
2. 优化起点路径设置代码，使训练过程可以随机选择起点路径.
3. 加入转角的限制，使智能体不能转向过大，最大不超过0.7rad,大约是40度。
4. 将输入数据中的车辆与局部目标点的横纵坐标差值替换为，车辆与局部目标点的距离值和偏差角。

2025年2月20日：
1. 尝试优化代码结构，减少代码复杂度。
2. 加入文字绘制代码，可在动画中查看速度、转角等信息。
3. 优化终点抵达判断机制，使奖励更加精准。之前是局部目标点更新为最后一个即视为抵达终点，修改为增加距离判断，足够近才行。
4. 增加输入特征数量，对每一辆观测车，加入它的长度和宽度数据到输入数据中去。
//5. 在奖励函数中加入对周围车辆的位置判断，根据距离车长和车宽，以及车的航向角，这些数据与安全有关，使智能体学习到相关策略。
'''
print(LOG)

import os
import random
import time
import datetime
import numpy as np
import gym
import pandas as pd
import pygame
from gym import spaces
from gym.utils import seeding
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import CircularLane, LineType, StraightLane, SineLane
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.envs.common.graphics import EnvViewer
from highway_env.envs.common.observation import observation_factory
from highway_env.envs.common.action import action_factory
from highway_env import utils


class EndFlag:
    NOTEND      = 'NOTEND'          # 未结束
    CRASH       = 'CRASH'           # 撞车
    TIMEOVER    = 'TIMEOVER'        # 时间结束
    OFFROAD     = 'OFFROAD'         # 驶离道路
    REACHEND    = 'REACHEND'        # 抵达终点
    REVERSE     = 'REVERSE'         # 倒车行驶（低于最低限速）
    OVERSPEED   = 'OVERSPEED'       # 超速
    UNDEFINE    = 'UNDEFINE'        # 未定义


class BaseEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        # Configuration
        self.config = self.default_config()
        self.configure(config)

        # Seeding
        self.np_random = None
        self.seed()

        # Scene
        self.road: Road = None

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.times = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        self._record_video_wrapper = None
        self.rendering_mode = 'human'
        self.enable_auto_render = False

        self.PERCEPTION_DISTANCE = 50       # 设置车辆感知距离
        self.off_road = False               # 是否驶出道路
        self.start_lane = None              # 起点车道
        self.end_lane = None                # 终点车道
        self.controlled_vehicles = []       # 控制车辆列表
        self.routes = []                    # 路由列表
        self.route = []                     # 路由
        self.route_points = []              # 路由的点序列
        self.local_goal_point = None        # 局部目标点
        self.local_lane_index = None        # 局部车道索引
        self.local_point_info = None        # 投影点相关信息[x, y, longitudinal, lateral]
        self.Episodes = 0  # 回合次数, 上面初始化重置后
        self.Succeed_episode = 0  # 成功的回合次数
        self.terminate_flag = EndFlag.NOTEND  # 终止标志
        self.old_terminate_flag = self.terminate_flag
        self.total_reward = 0
        self.reset()
        csv_title = self.config['observation']['features'] + ['width', 'length']
        self.csv_head_title = csv_title * 1  # 复制列表
        for n in range(1, self.config['observation']['vehicles_count']):
            for t in csv_title:
                self.csv_head_title.append(f'{t}_{n}')
        csv_addition_title = ['dis2centerline', 'can2left', 'can2right', 'dis2goal', 'heading2goal',
                              'heading2centerline',
                              'max_speed', 'min_speed', 'acceleration', 'steering']
        self.csv_head_title = self.csv_head_title + csv_addition_title
        self.csv_data = []

    @property
    def vehicle(self) -> Vehicle:
        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    @vehicle.setter
    def vehicle(self, vehicle: Vehicle) -> None:
        self.controlled_vehicles = [vehicle]

    @staticmethod
    def default_config():
        return {
            "observation":
                {
                    "type": "Kinematics",
                    "vehicles_count": 5,
                    "features": ["presence", "x", "y", "vx", "vy", "heading"],
                    "absolute": True
                },
            'action':
                {
                    'type': 'ContinuousAction',
                    'longitudinal': True,
                    'lateral': True,
                    'acceleration_range': None,
                    'steering_range': [-0.7, 0.7],
                },
            "min_speed": 0,  # [m/s]
            "max_speed": 30,  # [m/s]
            'ego_vehicle_init_speed': 10,   # 自车初始速度
            "other_vehicles_init_speed": 10, # 其它车辆初始速度
            "simulation_frequency": 50,  # [Hz]
            "policy_frequency": 10,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,  # [px]
            "screen_height": 600,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "show_global_paths": False,
            "start_lane_index": ("ser", "ses", 0),
            "goal_lane_index": ("ne", "wx", 0),
            "save_data_path": 'data',
            "save_experience": False,
            "manual_control": False,
            "real_time_rendering": False,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "incoming_vehicle_destination": None,
            "ego_spacing": 2,
            "vehicles_density": 1,
            "duration": 40,
            "vehicles_count": 5,
            "collision_reward": -10,
            "high_speed_reward": 0.5,
            "lane_center_reward": 0.5,
            "jerk_penalty": -1.0,
            "lane_change_penalty": -0.2,
            "heading_reward": -0.1,
            "overspeed_penalty": -10,
            "off_road_reward": -10,
            "negative_speed": -10,
            "reach_goal_reward": 10,
        }

    def seed(self, seed: int = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

    def define_spaces(self) -> None:
        ...

    def _automatic_rendering(self) -> None:
        """
        Automatically render the intermediate frames while an action is still ongoing.

        This allows to render the whole video and not only single steps corresponding to agent decision-making.
        If a RecordVideo wrapper has been set, use it to capture intermediate frames.
        """
        if self.viewer is not None and self.enable_auto_render:
            if self.config["show_trajectories"]:
                for v in self.road.vehicles:
                    v.draw_route(self.viewer)  # 绘制路线
            if self._record_video_wrapper and self._record_video_wrapper.video_recorder:
                self._record_video_wrapper.video_recorder.capture_frame()
            else:
                self.render(self.rendering_mode)

    def render_font(self):
        font_surface = pygame.Surface(self.viewer.sim_surface.get_size(), pygame.SRCALPHA)
        # 加载字体，例如使用pygame自带的字体或者系统字体
        font = pygame.font.Font(None, 20)  # None 表示使用默认字体，数字表示字体大小
        speed_surface = font.render(f'Speed: {self.vehicle.speed:.3f} m/s', True, (0, 255, 255))  # 渲染文字，True表示抗锯齿，(255, 255, 255)是颜色（白色）
        accel_surface = font.render(f'Accel: {self.vehicle.action["acceleration"]:.3f} m/ss', True, (255, 0, 255))
        steel_surface = font.render(f'Steel: {self.vehicle.action["steering"]:.3f} rad', True, (255, 255, 0))
        state_surface = font.render(f'State: {self.terminate_flag}', True, (0, 0, 255))
        old_state_surface = font.render(f'Last_S: {self.old_terminate_flag}', True, (255, 0, 0))
        if self.Episodes == 0:
            self.Episodes = 1
        Succeed_rate = font.render(f'E:{self.Episodes:4d} S:{self.Succeed_episode:4d} R:{self.Succeed_episode/self.Episodes*100:4.2f}%',
                                   True, (0, 255, 0))

        font_surface.blit(Succeed_rate, (0, 0))
        font_surface.blit(speed_surface, (0, 20))
        font_surface.blit(accel_surface, (0, 40))
        font_surface.blit(steel_surface, (0, 60))
        font_surface.blit(state_surface, (0, 80))
        font_surface.blit(old_state_surface, (0, 100))

        self.viewer.screen.blit(font_surface, (5, 5))

    def render(self, mode='human'):
        if self.config['offscreen_rendering']:
            return
        self.rendering_mode = mode
        if self.viewer is None:
            self.viewer = EnvViewer(self)
        self.enable_auto_render = True
        if not self.viewer.offscreen:
            self.viewer.display()   # 绘制主画布背景
            # 绘制全局路径
            if self.config['show_global_paths']:
                # 创建具有透明通道的新画布
                path_surface = pygame.Surface(self.viewer.sim_surface.get_size(), pygame.SRCALPHA)
                for i, p in enumerate(self.route_points):
                    # 坐标转换到像素坐标
                    screen_point = tuple(map(int, self.viewer.sim_surface.pos2pix(p[0], p[1])))
                    # 在新画布上绘制路径点
                    pygame.draw.circle(path_surface, (0, 0, 255, 150), screen_point, 3)  # 半透明蓝色点
                    if i > 0:
                        prev_p = self.route_points[i - 1]
                        prev_point = tuple(map(int, self.viewer.sim_surface.pos2pix(prev_p[0], prev_p[1])))
                        pygame.draw.line(path_surface, (0, 0, 255, 150), prev_point, screen_point, 1)  # 半透明蓝线
                screen_local_goal = self.viewer.sim_surface.pos2pix(self.local_goal_point[0], self.local_goal_point[1])
                pygame.draw.circle(path_surface, (255, 0, 0, 150), screen_local_goal, 4)    # 红点
                # path_surface = pygame.Surface.convert_alpha(path_surface)
                # 将路径表面叠加到主画布
                self.viewer.screen.blit(path_surface, (0, 0))
            # 绘制文字
            self.render_font()
            # 刷新显示
            pygame.display.flip()
            # 事件处理
            self.viewer.handle_events()

        if mode == 'rgb_array':
            # image = self.viewer.get_image()
            # return image
            # 捕获RGB通道
            rgb_data = pygame.surfarray.array3d(self.viewer.screen)
            # 捕获Alpha通道
            alpha_data = pygame.surfarray.array_alpha(self.viewer.screen)
            # 将RGB和Alpha通道合并为RGBA
            rgba_data = np.dstack((rgb_data, alpha_data))
            # 转换为H x W x C格式
            frame = np.moveaxis(rgba_data, 0, 1)
            # pygame.image.save(self.viewer.screen, 'temp.png')
            # from PIL import Image
            # Image.fromarray(frame).save('temp1.png')
            # from utils.Array2GIF import array2gif2
            # array2gif2([frame, frame, frame, frame, frame], 'temp3.gif', 10)
            return frame

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class RoundaboutEnv(BaseEnv):
    metadata = {'render.modes': ['human', 'rgb_array']}
    state_dim = 40 + 10
    action_dim = 2
    name = f'roundabout_v1_1_1x{state_dim}'

    def define_spaces(self) -> None:
        self.observation_type = observation_factory(self, self.config["observation"])   # 定义观测类
        self.action_type = action_factory(self, self.config["action"])                  # 定义动作类
        self.observation_space = spaces.Box(shape=(1, self.state_dim), low=-np.inf, high=np.inf, dtype=np.float32)  # 观测空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))        # 动作空间

    def define_route_index(self):
        if self.config['start_lane_index']:
            self.start_lane = self.config['start_lane_index']
        if self.config['goal_lane_index']:
            self.end_lane = self.config['goal_lane_index']

    def reset(self):
        self.define_route_index()   # 重置起点和终点索引
        self.off_road = False       # 是否驶出道路
        self.controlled_vehicles = []   # 控制车辆列表
        self.routes = []                # 路由列表
        self.route = []                 # 路由
        self.route_points = []          # 路由的点序列
        self.local_goal_point = None    # 局部目标点
        self.local_lane_index = None    # 局部车道索引
        self.local_point_info = None    # 投影点相关信息[x, y, longitudinal, lateral]
        self.old_terminate_flag = self.terminate_flag * 1
        self.terminate_flag = EndFlag.NOTEND  # 终止标志
        self.total_reward = 0
        self.define_spaces()
        self.times = 0
        self.steps = 0
        self._make_road()
        self._make_vehicles()
        self.shortest_route(self.start_lane[0], self.end_lane[1])   # 为主车重新规划路径
        self.route_points = self.get_path_points()                  # 将路径转化为点序列
        self.local_goal_point = self.get_local_goal(self.vehicle.position, self.route_points)

        return self._get_observation()

    def _make_road(self):
        # Circle lanes: (s)outh/(e)ast/(n)orth/(w)est (e)ntry/e(x)it.
        '''
        方向标记：
            s: 南 (south)
            e: 东 (east)
            n: 北 (north)
            w: 西 (west)
        连接类型：
            e: 入口 (entry)
            x: 出口 (exit)
        例如：
            "se": 从南侧入口进入环岛。
            "ex": 从东侧出口离开环岛。
        '''
        center = [0, 0]  # [m]
        radius = 20  # [m]
        alpha = 24  # [deg]

        net = RoadNetwork()
        radii = [radius, radius + 4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane("se", "ex",
                         CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ex", "ee",
                         CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ee", "nx",
                         CircularLane(center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("nx", "ne",
                         CircularLane(center, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("ne", "wx",
                         CircularLane(center, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("wx", "we",
                         CircularLane(center, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("we", "sx",
                         CircularLane(center, radii[lane], np.deg2rad(180 - alpha), np.deg2rad(90 + alpha),
                                      clockwise=False, line_types=line[lane]))
            net.add_lane("sx", "se",
                         CircularLane(center, radii[lane], np.deg2rad(90 + alpha), np.deg2rad(90 - alpha),
                                      clockwise=False, line_types=line[lane]))

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 85  # [m]
        a = 5  # [m]
        delta_st = 0.2 * dev  # [m]

        delta_en = dev - delta_st
        w = 2 * np.pi / dev
        net.add_lane("ser", "ses", StraightLane([2, access], [2, dev / 2], line_types=(s, c)))
        net.add_lane("ses", "se",
                     SineLane([2 + a, dev / 2], [2 + a, dev / 2 - delta_st], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("sx", "sxs",
                     SineLane([-2 - a, -dev / 2 + delta_en], [-2 - a, dev / 2], a, w, -np.pi / 2 + w * delta_en,
                              line_types=(c, c)))
        net.add_lane("sxs", "sxr", StraightLane([-2, dev / 2], [-2, access], line_types=(n, c)))

        net.add_lane("eer", "ees", StraightLane([access, -2], [dev / 2, -2], line_types=(s, c)))
        net.add_lane("ees", "ee",
                     SineLane([dev / 2, -2 - a], [dev / 2 - delta_st, -2 - a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("ex", "exs",
                     SineLane([-dev / 2 + delta_en, 2 + a], [dev / 2, 2 + a], a, w, -np.pi / 2 + w * delta_en,
                              line_types=(c, c)))
        net.add_lane("exs", "exr", StraightLane([dev / 2, 2], [access, 2], line_types=(n, c)))

        net.add_lane("ner", "nes", StraightLane([-2, -access], [-2, -dev / 2], line_types=(s, c)))
        net.add_lane("nes", "ne",
                     SineLane([-2 - a, -dev / 2], [-2 - a, -dev / 2 + delta_st], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("nx", "nxs",
                     SineLane([2 + a, dev / 2 - delta_en], [2 + a, -dev / 2], a, w, -np.pi / 2 + w * delta_en,
                              line_types=(c, c)))
        net.add_lane("nxs", "nxr", StraightLane([2, -dev / 2], [2, -access], line_types=(n, c)))

        net.add_lane("wer", "wes", StraightLane([-access, 2], [-dev / 2, 2], line_types=(s, c)))
        net.add_lane("wes", "we",
                     SineLane([-dev / 2, 2 + a], [-dev / 2 + delta_st, 2 + a], a, w, -np.pi / 2, line_types=(c, c)))
        net.add_lane("wx", "wxs",
                     SineLane([dev / 2 - delta_en, -2 - a], [-dev / 2, -2 - a], a, w, -np.pi / 2 + w * delta_en,
                              line_types=(c, c)))
        net.add_lane("wxs", "wxr", StraightLane([-dev / 2, -2], [-access, -2], line_types=(n, c)))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self):
        position_deviation = 2  # 位置扰动
        speed_deviation = 2     # 速度扰动
        _longitudinal = 5       # 纵向距离
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = utils.near_split(self.config["vehicles_count"],
                                                num_bins=self.config["controlled_vehicles"])
        for others in other_per_controlled:
            # 获取起点位置和方向
            _start_key = self.start_lane[0]
            _end_keys = list(self.road.network.graph[self.start_lane[0]])
            _end_key = random.choice(_end_keys)
            _id = 0
            _start_lane_index = (_start_key, _end_key, _id)
            ego_lane = self.road.network.get_lane(_start_lane_index)
            start_pos = ego_lane.position(125, 0)
            start_heading = ego_lane.heading_at(140)
            # 创建主车
            ego_vehicle = self.action_type.vehicle_class(self.road,
                                                         start_pos,
                                                         speed=self.config['ego_vehicle_init_speed'],
                                                         heading=start_heading)
            self.controlled_vehicles.append(ego_vehicle)
            self.road.vehicles.append(ego_vehicle)
            # 添加其他车辆
            for _ in range(others):
                lane_from = random.choice([i for i in self.road.network.graph.keys()])
                lane_to = random.choice([i for i in self.road.network.graph[lane_from].keys()])
                lane_id = random.randint(0, len(self.road.network.graph[lane_from][lane_to]) - 1)
                lane_index = (lane_from, lane_to, lane_id)
                if lane_index == _start_lane_index:
                    _longitudinal = 10
                vehicle = other_vehicles_type.make_on_lane(self.road,
                                                           lane_index,
                                                           longitudinal=_longitudinal + self.np_random.randn() * position_deviation,
                                                           speed=self.config['other_vehicles_init_speed'] + self.np_random.randn() * speed_deviation)
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def step(self, _action):
        if self.steps == 0:
            self.Episodes += 1  # 回合次数+1
        self.steps += 1
        self._simulate(_action)
        self.local_goal_point = self.get_local_goal(self.vehicle.position, self.route_points, lookahead_distance=int(self.vehicle.speed / self.config['policy_frequency']))
        _done = self._is_terminal()
        _observation = self._get_observation()
        _reward = self._get_reward()
        self.total_reward += _reward
        # 获取车辆信息
        _info = {
            'speed': self.vehicle.speed,
            'heading': self.vehicle.heading,
            'lane_index': self.vehicle.lane_index,
            'position': self.vehicle.position
        }
        if self.terminate_flag == EndFlag.REACHEND:
            self.Succeed_episode += 1
        if self.config['save_experience']:
            self.save_experience(_observation, _reward, _done, _action)
        return _observation, _reward, _done, _info

    def _simulate(self, _action=None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        dt = 1 / self.config["simulation_frequency"]
        for frame in range(frames):
            # Forward action to the vehicle
            if _action is not None \
                    and not self.config["manual_control"] \
                    and self.times % frames == 0:
                self.action_type.act(_action)
            self.road.act()
            self.road.step(dt)
            self.times += 1
            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()
        self.enable_auto_render = False

    def is_lane_change_allowed(self, lane_index):
        side_lanes = self.road.network.side_lanes(lane_index)
        left_side_lane_index = (lane_index[0], lane_index[1], lane_index[2] - 1)
        right_side_lane_index = (lane_index[0], lane_index[1], lane_index[2] + 1)
        _l, _r = False, False
        if left_side_lane_index in side_lanes:
            _l = True
        if right_side_lane_index in side_lanes:
            _r = True
        return _l, _r

    def _get_observation(self):
        df_columns = self.observation_type.features + ['width', 'length']
        if not self.road:
            return np.zeros(self.observation_space.shape).flatten()
        # Add ego-vehicle
        ego_data = self.observation_type.observer_vehicle.to_dict()
        ego_data.update({'width':self.vehicle.WIDTH, 'length':self.vehicle.LENGTH})
        df = pd.DataFrame.from_records([ego_data])[df_columns]
        # Add nearby traffic
        close_vehicles = self.road.close_vehicles_to(self.observation_type.observer_vehicle,
                                                     self.PERCEPTION_DISTANCE,
                                                     count=self.config['observation']['vehicles_count'] - 1,
                                                     see_behind=self.observation_type.see_behind,
                                                     sort=self.observation_type.order == "sorted")
        if close_vehicles:
            origin = self.observation_type.observer_vehicle if not self.observation_type.absolute else None
            _data_list = []
            for v in close_vehicles[-self.config['observation']['vehicles_count'] + 1:]:
                d = v.to_dict(origin, observe_intentions=self.observation_type.observe_intentions)
                d.update({'width': v.WIDTH, 'length': v.LENGTH})
                _data_list.append(d)
            df = df.append(pd.DataFrame.from_records(_data_list)[df_columns], ignore_index=True)
        # Normalize and clip
        if self.observation_type.normalize:
            df = self.observation_type.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.config['observation']['vehicles_count']:
            rows = np.zeros(
                (self.config['observation']['vehicles_count'] - df.shape[0],
                 len(df_columns)))
            df = df.append(pd.DataFrame(data=rows, columns=df_columns), ignore_index=True)
        # Reorder
        df = df[df_columns]
        obs = df.values.copy()
        if self.observation_type.order == "shuffled":
            self.np_random.shuffle(obs[1:])
        # Add road-related observations
        lane = self.road.network.get_lane(self.local_lane_index)    # 获取当前局部目标车道对象
        _s, _r = lane.local_coordinates(self.vehicle.position)      # 车辆在目标车道上的位置
        _is_lane_change_allowed = self.is_lane_change_allowed(self.vehicle.lane_index)
        delta_position = self.local_goal_point - self.vehicle.position
        # 新增观测信息
        road_features = [
            _r,     # 距离车道中心线距离
            int(_is_lane_change_allowed[0]),    # 能否向左变道
            int(_is_lane_change_allowed[1]),    # 能否向右变道
            np.linalg.norm(delta_position),                         # 车辆与局部目标点的距离
            np.arctan2(delta_position[1], delta_position[0]),       # 车辆与局部目标点的航向角
            ((self.vehicle.heading + self.vehicle.action['steering']) / 2 - lane.heading_at(_s) + np.pi) \
            % (2 * np.pi) - np.pi,              # 车辆航向角和车道线切线角偏差
            self.config["max_speed"],           # 最大速度
            self.config["min_speed"],           # 最小速度
            self.vehicle.action['acceleration'],    # 上时刻的加速度
            self.vehicle.action['steering'],    # 上时刻的转角
        ]
        obs = np.append(obs, road_features)     # 未指定维度，展开再拼接。
        # reshape->(1,len)
        obs = obs.reshape(1, -1)
        return obs.astype(self.observation_type.space().dtype)

    def _get_reward(self):
        # 原始奖励项
        reward = 0
        # reach_goal_reward
        if (self.local_goal_point == self.route_points[-1]).all():
            reward += self.config["reach_goal_reward"]
        # Speed reward
        if self.vehicle.speed > self.config["min_speed"]:
            speed_reward = self.config["high_speed_reward"] * (self.vehicle.speed / self.config["max_speed"])
            if speed_reward > self.config["high_speed_reward"]:
                speed_reward = self.config["high_speed_reward"]
            reward += speed_reward
            if self.vehicle.speed > self.config["max_speed"]:
                reward += self.config["overspeed_penalty"]
        # Collision penalty
        collision_penalty = self.config["collision_reward"] if self.vehicle.crashed else 0
        reward += collision_penalty
        # Off Road penalty
        off_road_penalty = self.config['off_road_reward'] if self.off_road else 0
        reward += off_road_penalty
        # Lane center reward
        lane = self.road.network.get_lane(self.local_lane_index)    # 局部目标点所在车道
        _s, _r = lane.local_coordinates(self.vehicle.position)
        # _s = self.local_point_info[2]
        # _r = self.local_point_info[3]
        lateral_offset = abs(_r)
        lane_center_reward = self.config["lane_center_reward"] * (1 if lateral_offset < (lane.width_at(_s) / 3) else 0)
        reward += lane_center_reward
        # Heading reward    (heading顺时针为正，逆时针为负；steer顺时针为正，逆时针为负)
        heading_error = ((self.vehicle.heading + self.vehicle.action['steering'])/2 - lane.heading_at(_s) + np.pi) % (2 * np.pi) - np.pi
        heading_reward = self.config["heading_reward"] * (abs(heading_error) / np.pi)
        reward += heading_reward
        # Jerk penalty
        a_jerk = np.abs((self.vehicle.action['acceleration'] - getattr(self.vehicle, 'last_acceleration', 0)) / (1 / self.config["policy_frequency"]))
        s_jerk= np.abs((self.vehicle.action['steering'] - getattr(self.vehicle, 'last_steering', 0)) / (1 / self.config["policy_frequency"]))
        comfort_a = 3
        comfort_s = 0.7
        if a_jerk > comfort_a:
            a_jerk = comfort_a
        if s_jerk > comfort_s:
            s_jerk = comfort_s
        jerk = a_jerk/comfort_a * 0.2 + s_jerk/comfort_s * 0.8
        jerk_penalty = self.config["jerk_penalty"] * jerk
        reward += jerk_penalty
        self.vehicle.last_acceleration = self.vehicle.action['acceleration']
        self.vehicle.last_steering = self.vehicle.action['steering']
        # Lane change penalty
        if self.vehicle.lane_index != getattr(self.vehicle, 'last_lane_index', self.vehicle.lane_index):
            if self.steps > 1:  # 避免第一步处罚
                lane_change_penalty = self.config["lane_change_penalty"]
                reward += lane_change_penalty
        self.vehicle.last_lane_index = self.vehicle.lane_index
        # Negative speed reward
        negative_speed_reward = self.config["negative_speed"] if self.vehicle.speed < 0 else 0
        reward += negative_speed_reward
        low_reward = self.config["collision_reward"] + \
            self.config["jerk_penalty"] +  \
            self.config["heading_reward"] + \
            self.config["off_road_reward"] + \
            self.config["negative_speed"] + \
            self.config["overspeed_penalty"] + \
            self.config["lane_change_penalty"]
        high_reward = self.config["high_speed_reward"] + self.config["lane_center_reward"] + self.config["reach_goal_reward"]
        reward = utils.lmap(reward, [low_reward, high_reward], [0, 1])
        reward = 0 if reward < 0 else reward
        return reward

    def _is_terminal(self):
        done = False
        if (self.local_goal_point == self.route_points[-1]).all():
            distance_ = np.linalg.norm(self.local_goal_point - self.vehicle.position)
            if distance_ < self.vehicle.WIDTH:
                print('抵达目标点')
                self.terminate_flag = EndFlag.REACHEND  # 终止标志
                done = True
        if self._is_vehicle_off_road():
            print('驶出道路')
            self.terminate_flag = EndFlag.OFFROAD
            done = True
        if self.vehicle.crashed:
            print('发生碰撞')
            self.terminate_flag = EndFlag.CRASH
            done = True
        if self.steps >= self.config["duration"] * self.config['policy_frequency']:
            print('时间结束')
            self.terminate_flag = EndFlag.TIMEOVER
            done = True
        if self.vehicle.speed <= self.config['min_speed']-0.01:
            print("低于最低速度")
            self.terminate_flag = EndFlag.REVERSE
            done = True
        if self.vehicle.speed > self.config['max_speed']+0.01:
            print("超速")
            self.terminate_flag = EndFlag.OVERSPEED
            done = True
        return done

    def _is_vehicle_off_road(self) -> bool:
        self.off_road = True
        if not self.vehicle or not self.road:
            return True
        position = self.vehicle.position
        lane_index = self.road.network.get_closest_lane_index(position)
        if lane_index is None:
            return True
        lane = self.road.network.get_lane(lane_index)
        if not lane.on_lane(position):
            return True
        self.off_road = False
        return False

    def shortest_route(self,start:str, goal:str):
        '''
        从规划的路径中选择最短路径。
        :param start: 起点编号
        :param goal: 目标编号
        :return:
        '''
        self.global_route_plan(start,goal,[start],0)
        if self.routes:
            ...
            shortest_route = min(self.routes,key=lambda a:a[1])
            _path = shortest_route[0]
            self.route = [self.vehicle.lane_index] + [(_path[i], _path[i + 1], 0) for i in range(len(_path) - 1)]
        else:
            self.route = [self.vehicle.lane_index]

    def global_route_plan(self,start:str, goal:str, path:list, length:float):
        '''
        全局路径规划。
        :param start: 起点编号
        :param goal: 目标编号
        :param path: 当前路线
        :param length: 当前路线长度
        :return:
        '''
        if start in self.road.network.graph.keys():
            end_list = [i for i in self.road.network.graph[start].keys()]
            for e in end_list:
                if e in path:
                    continue
                _len = self.road.network.get_lane((start,e,0)).length
                if e == goal:
                    self.routes.append([path + [e], length + _len])
                    # return
                self.global_route_plan(e, goal, path + [e],length + _len)
        else:
            ...

    def get_path_points(self):
        """
        将路径编号列表转换为中心线的点序列。
        :return: 全局路径的点列表
        """
        path_points = []
        point = None
        for edge in self.route:
            # 获取当前路径段上的车道
            lane = self.road.network.get_lane(edge)
            # 在路径段上采样多个点，用于描述中心线
            for i in range(0, int(lane.length), 2):  # 每2米采样一个点
                new_point = lane.position(i, 0)  # 获取道路中心线点
                if i == 0 and point is not None:
                    distance_ = np.linalg.norm(np.array(new_point) - np.array(point))
                    if distance_ > 2:   # 增加中间点
                        path_points.append([(new_point[0]+point[0])/2, (new_point[1]+point[1])/2])
                point = new_point
                path_points.append(point)
        return path_points

    def get_position_on_path(self, position, path_points):
        """
        计算车辆在路径上的纵向和横向位置。
        :param position: 车辆当前位置 (x, y)
        :param path_points: 全局路径上的点序列 [(x1, y1), (x2, y2), ...]
        :return: (longitudinal position, lateral position)
        """
        min_dist = float('inf')
        proj_index = 0
        proj_point = None
        # 找到距离车辆最近的路径点
        for i, point in enumerate(path_points):
            dist = np.linalg.norm(np.array(position) - np.array(point))
            if dist < min_dist:
                min_dist = dist
                proj_index = i
                proj_point = point
        # 计算纵向位置
        longitudinal = sum(
            np.linalg.norm(np.array(path_points[j]) - np.array(path_points[j + 1]))
            for j in range(proj_index)
        )
        # 计算横向位置
        direction = np.array(proj_point) - np.array(path_points[proj_index - 1])
        relative_vec = np.array(position) - np.array(proj_point)
        lateral = np.cross(direction / np.linalg.norm(direction), relative_vec)
        self.local_lane_index = self.road.network.get_closest_lane_index(proj_point)
        self.local_point_info = [proj_point[0], proj_point[1], longitudinal, lateral]
        return longitudinal, lateral

    def get_local_goal(self, position, path_points, lookahead_distance=5):
        """
        获取局部目标点，引导车辆前进。
        :param position: 当前车辆位置 (x, y)
        :param path_points: 全局路径上的点序列
        :param lookahead_distance: 向前查找的距离
        :return: 局部目标点坐标 (x, y)
        """
        if lookahead_distance < 4:
            lookahead_distance = 4
        longitudinal, _ = self.get_position_on_path(position, path_points)
        distance = 0
        for i in range(len(path_points) - 1):
            segment_length = np.linalg.norm(np.array(path_points[i + 1]) - np.array(path_points[i]))
            distance += segment_length
            if distance > longitudinal + lookahead_distance:
                return path_points[i + 1]
        # 如果路径不足以提供前方【lookahead_distance】米距离，则返回最后一个点
        return path_points[-1]

    def save_experience(self, _obs, _reward, _done, _action):
        ...
        if self.terminate_flag == EndFlag.NOTEND:
            _data = np.append(_obs, [_reward, _done, _action[0], _action[1]])
            self.csv_data.append(_data)
        else:
            if self.steps >= self.config['policy_frequency']:
                _now = datetime.datetime.now()
                file_path = f'{self.name}_{self.terminate_flag}_{self.total_reward:.4f}_{_now.strftime("%Y_%m_%d_%H_%M_%S_%f")}.csv'
                _csv_title = self.csv_head_title + ['reward', 'done', 'acceleration', 'steering']
                df = pd.DataFrame(self.csv_data, columns=_csv_title)
                df.to_csv(os.path.join(self.config['save_data_path'], file_path), index=False)




if __name__ == '__main__':
    env = RoundaboutEnv()
    env.configure(
        {
            "simulation_frequency": 30,
            "policy_frequency": 5,
            "screen_width": 800,
            "screen_height": 800,
            "centering_position": [0.3, 0.5],
            "scaling": 10,
            "show_trajectories": False,
            "render_agent": True,
            "manual_control": False,
            "offscreen_rendering": False,
            "show_global_paths": True,
            "real_time_rendering": True,
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
        }
    )
    env.reset()
    # print(env.road.network.graph)
    '''
    self.start_lane = ("ser", "ses", 0)  # 起点车道
    self.end_lane = ("exr", "exs", 0)  # 终点车道
    '''
    env.shortest_route("ser", "ne")
    print(env.routes)
    t1 = time.time()
    while True:
        env.render(mode='rgb_array')
        action = env.action_space.sample()
        print(action,env.vehicle.action)
        obs, reward, done, info = env.step(action)
        # print(action, reward, done, obs.shape, obs)
        if done:
            print(env.steps,env.times)
            break
    t2 = time.time()
    print(f'times: {t2-t1} s')
    env.close()
