from typing import List, Tuple, Set
import numpy as np
from slamCar.Mymap.map_display import Drawer
from scipy import ndimage
import time
FREE_THRESH = 230
OBSTACLE_THRESH = 60


class ListGridMap:

    def __init__(self, MAP_SIZE_PIXELS, MAP_SIZE_METERS):
        self.w = MAP_SIZE_PIXELS
        self.h = MAP_SIZE_PIXELS
        self.map_size_pixels = MAP_SIZE_PIXELS

        # 主显示地图 grid[y][x]
        self.grid: List[List[int]] = [[128 for _ in range(self.w)] for _ in range(self.h)]

        self.obstacle_votes: np.ndarray = np.zeros((self.h, self.w), dtype=np.int16)

        self.free_votes: np.ndarray = np.zeros((self.h, self.w), dtype=np.int16)

        self.locked_map: np.ndarray = np.full((self.h, self.w), 128, dtype=np.uint8)  # 128=未锁定

        self.is_locked: np.ndarray = np.zeros((self.h, self.w), dtype=bool)

        self.vote_params = {
            'obstacle_lock_threshold': 4,  # 障碍物锁定需要的投票数
            'free_lock_threshold': 10,  # 自由空间锁定需要的投票数
            'conflict_resolve_ratio': 1,  # 冲突解决比例：一方票数需要是另一方的几倍才能获胜
            'noise_filter_threshold': 2,  # 噪声过滤：低于此票数的不参与决策
        }

        self.map_scale_meters_per_pixel = MAP_SIZE_METERS / float(MAP_SIZE_PIXELS)
        self.CurrCarPose = None

        self.drawer = Drawer(MAP_SIZE_PIXELS, MAP_SIZE_METERS, title="Mapping",
                             show_trajectory=True, origin_lower_left=True)

        self.enable_map_regularization = True
        self.regularization_params = {
            'min_free_cluster_size': 1000,
            'min_obstacle_thickness': 2,
            'wall_straightening_kernel': 3,
            'unknown_gap_threshold': 400,
            'wall_dilation_size': 3,
        }

    def UpdateMap(self, mapbytes, origin_lower_left=True):

        slam_map = np.reshape(np.frombuffer(mapbytes, dtype=np.uint8),
                              (self.map_size_pixels, self.map_size_pixels))

        self._update_votes(slam_map)

        self._update_locked_map()

        final_map = self._generate_final_map(slam_map)

        self.grid = final_map.tolist()

    def _update_votes(self, slam_map: np.ndarray):

        h, w = slam_map.shape

        obstacle_mask = (slam_map <= OBSTACLE_THRESH)
        free_mask = (slam_map >= FREE_THRESH)

        votable_mask = ~self.is_locked

        obstacle_vote_mask = obstacle_mask & votable_mask
        self.obstacle_votes[obstacle_vote_mask] += 1

        free_vote_mask = free_mask & votable_mask
        self.free_votes[free_vote_mask] += 1

    def _update_locked_map(self):

        obs_threshold = self.vote_params['obstacle_lock_threshold']
        free_threshold = self.vote_params['free_lock_threshold']
        conflict_ratio = self.vote_params['conflict_resolve_ratio']
        noise_threshold = self.vote_params['noise_filter_threshold']

        h, w = self.obstacle_votes.shape

        for y in range(h):
            for x in range(w):
                # 跳过已锁定的像素
                if self.is_locked[y, x]:
                    continue

                obs_votes = self.obstacle_votes[y, x]
                free_votes = self.free_votes[y, x]

                # 噪声过滤：投票数太少的不处理
                total_votes = obs_votes + free_votes
                if total_votes < noise_threshold:
                    continue

                # 决策逻辑
                lock_value = None

                if (obs_votes >= obs_threshold and
                        obs_votes >= free_votes * conflict_ratio):
                    lock_value = 0

                elif (free_votes >= free_threshold and
                      free_votes >= obs_votes * conflict_ratio):
                    lock_value = 255  # 锁定为自由空间

                elif (obs_votes >= obs_threshold and free_votes >= free_threshold):
                    if obs_votes > free_votes * 1.2:
                        lock_value = 0
                    elif free_votes > obs_votes * 1.2:
                        lock_value = 255

                if lock_value is not None:
                    self.locked_map[y, x] = lock_value
                    self.is_locked[y, x] = True

    def _generate_final_map(self, slam_map: np.ndarray) -> np.ndarray:

        final_map = slam_map.copy()

        locked_mask = self.is_locked
        final_map[locked_mask] = self.locked_map[locked_mask]

        return final_map

    def get_vote_stats(self) -> dict:

        total_pixels = self.w * self.h
        locked_pixels = np.sum(self.is_locked)
        locked_obstacles = np.sum(self.is_locked & (self.locked_map <= OBSTACLE_THRESH))
        locked_free = np.sum(self.is_locked & (self.locked_map >= FREE_THRESH))

        return {
            'total_pixels': total_pixels,
            'locked_pixels': int(locked_pixels),
            'locked_obstacles': int(locked_obstacles),
            'locked_free': int(locked_free),
            'lock_ratio': float(locked_pixels / total_pixels),
            'max_obstacle_votes': int(np.max(self.obstacle_votes)),
            'max_free_votes': int(np.max(self.free_votes)),
            'avg_obstacle_votes': float(np.mean(self.obstacle_votes[self.obstacle_votes > 0])) if np.any(
                self.obstacle_votes) else 0,
            'avg_free_votes': float(np.mean(self.free_votes[self.free_votes > 0])) if np.any(self.free_votes) else 0,
        }

    def m2pix(self, x_m, y_m):
        s = self.map_scale_meters_per_pixel
        return x_m / s, y_m / s

    def pix2m(self, x_pix, y_pix):
        s = self.map_scale_meters_per_pixel
        return x_pix * s, y_pix * s

    def SetCarPose(self, x_m, y_m, theta_deg):
        x_pix, y_pix = self.m2pix(x_m, y_m)
        self.CurrCarPose = (x_pix, y_pix, theta_deg)

    def GetCarPose(self):
        if self.CurrCarPose is None:
            raise RuntimeError("车辆位姿尚未设置")
        return self.CurrCarPose[0], self.CurrCarPose[1], self.CurrCarPose[2]

    def draw(self):
        # 在HUD中添加投票统计
        stats = self.get_vote_stats()
        extra_text = (f"Vote Stats:\n"
                      f"Locked: {stats['locked_pixels']} ({stats['lock_ratio']:.1%})\n"
                      f"Locked Obs: {stats['locked_obstacles']}\n"
                      f"Locked Free: {stats['locked_free']}")

        self.drawer.display(self.grid, self.CurrCarPose, extra_text=extra_text)
        print("drawing the map...")