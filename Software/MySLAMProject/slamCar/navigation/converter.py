from __future__ import annotations
from typing import List, Tuple


class Converter:

    def __init__(self, map_size_pixels: int, map_size_meters: float):

        self.map_size_pixels = map_size_pixels
        self.map_size_meters = map_size_meters
        self.meters_per_pixel = map_size_meters / map_size_pixels

        # SLAM地图中的起点像素坐标
        self.origin_pix_x = map_size_pixels / 2.0
        self.origin_pix_y = map_size_pixels / 2.0

    def world_to_pixel(self, x_m: float, y_m: float) -> Tuple[int, int]:

        # 世界坐标以起点为原点，需要加上地图中心偏移
        x_pix = self.origin_pix_x + (x_m / self.meters_per_pixel)
        y_pix = self.origin_pix_y + (y_m / self.meters_per_pixel)
        return (x_pix, y_pix)

    def pixel_to_world(self, x_pix: float, y_pix: float) -> Tuple[float, float]:

        x_m = (x_pix - self.origin_pix_x) * self.meters_per_pixel
        y_m = (y_pix - self.origin_pix_y) * self.meters_per_pixel
        return (x_m, y_m)

    def is_within_map(self, x_pix: float, y_pix: float) -> bool:
        """检查像素坐标是否在地图范围内"""
        return (0 <= x_pix < self.map_size_pixels and
                0 <= y_pix < self.map_size_pixels)
