# planning.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional, Iterable, Dict, Any,Set
import math
import time
from slamCar.navigation.config import *

class FrontierExplorer:

    def __init__(self,
                 free_thresh: int = FREE_thresh,
                 unknown_val: int = 128,
                 obstacle_val: int = OBSTACLE_val,
                 obs_clearance_pix: int = 10,
                 gain_radius_pix: int = 8,  # 信息增益计算半径
                 connectivity_check_radius: int = 200,  # 连通性检查半径
                 min_distance_pix: int = 30,  # 最小选择距离，避免打转
                 max_distance_pix: int = 200,  # 最大考虑距离，避免不连通
                 spatial_sample_grid: int = 10,  # 空间采样网格大小
                 max_candidates_to_eval: int = 50,
                 min_wall_distance_pix: int = 50,
                 goal_bias_weight: float = 0.6):  # 新增参数：最小墙壁距离):  # 最多评估的候选点数

        self.free_th = int(free_thresh)
        self.unknown_v = int(unknown_val)
        self.obst_v = int(obstacle_val)
        self.clear_r = int(obs_clearance_pix)
        self.gain_r = int(gain_radius_pix)
        self.conn_r = int(connectivity_check_radius)
        self.min_dist = int(min_distance_pix)
        self.max_dist = int(max_distance_pix)
        self.sample_grid = int(spatial_sample_grid)
        self.max_eval = int(max_candidates_to_eval)
        self.SPATIALLY = None
        self.min_wall_dist = int(min_wall_distance_pix)  # 新增：最小墙壁距离
        self.goal_bias_weight = goal_bias_weight  # 新增

    def detect_frontiers_optimized(self, grid: List[List[int]],
                                   maze_rect: Optional[Tuple[int, int, int, int]],
                                   changed_regions: Optional[List[Tuple[int, int, int, int]]] = None) -> List[
        Tuple[int, int]]:
        """
        增量式前沿检测：只检查地图变化区域
        changed_regions: [(x0, y0, x1, y1), ...] 最近更新的地图区域
        """
        if changed_regions is None:
            # 首次全图检测，使用向量化操作加速
            return self._detect_frontiers_vectorized(grid, maze_rect)

        # 增量检测：只处理变化区域周围
        frontiers = []
        for region in changed_regions:
            # 扩展搜索范围（包含clearance半径）
            expanded = self._expand_region(region, self.clear_r + 5)
            frontiers.extend(self._detect_in_region(grid, expanded, maze_rect))

        return self._remove_duplicates(frontiers)

    def _detect_frontiers_vectorized(self, grid, maze_rect):
        """使用numpy向量化操作加速前沿检测"""
        import numpy as np

        grid_np = np.array(grid)
        H, W = grid_np.shape

        # 一次性计算所有自由空间
        free_mask = grid_np >= self.free_th

        # 使用卷积快速检测边界
        from scipy.ndimage import convolve
        kernel = np.ones((3, 3), dtype=int)
        kernel[1, 1] = 0

        # 检测邻域中的未知区域
        unknown_mask = (grid_np > self.obst_v) & (grid_np < self.free_th)
        has_unknown_neighbor = convolve(unknown_mask.astype(int), kernel, mode='constant') > 0

        # 检测障碍物距离（可以预计算并缓存）
        safe_mask = self._compute_safe_mask(grid_np)

        # 组合所有条件
        frontier_mask = free_mask & has_unknown_neighbor & safe_mask

        # 提取坐标
        y_coords, x_coords = np.where(frontier_mask)
        return list(zip(x_coords, y_coords))

    def _precompute_robust_free(self, grid, maze_rect):
        """预计算整个地图的厚实自由空间（可缓存）"""
        import numpy as np
        from scipy.ndimage import uniform_filter

        grid_np = np.array(grid, dtype=float)

        # 使用卷积快速计算局部自由空间比例
        free_mask = (grid_np >= self.free_th).astype(float)
        kernel_size = 11  # 5*2+1
        local_free_ratio = uniform_filter(free_mask, size=kernel_size, mode='constant')

        return local_free_ratio > 0.5  # 返回布尔掩码

    def _fast_connectivity_check(self, candidates, curr_pos, robust_map, maze_rect):
        """使用预计算结果的快速连通性检查"""
        from collections import deque

        W, H = robust_map.shape[1], robust_map.shape[0]
        x0, y0, x1, y1 = maze_rect

        start_x, start_y = int(curr_pos[0]), int(curr_pos[1])

        # 快速验证起点
        if not robust_map[start_y, start_x]:
            start_x, start_y = self._find_nearest_free(start_x, start_y, robust_map, maze_rect)
            if start_x is None:
                return candidates[:2]

        # 使用距离限制的BFS（避免扩展整个地图）
        max_search_dist = self.max_dist * 1.5
        visited = {(start_x, start_y)}
        queue = deque([(start_x, start_y, 0)])  # (x, y, distance)

        reachable_candidates = set()
        candidates_set = set(candidates)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]

        while queue and len(reachable_candidates) < len(candidates):
            x, y, dist = queue.popleft()

            # 检查是否是候选点
            if (x, y) in candidates_set:
                reachable_candidates.add((x, y))

            # 距离限制早停
            if dist >= max_search_dist:
                continue

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if (nx, ny) in visited:
                    continue

                if not (x0 <= nx < x1 and y0 <= ny < y1):
                    continue

                if not robust_map[ny, nx]:
                    continue

                visited.add((nx, ny))
                queue.append((nx, ny, dist + 1))

        result = [c for c in candidates if c in reachable_candidates]
        return result if result else candidates[:2]

    #——————————————————————————————————————————————————————————————————————————————————————
    #公共调用
    #———————————————————————————————————————————————————————————————————————————————————————


    def pick_goal(self,
                  curr_pix: Tuple[float, float],
                  candidates: List[Tuple[int, int]],
                  grid: List[List[int]],
                  goal_pix: Optional[Tuple[int, int]] = None,
                  exclude_radius_pix: float = 3,
                  maze_rect: Optional[Tuple[int, int, int, int]] = None) -> List[Tuple[int, int]]:

        if not candidates:
            return []

        start_time = time.time()
        cx, cy = int(curr_pix[0]), int(curr_pix[1])


        distance_filtered = self._filter_by_distance(candidates, (cx, cy), exclude_radius_pix)
        if not distance_filtered:
            return []

        spatially_sampled = self._spatial_sampling(distance_filtered)

        connected_candidates = self._filter_by_connectivity_simple(spatially_sampled, (cx, cy), grid,maze_rect)
        if not connected_candidates:

            return []

        if goal_pix is not None:
            goal_biased_candidates = self._filter_by_goal_proximity(
                connected_candidates, (cx, cy), goal_pix
            )
        else:
            goal_biased_candidates = connected_candidates

        self.SPATIALLY = connected_candidates


        elapsed = time.time() - start_time
        print(
            f"前沿选择耗时: {elapsed:.3f}s, 候选点: {len(candidates)} -> {len(distance_filtered)} -> {len(spatially_sampled)} -> {len(connected_candidates)}->{len(goal_biased_candidates)}")

        return goal_biased_candidates

    def _filter_by_goal_proximity(self,
                                  candidates: List[Tuple[int, int]],
                                  curr_pix: Tuple[int, int],
                                  goal_pix: Tuple[int, int]) -> List[Tuple[int, int]]:

        if not candidates:
            return []

        cx, cy = curr_pix
        gx, gy = goal_pix

        # 当前位置到终点的向量
        goal_vec = (gx - cx, gy - cy)
        goal_dist = math.sqrt(goal_vec[0] ** 2 + goal_vec[1] ** 2)

        if goal_dist < 1e-6:
            # 当前位置已接近终点，直接返回原列表
            return candidates

        # 归一化方向向量
        goal_dir = (goal_vec[0] / goal_dist, goal_vec[1] / goal_dist)

        scored_candidates = []

        for px, py in candidates:
            # 候选点到终点的距离
            dist_to_goal = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)

            # 当前位置→候选点的向量
            cand_vec = (px - cx, py - cy)
            cand_dist = math.sqrt(cand_vec[0] ** 2 + cand_vec[1] ** 2)

            if cand_dist < 1e-6:
                continue

            # 方向对齐度（点积，范围[-1, 1]，1表示完全朝向终点）
            alignment = (cand_vec[0] * goal_dir[0] + cand_vec[1] * goal_dir[1]) / cand_dist

            # 综合得分：
            # - 距离终点越近越好（归一化到[0,1]）
            # - 方向越对齐越好（归一化到[0,1]）
            # - 使用可配置的权重平衡
            distance_score = 1.0 / (1.0 + dist_to_goal / 100.0)  # 距离得分
            alignment_score = (alignment + 1.0) / 2.0  # 对齐得分：[-1,1] → [0,1]

            # 综合得分（权重可调）
            final_score = (self.goal_bias_weight * distance_score +
                           (1 - self.goal_bias_weight) * alignment_score)

            scored_candidates.append(((px, py), final_score))

        # 按得分降序排序
        scored_candidates.sort(key=lambda x: -x[1])

        # 返回排序后的坐标列表
        result = [coord for coord, _ in scored_candidates]
        
        # MODIFICATION: 检查是否有候选点，避免在空列表上索引 [0]
        if scored_candidates:
            print(f"终点导向筛选: {len(candidates)}个候选点, "
                  f"最佳得分={scored_candidates[0][1]:.3f}")
        else:
             print(f"终点导向筛选: {len(candidates)}个候选点, 没有有效得分")

        return result


    def _filter_by_distance(self, candidates: List[Tuple[int, int]],
                            curr_pos: Tuple[int, int],
                            exclude_radius: float) -> List[Tuple[int, int]]:
        """距离筛选：排除太近和太远的点"""
        cx, cy = curr_pos
        exclude_r2 = exclude_radius * exclude_radius
        min_r2 = self.min_dist * self.min_dist
        max_r2 = self.max_dist * self.max_dist

        filtered = []
        for px, py in candidates:
            dist2 = (px - cx) ** 2 + (py - cy) ** 2
            if exclude_r2 < dist2  and dist2 >= min_r2:
                filtered.append((px, py))

        return filtered

    def _spatial_sampling(self, candidates: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """空间采样：在网格中每个格子只保留一个候选点"""
        if len(candidates) <= self.max_eval:
            return candidates

        # 将空间分割为网格，每个网格保留一个点
        grid_dict = {}
        for px, py in candidates:
            grid_x = px // self.sample_grid
            grid_y = py // self.sample_grid
            grid_key = (grid_x, grid_y)

            if grid_key not in grid_dict:
                grid_dict[grid_key] = []
            grid_dict[grid_key].append((px, py))

        # 每个网格选择一个代表点（可以是随机选择或者选择网格中心附近的）
        sampled = []
        for grid_points in grid_dict.values():
            # 选择网格内的第一个点作为代表（也可以用其他策略）
            sampled.append(grid_points[0])


        return sampled

    def _filter_by_connectivity_simple(self, candidates, curr_pos, grid,maze_rect: Optional[Tuple[int, int, int, int]] = None):

        from collections import deque

        W = len(grid[0])
        H = len(grid)

        x0 = maze_rect[0]
        y0 = maze_rect[1]
        x1 = maze_rect[2]
        y1 = maze_rect[3]

        def is_robust_free(x, y):

            free_count = 0
            total = 0
            
            # ==========================================================
            # MODIFICATION: 检查范围从 11x11 (range(-5, 6)) 缩小到 5x5 (range(-2, 3))
            # 这样可以允许机器人在较窄的走廊中找到“厚实”起点
            # ==========================================================
            for dy in range(-2, 3):
                for dx in range(-2, 3):
            # ==========================================================
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and x0<=nx<x1 and y0<=ny<y1:
                        total += 1
                        if grid[ny][nx] >= self.free_th:
                            free_count += 1
            return total > 0 and free_count / total > 0.5 # 保持 50% 的阈值

        # 1. 找到有效的起始点
        start_x, start_y = int(curr_pos[0]), int(curr_pos[1])

        # 如果当前位置不在厚实区域，在附近搜索
        if not (0 <= start_x < W and 0 <= start_y < H) or not is_robust_free(start_x, start_y):
            print("当前位置不在厚实区域，在附近搜索")
            for r in range(1, 8):
                found = False
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if dx * dx + dy * dy > r * r:
                            continue
                        nx, ny = start_x + dx, start_y + dy
                        if 0 <= nx < W and 0 <= ny < H and x0<=nx<x1 and y0<=ny<y1 and is_robust_free(nx, ny):
                            start_x, start_y = nx, ny
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            else:
                # 找不到厚实起点，降级处理
                print("无法找到厚实起点，降级处理")
                return self._fallback_connectivity_check(candidates, curr_pos, grid)

        # 2. BFS扩展厚实连通区域
        robust_region = set()
        queue = deque([(start_x, start_y)])
        robust_region.add((start_x, start_y))

        # 使用8连通扩展，提高连通性
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]

        while queue:
            x, y = queue.popleft()

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if (nx, ny) in robust_region:
                    continue

                if not (0 <= nx < W and 0 <= ny < H and x0<=nx<x1 and y0<=ny<y1):
                    continue

                # 基本自由空间检查
                if grid[ny][nx] < self.free_th:
                    continue

                # 厚实性检查
                if not is_robust_free(nx, ny):
                    continue

                robust_region.add((nx, ny))
                queue.append((nx, ny))

        # 3. 筛选在厚实连通区域内的前沿点
        qualified = []
        for px, py in candidates:
            if (px, py) in robust_region:
                qualified.append((px, py))

        # 4. 容错机制：结果太少时降级处理
        min_required = max(1, len(candidates) // 5)  # 至少保留20%或1个
        if len(qualified) < min_required:
            print(f"厚实连通区域前沿点过少({len(qualified)}/{len(candidates)})，")

        if not qualified:
            print("没有connectivity")
        return qualified if qualified else candidates[:2]

    def _fallback_connectivity_check(self, candidates, curr_pos, grid):
        """降级方案：使用普通连通性检查（不要求厚实）"""
        from collections import deque

        W = len(grid[0])
        H = len(grid)

        def is_basic_free(x, y):
            """基本自由空间检查"""
            return 0 <= x < W and 0 <= y < H and grid[y][x] >= self.free_th

        # 寻找有效起始点
        start_x, start_y = int(curr_pos[0]), int(curr_pos[1])

        if not is_basic_free(start_x, start_y):
            # 在附近寻找最近的自由空间
            for r in range(1, 10):
                found = False
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if dx * dx + dy * dy > r * r:
                            continue
                        nx, ny = start_x + dx, start_y + dy
                        if is_basic_free(nx, ny):
                            start_x, start_y = nx, ny
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            else:
                # 完全找不到连通起点，返回距离最近的几个前沿点
                return candidates[:2]

        # BFS扩展普通连通区域
        basic_region = set()
        queue = deque([(start_x, start_y)])
        basic_region.add((start_x, start_y))

        # 使用4连通，更保守
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if (nx, ny) in basic_region:
                    continue

                if not is_basic_free(nx, ny):
                    continue

                basic_region.add((nx, ny))
                queue.append((nx, ny))

        # 筛选在普通连通区域内的前沿点
        qualified = []
        for px, py in candidates:
            if (px, py) in basic_region:
                qualified.append((px, py))

        return qualified if qualified else candidates[:2]

    def detect_frontiers(self, grid: List[List[int]],
                         maze_rect: Optional[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
        """
        前沿检测方法保持不变，但可以在这里添加一些预筛选
        """
        H = len(grid)
        W = len(grid[0]) if H else 0
        if H == 0 or W == 0:
            return []

        if maze_rect is None:
            x0, y0, x1, y1 = 0, 0, W - 1, H - 1
        else:
            x0, y0, x1, y1 = maze_rect
            x0 = max(0, x0);
            y0 = max(0, y0);
            x1 = min(W - 1, x1);
            y1 = min(H - 1, y1)

        def has_unknown_ngb(x: int, y: int) -> bool:
            for jy in (-1, 0, 1):
                for ix in (-1, 0, 1):
                    if ix == 0 and jy == 0: continue
                    xn, yn = x + ix, y + jy
                    if 0 <= xn < W and 0 <= yn < H and self.obst_v  < grid[yn][xn] < self.free_th:
                        return True
            return False

        def safe_from_obstacle(x: int, y: int) -> bool:
            r = self.clear_r
            rr = r * r
            for jy in range(-r, r + 1):
                for ix in range(-r, r + 1):
                    if ix * ix + jy * jy > rr:
                        continue
                    xn, yn = x + ix, y + jy
                    if 0 <= xn < W and 0 <= yn < H and grid[yn][xn] <= self.obst_v:
                        return False
            return True

        res: List[Tuple[int, int]] = []
        for y in range(max(y0, 1), min(y1, H - 2) + 1):
            row = grid[y]
            for x in range(max(x0, 1), min(x1, W - 2) + 1):
                v = row[x]
                if v >= self.free_th and has_unknown_ngb(x, y) and safe_from_obstacle(x, y):
                    res.append((x, y))

        return res