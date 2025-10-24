from slamCar.navigation.config import  *

from typing import List, Tuple, Optional, Dict
import math



class PathPlanner:

    def __init__(self,
                 allow_diagonal: bool = True,
                 inflate_radius_pix: int = 15,
                 free_thresh: int = FREE_thresh,         # >= 此灰度视作自由
                 unknown_val: int = 128,         # 未知
                 obstacle_val: int = OBSTACLE_val,          # 障碍
                 max_snap_radius: int = 8,        # 起/终点就近“吸附”到可通行格的最大半径
                 # 新增简化参数
                 enable_path_simplification: bool = True,
                 target_waypoint_distance_pix: int = 7
                 ):
        self.allow_diag = bool(allow_diagonal)
        self.inflate_r = int(max(0, inflate_radius_pix))
        self.free_th = int(free_thresh)
        self.unknown_v = int(unknown_val)
        self.obst_v = int(obstacle_val)
        self.max_snap_r = int(max(0, max_snap_radius))

        # 新增属性
        self.enable_simplification = enable_path_simplification
        self.target_distance = target_waypoint_distance_pix

    def _jump_point_search(self,
                           passable: List[List[bool]],
                           start: Tuple[int, int],
                           goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        跳点搜索（JPS）：A*的网格优化版本
        通过"跳过"对称路径来减少搜索节点数

        适用场景：
        - 网格地图
        - 允许对角移动
        - 障碍物不密集
        """
        import heapq

        H, W = len(passable), len(passable[0])
        sx, sy = start
        gx, gy = goal

        def heuristic(x: int, y: int) -> float:
            dx, dy = abs(x - gx), abs(y - gy)
            D, D2 = 1.0, 1.414213562
            return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

        def is_walkable(x: int, y: int) -> bool:
            return 0 <= x < W and 0 <= y < H and passable[y][x]

        def jump(x: int, y: int, dx: int, dy: int) -> Optional[Tuple[int, int]]:
            """
            跳跃函数：沿方向(dx,dy)跳跃，直到找到关键点或障碍
            """
            nx, ny = x + dx, y + dy

            # 越界或障碍
            if not is_walkable(nx, ny):
                return None

            # 到达目标
            if (nx, ny) == (gx, gy):
                return (nx, ny)

            # 对角移动
            if dx != 0 and dy != 0:
                # 检查强制邻居（forced neighbors）
                if (is_walkable(nx - dx, ny) and not is_walkable(nx - dx, y)) or \
                        (is_walkable(nx, ny - dy) and not is_walkable(x, ny - dy)):
                    return (nx, ny)

                # 在水平和垂直方向上递归跳跃
                if jump(nx, ny, dx, 0) is not None or jump(nx, ny, 0, dy) is not None:
                    return (nx, ny)

            # 直线移动
            else:
                if dx != 0:  # 水平
                    if (is_walkable(nx, ny + 1) and not is_walkable(x, ny + 1)) or \
                            (is_walkable(nx, ny - 1) and not is_walkable(x, ny - 1)):
                        return (nx, ny)
                else:  # 垂直
                    if (is_walkable(nx + 1, ny) and not is_walkable(nx + 1, y)) or \
                            (is_walkable(nx - 1, ny) and not is_walkable(nx - 1, y)):
                        return (nx, ny)

            # 继续跳跃
            return jump(nx, ny, dx, dy)

        def get_successors(x: int, y: int, parent_x: int, parent_y: int) -> List[Tuple[int, int]]:
            """获取跳点的后继节点"""
            successors = []

            # 确定搜索方向（基于父节点）
            if parent_x == -1:  # 起点
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                              (-1, -1), (1, -1), (-1, 1), (1, 1)]
            else:
                # 剪枝：只搜索"自然"和"强制"邻居
                dx = (x - parent_x) // max(1, abs(x - parent_x)) if x != parent_x else 0
                dy = (y - parent_y) // max(1, abs(y - parent_y)) if y != parent_y else 0

                directions = []
                if dx != 0 and dy != 0:  # 对角
                    directions.extend([(dx, 0), (0, dy), (dx, dy)])
                else:  # 直线
                    directions.append((dx if dx != 0 else 0, dy if dy != 0 else 0))

            for dx, dy in directions:
                jump_point = jump(x, y, dx, dy)
                if jump_point:
                    successors.append(jump_point)

            return successors

        # JPS主循环
        open_heap = [(heuristic(sx, sy), 0, sx, sy, -1, -1)]  # (f, g, x, y, px, py)
        closed = set()
        parents = {}
        g_scores = {(sx, sy): 0}

        while open_heap:
            _, g, x, y, px, py = heapq.heappop(open_heap)

            if (x, y) in closed:
                continue

            if (x, y) == (gx, gy):
                # 重建路径（需要在跳点之间插值）
                return self._reconstruct_jps_path(parents, (gx, gy), (sx, sy), passable)

            closed.add((x, y))
            parents[(x, y)] = (px, py)

            # 获取后继跳点
            for nx, ny in get_successors(x, y, px, py):
                if (nx, ny) in closed:
                    continue

                # 计算代价
                dx, dy = abs(nx - x), abs(ny - y)
                if dx == dy:
                    cost = dx * 1.414213562
                else:
                    cost = dx + dy

                tentative_g = g + cost

                if (nx, ny) not in g_scores or tentative_g < g_scores[(nx, ny)]:
                    g_scores[(nx, ny)] = tentative_g
                    f = tentative_g + heuristic(nx, ny)
                    heapq.heappush(open_heap, (f, tentative_g, nx, ny, x, y))

        return []

    def _reconstruct_jps_path(self,
                              parents: Dict,
                              goal: Tuple[int, int],
                              start: Tuple[int, int],
                              passable: List[List[bool]]) -> List[Tuple[int, int]]:
        """
        重建JPS路径：在跳点之间插入中间点
        """
        # 提取跳点
        jump_points = [goal]
        current = goal

        while current in parents and parents[current] != (-1, -1):
            current = parents[current]
            jump_points.append(current)

        jump_points.reverse()

        # 在跳点之间插值
        full_path = []
        for i in range(len(jump_points) - 1):
            x0, y0 = jump_points[i]
            x1, y1 = jump_points[i + 1]

            # Bresenham插值
            interpolated = self._bresenham_line(x0, y0, x1, y1)
            full_path.extend(interpolated[:-1])  # 避免重复点

        full_path.append(jump_points[-1])
        return full_path

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham直线算法"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            points.append((x, y))

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return points
    #——————————————————————————————————————————————————————————————————————————————————————
    #公共调用
    #———————————————————————————————————————————————————————————————————————————————————————
    def _astar(self,
               passable: List[List[bool]],
               start: Tuple[int, int],
               goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        H = len(passable)
        W = len(passable[0]) if H else 0
        if H == 0 or W == 0:
            return []

        sx, sy = start
        gx, gy = goal

        # 邻接偏移与移动代价
        if self.allow_diag:
            nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                    (-1, -1, math.sqrt(2)), (1, -1, math.sqrt(2)),
                    (-1, 1, math.sqrt(2)), (1, 1, math.sqrt(2))]
        else:
            nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]

        def heuristic(x: int, y: int) -> float:
            dx = abs(x - gx)
            dy = abs(y - gy)
            if self.allow_diag:
                D = 1.0
                D2 = math.sqrt(2)
                return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)  # octile
            else:
                return dx + dy  # 曼哈顿

        # 打开表（小根堆）: (f, tie, x, y)
        import heapq
        open_heap: List[Tuple[float, int, int, int]] = []
        tie = 0

        g = { (sx, sy): 0.0 }
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

        heapq.heappush(open_heap, (heuristic(sx, sy), tie, sx, sy))
        tie += 1

        closed = set()

        while open_heap:
            _, _, x, y = heapq.heappop(open_heap)
            if (x, y) in closed:
                continue
            if (x, y) == (gx, gy):
                return self._reconstruct(parent, (gx, gy), (sx, sy))
            closed.add((x, y))

            for dx, dy, step_cost in nbrs:
                nx, ny = x + dx, y + dy
                if not self._is_inside(nx, ny, W, H):
                    continue
                if not passable[ny][nx]:
                    continue
                # 8 邻接时禁止“拐角穿越”：斜向移动要求两侧正交邻接均可通行
                if self.allow_diag and dx != 0 and dy != 0:
                    if not (passable[y][ny - dy] if False else True):  # 占位，便于阅读
                        pass  # 无效
                    ax, ay = x + dx, y
                    bx, by = x, y + dy
                    if not (passable[ay][ax] and passable[by][bx]):
                        continue

                new_g = g[(x, y)] + step_cost

                # 稍微加入“直线偏好”的微小代价，有助于打破等代价抖动
                # （也可以用父方向向量的微罚项；此处保持简单）
                if (nx, ny) not in g or new_g < g[(nx, ny)]:
                    g[(nx, ny)] = new_g
                    parent[(nx, ny)] = (x, y)
                    f = new_g + heuristic(nx, ny)
                    heapq.heappush(open_heap, (f, tie, nx, ny))
                    tie += 1

        return []

    def simplify_path(self,
                      raw_path: List[Tuple[int, int]],
                      grid: List[List[int]],
                      target_distance_pix: int = 20,
                      min_distance_pix: int = 2,
                      turn_angle_threshold_deg: float = 20.0) -> List[Tuple[int, int]]:

        if len(raw_path) <= 2:
            return raw_path

        simplified = [raw_path[0]]  # 起点必须保留
        current_idx = 0

        while current_idx < len(raw_path) - 1:
            best_next_idx = self._find_best_next_waypoint(
                raw_path, grid, current_idx,
                target_distance_pix, min_distance_pix, turn_angle_threshold_deg
            )

            simplified.append(raw_path[best_next_idx])
            current_idx = best_next_idx

        # 确保终点被包含（可能已经在上面添加了）
        if simplified[-1] != raw_path[-1]:
            simplified.append(raw_path[-1])

        return simplified

    def _find_best_next_waypoint(self,
                                 raw_path: List[Tuple[int, int]],
                                 grid: List[List[int]],
                                 current_idx: int,
                                 target_dist: int,
                                 min_dist: int,
                                 turn_threshold_deg: float) -> int:

        current_pos = raw_path[current_idx]

        # 候选点：从当前点向后搜索
        best_idx = current_idx + 1  # 最差情况下选择下一个点
        best_score = float('inf')

        max_search_range = min(current_idx + target_dist * 2, len(raw_path) - 1)

        for candidate_idx in range(current_idx + 1, max_search_range + 1):
            candidate_pos = raw_path[candidate_idx]
            distance = self._pixel_distance(current_pos, candidate_pos)

            # 距离太小，跳过（除非是最后一个点）
            if distance < min_dist and candidate_idx < len(raw_path) - 1:
                continue

            # 检查是否穿墙
            if not self._is_line_clear(grid, current_pos, candidate_pos):
                continue

            # 计算得分（距离越接近target_dist越好）
            distance_score = abs(distance - target_dist)

            # 检查是否为转弯点，转弯点允许较短距离
            is_turn = self._is_turning_point(raw_path, candidate_idx, turn_threshold_deg)
            if is_turn and distance >= min_dist:
                distance_score *= 0.7  # 转弯点得分加权

            if distance_score < best_score:
                best_score = distance_score
                best_idx = candidate_idx

            # 如果找到了接近目标距离的点，可以早期退出
            if distance >= target_dist * 0.9 and distance <= target_dist * 1.1:
                break

        return best_idx

    def _pixel_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """计算两个像素点之间的欧几里德距离"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _is_line_clear(self, grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> bool:

        H = len(grid)
        W = len(grid[0]) if H > 0 else 0

        x0, y0 = start
        x1, y1 = end

        # Bresenham直线算法
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        if dx == 0 and dy == 0:
            return True

        x, y = x0, y0
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1

        if dx > dy:
            error = dx / 2
            while x != x1:
                # 检查当前点是否为障碍物
                if not (0 <= x < W and 0 <= y < H) or grid[y][x] <= self.obst_v:
                    return False

                error -= dy
                if error < 0:
                    y += y_inc
                    error += dx
                x += x_inc
        else:
            error = dy / 2
            while y != y1:
                # 检查当前点是否为障碍物
                if not (0 <= x < W and 0 <= y < H) or grid[y][x] <= self.obst_v:
                    return False

                error -= dx
                if error < 0:
                    x += x_inc
                    error += dy
                y += y_inc

        # 检查终点
        if not (0 <= x1 < W and 0 <= y1 < H) or grid[y1][x1] <= self.obst_v:
            return False

        return True

    def _is_turning_point(self, path: List[Tuple[int, int]], idx: int, angle_threshold: float) -> bool:
        """
        检查某个路径点是否为转弯点

        通过计算前后方向向量的角度变化来判断
        """
        if idx < 2 or idx >= len(path) - 1:
            return False

        # 获取三个连续点
        p1 = path[idx - 2]
        p2 = path[idx]
        p3 = path[idx + 2] if idx + 2 < len(path) else path[-1]

        # 计算两个方向向量
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # 计算角度变化
        angle_change = self._vector_angle_diff(v1, v2)

        return abs(angle_change) > angle_threshold

    def _vector_angle_diff(self, v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """计算两个向量之间的角度差（度）"""
        import math

        # 防止零向量
        len1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        len2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if len1 < 1e-6 or len2 < 1e-6:
            return 0.0

        # 计算角度
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])

        # 角度差，归一化到[-180, 180]
        diff = math.degrees(angle2 - angle1)
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        return diff

    # 修改原有的plan_path方法，添加路径简化选项
    def plan_path(self,
                  grid: List[List[int]],
                  start_pix: Tuple[int, int],
                  goal_pix: Tuple[int, int],
                  maze_rect: Optional[Tuple[int, int, int, int]],
                  simplify: bool = True,
                  target_distance_pix: int = 40) -> List[Tuple[int, int]]:

        # 调用原有的A*算法
        raw_path = self._plan_path_original(grid, start_pix, goal_pix, maze_rect)

        if not raw_path or not simplify:
            print("not raw path")
            return raw_path

        # 简化路径
        simplified_path = self.simplify_path(
            raw_path, grid,
            target_distance_pix=target_distance_pix,
            min_distance_pix=max(2, target_distance_pix // 3)
        )

        print(f"路径简化: {len(raw_path)} -> {len(simplified_path)} 点 "
              f"(压缩率: {len(simplified_path) / len(raw_path):.2%})")

        return simplified_path

    def _plan_path_original(self,
                            grid: List[List[int]],
                            start_pix: Tuple[int, int],
                            goal_pix: Tuple[int, int],
                            maze_rect: Optional[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:

        # 这里是原来plan_path方法的所有代码
        # [保持原有的A*实现不变]
        if not grid or not grid[0]:
            print("121212")
            return []

        H = len(grid)
        W = len(grid[0])

        # 1) 约束 ROI
        if maze_rect is None:
            x0, y0, x1, y1 = 0, 0, W - 1, H - 1
        else:
            x0, y0, x1, y1 = maze_rect
            x0 = max(0, min(x0, W - 1))
            x1 = max(0, min(x1, W - 1))
            y0 = max(0, min(y0, H - 1))
            y1 = max(0, min(y1, H - 1))
            if x1 < x0 or y1 < y0:
                print("232323")
                return []

        # 2) 在 ROI 内构建"可通行掩模"并对障碍做膨胀
        passable, roi_W, roi_H = self._build_passable_with_inflation(grid, (x0, y0, x1, y1))

        # 3) 将 start/goal 转到 ROI 局部坐标，并吸附到最近可通行点
        sx, sy = start_pix
        gx, gy = goal_pix
        if not (x0 <= sx <= x1 and y0 <= sy <= y1 and x0 <= gx <= x1 and y0 <= gy <= y1):

            return []

        lsx, lsy = sx - x0, sy - y0
        lgx, lgy = gx - x0, gy - y0

        if not self._is_inside(lsx, lsy, roi_W, roi_H) or not self._is_inside(lgx, lgy, roi_W, roi_H):
            return []

        if not passable[lsy][lsx]:

            snap = self._snap_to_nearest_passable(passable, lsx, lsy)
            if snap is None:

                return []
            lsx, lsy = snap

        if not passable[lgy][lgx]:

            snap = self._snap_to_nearest_passable(passable, lgx, lgy)
            if snap is None:
                return []
            lgx, lgy = snap

        # 若起点=终点，直接返回
        if (lsx, lsy) == (lgx, lgy):
            return [(lsx + x0, lsy + y0)]

        # 4) A* 搜索
        path_local = self._astar(passable, (lsx, lsy), (lgx, lgy))
        # 5) 回到全局像素坐标
        if not path_local:
            return []
        return [(x + x0, y + y0) for (x, y) in path_local]

    @staticmethod
    def _reconstruct(parent: Dict[Tuple[int, int], Tuple[int, int]],
                     goal: Tuple[int, int],
                     start: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [goal]
        cur = goal
        while cur != start:
            cur = parent.get(cur)
            if cur is None:
                return []  # 防御性：不应发生
            path.append(cur)
        path.reverse()
        return path

    def _build_passable_with_inflation(self,
                                       grid: List[List[int]],
                                       roi: Tuple[int, int, int, int]
                                       ) -> Tuple[List[List[bool]], int, int]:

        x0, y0, x1, y1 = roi
        H = len(grid)
        W = len(grid[0]) if H else 0
        roi_W = x1 - x0 + 1
        roi_H = y1 - y0 + 1

        # 初始 passable
        passable: List[List[bool]] = [[False] * roi_W for _ in range(roi_H)]
        obstacles: List[Tuple[int, int]] = []

        for y in range(y0, y1 + 1):
            row = grid[y]
            py = y - y0
            for x in range(x0, x1 + 1):
                v = row[x]
                px = x - x0
                if v >= self.free_th:
                    passable[py][px] = True
                if v <= self.obst_v:
                    obstacles.append((px, py))

        # 膨胀：把障碍周围 r 内的 passable 置 False
        r = self.inflate_r
        if r > 0 and obstacles:
            rr = r * r
            for (ox, oy) in obstacles:
                # 在局部小窗口内遍历，避免全图遍历
                x_min = max(0, ox - r)
                x_max = min(roi_W - 1, ox + r)
                y_min = max(0, oy - r)
                y_max = min(roi_H - 1, oy + r)
                for py in range(y_min, y_max + 1):
                    dy = py - oy
                    dy2 = dy * dy
                    for px in range(x_min, x_max + 1):
                        dx = px - ox
                        if dx * dx + dy2 <= rr:
                            passable[py][px] = False

        return passable, roi_W, roi_H

    def _build_passable_with_inflation_optimized(self,
                                                 grid: List[List[int]],
                                                 roi: Tuple[int, int, int, int]
                                                 ) -> Tuple[List[List[bool]], int, int]:
        """
        优化的障碍物膨胀：使用距离变换
        时间复杂度：O(W×H)，而非O(n×r²)
        """
        import numpy as np
        from scipy.ndimage import distance_transform_edt

        x0, y0, x1, y1 = roi
        H, W = len(grid), len(grid[0])
        roi_W, roi_H = x1 - x0 + 1, y1 - y0 + 1

        # 构建初始通行性地图
        passable_array = np.zeros((roi_H, roi_W), dtype=bool)

        for y in range(y0, y1 + 1):
            py = y - y0
            for x in range(x0, x1 + 1):
                px = x - x0
                v = grid[y][x]
                passable_array[py, px] = (v >= self.free_th)

        if self.inflate_r > 0:
            # 使用距离变换一次性计算所有点到最近障碍物的距离
            # distance_transform_edt：欧几里得距离变换，O(W×H)复杂度
            distances = distance_transform_edt(passable_array)

            # 距离小于膨胀半径的点标记为不可通行
            passable_array = distances > self.inflate_r

        # 转换为List[List[bool]]
        passable = [[bool(passable_array[y, x]) for x in range(roi_W)]
                    for y in range(roi_H)]

        return passable, roi_W, roi_H
    #——————————————————————————————————————————————————————————————————————————————————————
    #公共调用
    #———————————————————————————————————————————————————————————————————————————————————————
    def _build_passable_fast_no_scipy(self,
                                      grid: List[List[int]],
                                      roi: Tuple[int, int, int, int]
                                      ) -> Tuple[List[List[bool]], int, int]:
        """
        不依赖scipy的快速膨胀算法：基于BFS的多源最短路径
        """
        from collections import deque

        x0, y0, x1, y1 = roi
        roi_W, roi_H = x1 - x0 + 1, y1 - y0 + 1

        # 初始化
        passable = [[False] * roi_W for _ in range(roi_H)]
        distance_map = [[float('inf')] * roi_W for _ in range(roi_H)]

        # 收集所有障碍物作为BFS的多个起点
        queue = deque()

        for y in range(y0, y1 + 1):
            py = y - y0
            for x in range(x0, x1 + 1):
                px = x - x0
                v = grid[y][x]

                if v >= self.free_th:
                    passable[py][px] = True

                if v <= self.obst_v:
                    distance_map[py][px] = 0
                    queue.append((px, py, 0))

        # 多源BFS：从所有障碍物同时扩散
        if self.inflate_r > 0:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (1, -1), (-1, 1), (1, 1)]

            while queue:
                x, y, dist = queue.popleft()

                if dist >= self.inflate_r:
                    continue

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    if not (0 <= nx < roi_W and 0 <= ny < roi_H):
                        continue

                    # 计算实际欧几里得距离
                    new_dist = math.sqrt(dx * dx + dy * dy) + dist

                    if new_dist < distance_map[ny][nx] and new_dist <= self.inflate_r:
                        distance_map[ny][nx] = new_dist
                        passable[ny][nx] = False  # 膨胀区域不可通行
                        queue.append((nx, ny, new_dist))

        return passable, roi_W, roi_H
    #——————————————————————————————————————————————————————————————————————————————————————
    #公共调用
    #———————————————————————————————————————————————————————————————————————————————————————
    def _snap_to_nearest_passable(self,
                                  passable: List[List[bool]],
                                  sx: int, sy: int,
                                  search_step: int = 3) -> Optional[Tuple[int, int]]:
        H = len(passable)
        W = len(passable[0]) if H else 0
        if self._is_inside(sx, sy, W, H) and passable[sy][sx]:
            return (sx, sy)

        from collections import deque
        q = deque()
        seen = set()
        q.append((sx, sy, 0))
        seen.add((sx, sy))

        # 根据 search_step 生成多尺度邻域
        dirs = []
        for dy in range(-search_step, search_step + 1):
            for dx in range(-search_step, search_step + 1):
                if dx == 0 and dy == 0:
                    continue
                # 仍保持8邻接原则，但允许跳步
                if max(abs(dx), abs(dy)) <= search_step:
                    dirs.append((dx, dy))

        while q:
            x, y, d = q.popleft()
            if d > self.max_snap_r:
                break
            if self._is_inside(x, y, W, H) and passable[y][x]:
                return (x, y)
            nd = d + 1
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if not self._is_inside(nx, ny, W, H):
                    continue
                if (nx, ny) in seen:
                    continue
                seen.add((nx, ny))
                q.append((nx, ny, nd))

        return None

    @staticmethod
    def _is_inside(x: int, y: int, W: int, H: int) -> bool:
        return 0 <= x < W and 0 <= y < H