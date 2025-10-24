from slamCar.navigation.config import  *

class GoalChecker:


    def __init__(self,
                 free_thresh: int = FREE_thresh,
                 unknown_val: int = 128,
                 obstacle_val: int = OBSTACLE_val,
                 check_radius_pix: int = 5,  # 终点周围需要自由的半径
                 path_check_interval: int = 1):  # 路径检查间隔（帧数）

        self.free_th = int(free_thresh)
        self.unknown_v = int(unknown_val)
        self.obst_v = int(obstacle_val)
        self.check_radius = int(check_radius_pix)
        self.check_interval = int(path_check_interval)

        self._frame_count = 0  # 帧计数器
        self._last_check_result = False  # 上次检查结果（缓存）

    def _has_path_bidirectional_bfs(self,
                                    grid: List[List[int]],
                                    start: Tuple[int, int],
                                    goal: Tuple[int, int],
                                    maze_rect: Optional[Tuple[int, int, int, int]]) -> bool:
        """
        双向BFS：从起点和终点同时搜索，相遇即找到路径
        理论加速：2倍以上（搜索空间从O(n²)降为O(2*(n/2)²) = O(n²/2)）
        """
        from collections import deque

        H, W = len(grid), len(grid[0])

        if maze_rect:
            x0, y0, x1, y1 = maze_rect
            x0 = max(0, x0);
            y0 = max(0, y0)
            x1 = min(W - 1, x1);
            y1 = min(H - 1, y1)
        else:
            x0, y0, x1, y1 = 0, 0, W - 1, H - 1

        sx, sy = start
        gx, gy = goal

        # 边界检查
        if not (x0 <= sx <= x1 and y0 <= sy <= y1): return False
        if not (x0 <= gx <= x1 and y0 <= gy <= y1): return False

        # 初始化前向和后向搜索
        forward_queue = deque([start])
        backward_queue = deque([goal])

        forward_visited = {start: 0}  # 存储距离
        backward_visited = {goal: 0}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # 动态计算最大迭代次数（基于曼哈顿距离）
        manhattan_dist = abs(gx - sx) + abs(gy - sy)
        max_iterations = min(manhattan_dist * 3, 5000)  # 更合理的上限

        iterations = 0

        while forward_queue and backward_queue and iterations < max_iterations:
            iterations += 1

            # 优先扩展较小的队列（平衡搜索）
            if len(forward_queue) <= len(backward_queue):
                # 前向搜索一步
                current = forward_queue.popleft()
                x, y = current

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    if (nx, ny) in forward_visited:
                        continue

                    if not (x0 <= nx <= x1 and y0 <= ny <= y1):
                        continue

                    if grid[ny][nx] < self.free_th:
                        continue

                    # 检查是否与后向搜索相遇
                    if (nx, ny) in backward_visited:
                        print(f"✓ 路径找到！前向步数: {forward_visited[current] + 1}, "
                              f"后向步数: {backward_visited[(nx, ny)]}")
                        return True

                    forward_visited[(nx, ny)] = forward_visited[current] + 1
                    forward_queue.append((nx, ny))

            else:
                # 后向搜索一步
                current = backward_queue.popleft()
                x, y = current

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    if (nx, ny) in backward_visited:
                        continue

                    if not (x0 <= nx <= x1 and y0 <= ny <= y1):
                        continue

                    if grid[ny][nx] < self.free_th:
                        continue

                    # 检查是否与前向搜索相遇
                    if (nx, ny) in forward_visited:
                        print(f"✓ 路径找到！前向步数: {forward_visited[(nx, ny)]}, "
                              f"后向步数: {backward_visited[current] + 1}")
                        return True

                    backward_visited[(nx, ny)] = backward_visited[current] + 1
                    backward_queue.append((nx, ny))

        return False

    def _has_path_astar(self,
                        grid: List[List[int]],
                        start: Tuple[int, int],
                        goal: Tuple[int, int],
                        maze_rect: Optional[Tuple[int, int, int, int]]) -> bool:
        """
        A*算法：使用启发式函数导向搜索
        适合：已知终点位置，需要快速判断可达性
        """
        import heapq

        H, W = len(grid), len(grid[0])

        if maze_rect:
            x0, y0, x1, y1 = maze_rect
            x0 = max(0, x0);
            y0 = max(0, y0)
            x1 = min(W - 1, x1);
            y1 = min(H - 1, y1)
        else:
            x0, y0, x1, y1 = 0, 0, W - 1, H - 1

        sx, sy = start
        gx, gy = goal

        if not (x0 <= sx <= x1 and y0 <= sy <= y1): return False
        if not (x0 <= gx <= x1 and y0 <= gy <= y1): return False

        def heuristic(pos):
            """曼哈顿距离启发式"""
            return abs(pos[0] - gx) + abs(pos[1] - gy)

        # 优先队列：(f_score, g_score, position)
        open_set = [(heuristic(start), 0, start)]
        visited = {start}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # 根据启发式距离设置更紧的上限
        max_iterations = heuristic(start) * 2  # 理论最短路径的2倍
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1

            f_score, g_score, current = heapq.heappop(open_set)
            x, y = current

            # 到达目标（使用容差）
            if abs(x - gx) <= 8 and abs(y - gy) <= 8:
                print(f"✓ A*找到路径！步数: {g_score}, 迭代: {iterations}")
                return True

            # 早期剪枝：如果当前代价已经过高
            if g_score > heuristic(start) * 3:
                continue

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if (nx, ny) in visited:
                    continue

                if not (x0 <= nx <= x1 and y0 <= ny <= y1):
                    continue

                if grid[ny][nx] < self.free_th:
                    continue

                visited.add((nx, ny))
                new_g = g_score + 1
                new_f = new_g + heuristic((nx, ny))

                heapq.heappush(open_set, (new_f, new_g, (nx, ny)))

        return False

    #——————————————————————————————————————————————————————————————————————————————————————
    #公共调用
    #———————————————————————————————————————————————————————————————————————————————————————

    def is_goal_reachable(self,
                          grid: List[List[int]],
                          curr_pix: Tuple[int, int],
                          goal_pix: Tuple[int, int],
                          maze_rect: Optional[Tuple[int, int, int, int]] = None,
                          force_check: bool = True) -> bool:

        # 帧间隔控制（减少计算开销）
        self._frame_count += 1
        if not force_check and self._frame_count % self.check_interval != 0:
            return self._last_check_result

        H = len(grid)
        W = len(grid[0]) if H > 0 else 0

        if W == 0 or H == 0:
            return False

        gx, gy = int(round(goal_pix[0])), int(round(goal_pix[1]))

        # 检查1：终点是否在地图范围内
        if not (0 <= gx < W and 0 <= gy < H):
            print(f"终点({gx},{gy})超出地图范围")
            self._last_check_result = False
            return False

        # 检查2：终点周围是否为自由空间
        if not self._is_goal_area_free(grid, gx, gy):
            print(f"终点区域未探索或有障碍物")
            self._last_check_result = False
            return False

        # 检查3：从当前位置到终点是否存在路径
        cx, cy = int(round(curr_pix[0])), int(round(curr_pix[1]))

        if self._has_path_bfs(grid, (cx, cy), (gx, gy), maze_rect):
            print(f"✓ 终点可达！当前({cx},{cy}) -> 终点({gx},{gy})")
            self._last_check_result = True
            return True
        else:
            print(f"✗ 终点尚不可达，继续探索")
            self._last_check_result = False
            return False

    def _is_goal_area_free(self, grid: List[List[int]],
                           gx: int, gy: int) -> bool:

        H, W = len(grid), len(grid[0])

        # 检查终点本身
        if not (0 <= gx < W and 0 <= gy < H):
            return False

        if grid[gy][gx] < self.free_th:
            return False  # 终点不是自由空间

        # 检查周围区域
        r = self.check_radius
        free_count = 0
        total_count = 0

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue

                nx, ny = gx + dx, gy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    total_count += 1
                    if grid[ny][nx] >= self.free_th:
                        free_count += 1

        # 至少70%的周围区域是自由空间
        if total_count == 0:
            return False

        free_ratio = free_count / total_count
        return free_ratio >= 0.7

    def _has_path_bfs(self,
                      grid: List[List[int]],
                      start: Tuple[int, int],
                      goal: Tuple[int, int],
                      maze_rect: Optional[Tuple[int, int, int, int]]) -> bool:

        from collections import deque

        H, W = len(grid), len(grid[0])

        # ROI范围
        if maze_rect:
            x0, y0, x1, y1 = maze_rect
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(W - 1, x1)
            y1 = min(H - 1, y1)
        else:
            x0, y0, x1, y1 = 0, 0, W - 1, H - 1

        sx, sy = start
        gx, gy = goal

        # 边界检查
        if not (x0 <= sx <= x1 and y0 <= sy <= y1):
            return False
        if not (x0 <= gx <= x1 and y0 <= gy <= y1):
            return False

        # BFS
        queue = deque([(sx, sy)])
        visited = {(sx, sy)}

        # 4连通（简化版，更快）
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        max_iterations = 5000  # 防止无限循环
        iterations = 0

        while queue and iterations < max_iterations:
            iterations += 1
            x, y = queue.popleft()

            # 到达终点
            if abs(x - gx) <= 8 and abs(y - gy) <= 8:
                return True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                if (nx, ny) in visited:
                    continue

                # 范围检查
                if not (x0 <= nx <= x1 and y0 <= ny <= y1):
                    continue

                # 可通行检查：自由空间
                if grid[ny][nx] < self.free_th:
                    continue

                visited.add((nx, ny))
                queue.append((nx, ny))

        return False