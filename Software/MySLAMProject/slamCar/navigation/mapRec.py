from __future__ import annotations
from typing import List, Tuple, Optional


class MapRec:
    def __init__(self,
                 margin_pix: int = 20,          # ROI在四周外扩的像素边距（≥机器人半宽/像素）
                 min_width_pix: int = 5,      # 过滤掉过窄的盒子
                 min_height_pix: int = 5,
                 min_area_pix: int = 400,      # 最小面积过滤
                 allow_shrink: bool = False,   # 是否允许ROI收缩（默认不允许：更稳）
                 max_shrink_per_update: int = 4  # 若允许收缩，每次边界最多内缩像素
                 ):
        self.rect_pix: Optional[Tuple[int,int,int,int]] = None  # (x0,y0,x1,y1)
        self.margin_pix = int(margin_pix)
        self.min_w = int(min_width_pix)
        self.min_h = int(min_height_pix)
        self.min_area = int(min_area_pix)
        self.allow_shrink = bool(allow_shrink)
        self.max_shrink = int(max_shrink_per_update)

    def _compute_grid_hash_fast(self, grid: List[List[int]],
                                region: Optional[Tuple[int, int, int, int]] = None) -> int:
        """
        快速计算地图哈希（采样方式，避免全图遍历）
        """
        H, W = len(grid), len(grid[0])

        if region:
            x0, y0, x1, y1 = region
        else:
            # 采样策略：只检查每隔N行/列
            step = max(1, min(H, W) // 20)
            x0, y0, x1, y1 = 0, 0, W - 1, H - 1

        # 采样哈希
        sample_points = []
        for y in range(y0, y1 + 1, max(1, (y1 - y0) // 10)):
            for x in range(x0, x1 + 1, max(1, (x1 - x0) // 10)):
                if 0 <= y < H and 0 <= x < W:
                    sample_points.append(grid[y][x])

        return hash(tuple(sample_points))

    def _initial_scan_optimized(self, grid: List[List[int]],
                                ensure_contains: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        首次扫描优化：使用行列独立扫描
        时间复杂度：O(H + W) 而非 O(H×W)
        """
        H, W = len(grid), len(grid[0])

        # 策略1：先找到有效行范围（只扫描列）
        y_min, y_max = self._find_valid_rows(grid)
        if y_min is None:
            return self.rect_pix

        # 策略2：在有效行范围内找列边界
        x_min, x_max = self._find_valid_columns(grid, y_min, y_max)
        if x_min is None:
            return self.rect_pix

        # 计算边界
        return self._compute_final_bounds(
            x_min, y_min, x_max, y_max, W, H, ensure_contains
        )

    def _find_valid_rows(self, grid: List[List[int]]) -> Tuple[Optional[int], Optional[int]]:
        """
        快速找到包含已知区域的行范围
        只需要O(H×采样点)的复杂度
        """
        H, W = len(grid), len(grid[0])

        # 采样策略：每行只检查几个点
        sample_points = [0, W // 4, W // 2, 3 * W // 4, W - 1]

        y_min = None
        y_max = None

        # 从上往下找第一个有效行
        for y in range(H):
            has_known = any(grid[y][x] <= self._known_threshold
                            for x in sample_points if 0 <= x < W)
            if has_known:
                y_min = y
                break

        if y_min is None:
            return None, None

        # 从下往上找最后一个有效行
        for y in range(H - 1, y_min - 1, -1):
            has_known = any(grid[y][x] <= self._known_threshold
                            for x in sample_points if 0 <= x < W)
            if has_known:
                y_max = y
                break

        return y_min, y_max

    def _find_valid_columns(self, grid: List[List[int]],
                            y_min: int, y_max: int) -> Tuple[Optional[int], Optional[int]]:
        """
        在有效行范围内找到列边界
        """
        W = len(grid[0])

        # 采样行
        sample_rows = []
        step = max(1, (y_max - y_min) // 5)
        for y in range(y_min, y_max + 1, step):
            sample_rows.append(y)
        if y_max not in sample_rows:
            sample_rows.append(y_max)

        x_min = W
        x_max = -1

        for y in sample_rows:
            row = grid[y]
            # 从左边找
            for x in range(W):
                if row[x] <= self._known_threshold:
                    x_min = min(x_min, x)
                    break

            # 从右边找
            for x in range(W - 1, -1, -1):
                if row[x] <= self._known_threshold:
                    x_max = max(x_max, x)
                    break

        if x_max < x_min:
            return None, None

        return x_min, x_max

    def _update_from_changed_region(self, grid: List[List[int]],
                                    changed_region: Tuple[int, int, int, int],
                                    ensure_contains: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        增量更新：只扫描变化的区域
        适合实时SLAM中地图增量更新的场景
        """
        H, W = len(grid), len(grid[0])
        cx0, cy0, cx1, cy1 = changed_region

        # 扩展搜索区域（确保不遗漏边界）
        margin = 5
        cx0 = max(0, cx0 - margin)
        cy0 = max(0, cy0 - margin)
        cx1 = min(W - 1, cx1 + margin)
        cy1 = min(H - 1, cy1 + margin)

        # 只在变化区域内寻找边界
        x_min, y_min = W, H
        x_max, y_max = -1, -1
        known_count = 0

        for y in range(cy0, cy1 + 1):
            row = grid[y]
            for x in range(cx0, cx1 + 1):
                if row[x] <= self._known_threshold:
                    known_count += 1
                    x_min = min(x_min, x)
                    x_max = max(x_max, x)
                    y_min = min(y_min, y)
                    y_max = max(y_max, y)

        if known_count == 0:
            return self.rect_pix

        # 与现有ROI合并
        if self.rect_pix is not None:
            px0, py0, px1, py1 = self.rect_pix
            x_min = min(x_min, px0)
            y_min = min(y_min, py0)
            x_max = max(x_max, px1)
            y_max = max(y_max, py1)

        return self._compute_final_bounds(
            x_min, y_min, x_max, y_max, W, H, ensure_contains
        )

    #——————————————————————————————————————————————————————————————————————————————————————
    #公共调用
    #———————————————————————————————————————————————————————————————————————————————————————


    def update_from_grid(self, grid: List[List[int]],ensure_contains: Optional[Tuple[int,int]] = None) -> Optional[Tuple[int,int,int,int]]:

        if not grid or not grid[0]:
            return self.rect_pix

        H = len(grid)
        W = len(grid[0])

        x0, y0 = W, H
        x1, y1 = -1, -1
        known_count = 0

        for y in range(H):
            row = grid[y]
            for x in range(W):
                v = row[x]
                if v <= 20:
                    known_count += 1
                    if x < x0: x0 = x
                    if x > x1: x1 = x
                    if y < y0: y0 = y
                    if y > y1: y1 = y

        if known_count == 0:

            return self.rect_pix

        # 初始包围盒
        if x1 < x0 or y1 < y0:
            return self.rect_pix

        w = x1 - x0 + 1
        h = y1 - y0 + 1
        if w < self.min_w or h < self.min_h or (w * h) < self.min_area:
            return self.rect_pix
        x0m = max(0, x0 +4)
        y0m = max(0, y0  +4)
        x1m = min(W - 1, x1-8 )
        y1m = min(H - 1, y1 -8)

        if ensure_contains is not None:
            cx, cy = ensure_contains
            x0m = min(x0m, max(0, cx))
            y0m = min(y0m, max(0, cy))
            x1m = max(x1m, min(W - 1, cx))
            y1m = max(y1m, min(H - 1, cy))

        candidate = (x0m, y0m, x1m, y1m)

        if self.rect_pix is None:
            self.rect_pix = candidate
            return self.rect_pix

        cx0, cy0, cx1, cy1 = candidate
        px0, py0, px1, py1 = self.rect_pix

        nx0 = min(px0, cx0)
        ny0 = min(py0, cy0)
        nx1 = max(px1, cx1)
        ny1 = max(py1, cy1)

        if self.allow_shrink:

            nx0 = self._soft_shrink(px0, cx0, inward=True)
            ny0 = self._soft_shrink(py0, cy0, inward=True)
            nx1 = self._soft_shrink(px1, cx1, inward=False)
            ny1 = self._soft_shrink(py1, cy1, inward=False)

            if (nx1 - nx0 + 1) < self.min_w or (ny1 - ny0 + 1) < self.min_h:
                nx0, ny0, nx1, ny1 = px0, py0, px1, py1  # 放弃收缩

        self.rect_pix = (nx0, ny0, nx1, ny1)
        return self.rect_pix



    def _soft_shrink(self, prev_edge: int, cand_edge: int, inward: bool) -> int:

        if inward:

            if cand_edge <= prev_edge:
                return cand_edge  # 扩张或等同：直接采用（向外扩更稳）
            else:

                return min(prev_edge + self.max_shrink, cand_edge)
        else:

            if cand_edge >= prev_edge:
                return cand_edge
            else:
                return max(prev_edge - self.max_shrink, cand_edge)

    def is_inside(self, x_pix: float, y_pix: float) -> bool:

        if not self.rect_pix:
            return True
        x0, y0, x1, y1 = self.rect_pix
        return (x0 <= x_pix <= x1) and (y0 <= y_pix <= y1)

    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:

        return self.rect_pix
