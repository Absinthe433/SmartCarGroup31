
from typing import List, Tuple, Optional
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches


class Drawer:
    ROBOT_HEAD_LEN_M  = 0.09
    ROBOT_HEAD_W_M    = 0.06
    ROBOT_SHAFT_LEN_M = 0.05

    def __init__(self, map_size_pixels: int, map_size_meters: float,
                 title: str = "ListGridMap Viewer",
                 show_trajectory: bool = True,
                 origin_lower_left: bool = True):
        self.N = map_size_pixels
        self.map_size_meters = float(map_size_meters)
        self.m_per_pix = self.map_size_meters / float(self.N)
        self.title = title
        self.show_traj = show_trajectory
        self.origin = "lower" if origin_lower_left else "upper"

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        try: self.fig.canvas.set_window_title("SLAM")
        except Exception: pass
        #self.ax.set_title(self.title)
        self.ax.set_xlabel("X (m)"); self.ax.set_ylabel("Y (m)")
        self.ax.set_aspect("equal")
        self.ax.axis('off')
        self.extent_m = (0.0, self.N*self.m_per_pix, 0.0, self.N*self.m_per_pix)
        self.ax.set_xlim(self.extent_m[0], self.extent_m[1])
        self.ax.set_ylim(self.extent_m[2], self.extent_m[3])

        self.img_artist = None
        self.car_artist = None
        self.text_artist = None          # <<< 新增：复用的文字对象
        self.traj_line: Optional[mlines.Line2D] = None  # <<< 新增：复用的轨迹对象
        self.traj_xm: List[float] = []
        self.traj_ym: List[float] = []
        self.hud_artist = None

        self._roi_artist = None
        plt.ion(); plt.show(block=False)
        # 新增：前沿点可视化相关
        self.frontier_artists: List[plt.Circle] = []
        self.best_frontier_artist: Optional[plt.Circle] = None
        self.exit_pose_artist: Optional[plt.Circle] = None

    def _draw_frontier_points(self,
                              frontier_points: Optional[List[Tuple[int, int]]],
                              best_frontier_point: Optional[Tuple[int, int]]):
        """绘制前沿点，普通前沿点用蓝色圆点，最优前沿点用红色圆点"""

        # 清除旧的前沿点
        for artist in self.frontier_artists:
            try:
                artist.remove()
            except:
                pass
        self.frontier_artists.clear()

        if self.best_frontier_artist is not None:
            try:
                self.best_frontier_artist.remove()
            except:
                pass
            self.best_frontier_artist = None

        # 绘制所有前沿点（蓝色小圆点）
        if frontier_points:
            for x_pix, y_pix in frontier_points:
                x_m, y_m = self._pix_to_m(x_pix, y_pix)
                circle = plt.Circle((x_m, y_m), radius=0.01,
                                    color='cyan', alpha=0.8, zorder=4)
                self.ax.add_patch(circle)
                self.frontier_artists.append(circle)

        # 绘制最优前沿点（红色大圆点）
        if best_frontier_point is not None:
            x_pix, y_pix = best_frontier_point
            x_m, y_m = self._pix_to_m(x_pix, y_pix)
            self.best_frontier_artist = plt.Circle((x_m, y_m), radius=0.08,
                                                   color='blue', alpha=0.9, zorder=5)
            self.ax.add_patch(self.best_frontier_artist)

    def _draw_frontier_points_extra(self,
                              frontier_points: Optional[List[Tuple[int, int]]],
                              best_frontier_point: Optional[Tuple[int, int]]):
        """绘制前沿点，普通前沿点用蓝色圆点，最优前沿点用红色圆点"""

        # 清除旧的前沿点
        for artist in self.frontier_artists:
            try:
                artist.remove()
            except:
                pass
        self.frontier_artists.clear()

        if self.best_frontier_artist is not None:
            try:
                self.best_frontier_artist.remove()
            except:
                pass
            self.best_frontier_artist = None

        # 绘制所有前沿点（lvse小圆点）
        if frontier_points:
            for x_pix, y_pix in frontier_points:
                x_m, y_m = self._pix_to_m(x_pix, y_pix)
                circle = plt.Circle((x_m, y_m), radius=0.05,
                                    color='yellow', alpha=0.8, zorder=4)
                self.ax.add_patch(circle)
                self.frontier_artists.append(circle)

        # 绘制最优前沿点（红色大圆点）
        if best_frontier_point is not None:
            x_pix, y_pix = best_frontier_point
            x_m, y_m = self._pix_to_m(x_pix, y_pix)
            self.best_frontier_artist = plt.Circle((x_m, y_m), radius=0.08,
                                                   color='blue', alpha=0.9, zorder=5)
            self.ax.add_patch(self.best_frontier_artist)

    def _pix_to_m(self, x_pix: float, y_pix: float) -> Tuple[float, float]:
        return x_pix * self.m_per_pix, y_pix * self.m_per_pix

    def _draw_car(self, x_m: float, y_m: float, theta_deg: float):
        # 1) 箭头（为简单起见仍然移除重画；也可改成复用：更新 FancyArrow 的 verts）
        if self.car_artist is not None:
            try: self.car_artist.remove()
            except Exception: pass
            self.car_artist = None

        theta_rad = math.radians(theta_deg)
        dx = self.ROBOT_SHAFT_LEN_M * math.cos(theta_rad)
        dy = self.ROBOT_SHAFT_LEN_M * math.sin(theta_rad)

        self.car_artist = self.ax.arrow(
            x_m, y_m, dx, dy,
            head_width=self.ROBOT_HEAD_W_M,
            head_length=self.ROBOT_HEAD_LEN_M,
            fc="r", ec="r", length_includes_head=True, zorder=5
        )

        # 2) 文字：复用句柄而不是每帧新增
        label = f"({x_m:.2f} m, {y_m:.2f} m), θ={theta_deg:.1f}°"
        if self.text_artist is None:
            self.text_artist = self.ax.text(
                x_m, y_m, label, fontsize=9, color="green",
                va="bottom", zorder=6, clip_on=True
            )
        else:
            self.text_artist.set_position((x_m, y_m))
            self.text_artist.set_text(label)

    def _update_traj(self, x_m: float, y_m: float):
        if not self.show_traj:
            return
        self.traj_xm.append(x_m); self.traj_ym.append(y_m)
        if self.traj_line is None:
            self.traj_line = mlines.Line2D(self.traj_xm, self.traj_ym,
                                           linewidth=1.0, color="b", zorder=3)
            self.ax.add_line(self.traj_line)
        else:
            self.traj_line.set_data(self.traj_xm, self.traj_ym)

    def _draw_exit_pose(self, exit_pose_pix: Optional[Tuple[int, int]]):
        """绘制预测的出口点，用绿色大圆圈标记"""

        # 清除旧的出口点标记
        if self.exit_pose_artist is not None:
            try:
                self.exit_pose_artist.remove()
            except:
                pass
            self.exit_pose_artist = None

        # 绘制新的出口点
        if exit_pose_pix is not None:
            x_pix, y_pix = exit_pose_pix
            x_m, y_m = self._pix_to_m(x_pix, y_pix)
            self.exit_pose_artist = plt.Circle((x_m, y_m), radius=0.12,
                                               color='green', alpha=0.9, zorder=6,
                                               linewidth=2, fill=False)  # 绿色圆环
            self.ax.add_patch(self.exit_pose_artist)

    def display(self, grid: List[List[int]],
                car_pose_pix_deg: Optional[Tuple[float, float, float]] = None,
                extra_text: Optional[str] = None,
                roi_rect=None,
                frontier_points: Optional[List[Tuple[int, int]]] = None,
                best_frontier_point: Optional[Tuple[int, int]] = None,
                spatially_sampled : Optional[List[Tuple[int, int]]] = None,
                exit_pose_pix: Optional[Tuple[int, int]] = None
                ) -> bool:
        # 地图
        if self.img_artist is None:
            self.img_artist = self.ax.imshow(
                grid, cmap=cm.viridis, vmin=0, vmax=255,
                origin=self.origin, extent=self.extent_m, zorder=1
            )
        else:
            self.img_artist.set_data(grid)

        # 小车与轨迹
        if car_pose_pix_deg is not None:
            x_pix, y_pix, theta_deg = car_pose_pix_deg
            x_m, y_m = self._pix_to_m(x_pix, y_pix)
            self._draw_car(x_m, y_m, theta_deg)
            self._update_traj(x_m, y_m)

            # 绘制前沿点
            #self._draw_frontier_points(frontier_points, best_frontier_point)
            self._draw_frontier_points_extra(spatially_sampled,best_frontier_point)
            self._draw_exit_pose(exit_pose_pix)

        if extra_text is not None and len(extra_text) > 0:
            # 选个固定位置（图像坐标左上角内缩一点）
            x0 = self.extent_m[0] + 0.05 * (self.extent_m[1] - self.extent_m[0])
            y1 = self.extent_m[3] - 0.05 * (self.extent_m[3] - self.extent_m[2])
            if self.hud_artist is None:
                self.hud_artist = self.ax.text(
                    x0, y1, extra_text, fontsize=9, color="yellow",
                    va="top", ha="left", zorder=10,
                    bbox=dict(facecolor="black", alpha=0.3, pad=3, edgecolor="none")
                )
            else:
                self.hud_artist.set_position((x0, y1))
                self.hud_artist.set_text(extra_text)
        elif self.hud_artist is not None:
            # 允许临时隐藏
            self.hud_artist.set_text("")

        # ---------- 可选：在 SLAM 栅格上叠加 maze_rect ----------
        # roi_rect: (x0, y0, x1, y1) in pixel coordinates
        if roi_rect is not None and isinstance(roi_rect, (tuple, list)) and len(roi_rect) == 4:
            x0_pix, y0_pix, x1_pix, y1_pix = roi_rect

            # 计算像素->米的换算
            # grid 是 [H][W]，extent 是 (xmin, xmax, ymin, ymax)（米）
            H = len(grid)
            W = len(grid[0]) if H else 0
            xmin, xmax, ymin, ymax = self.extent_m

            # x/y 每像素对应的米数
            sx = (xmax - xmin) / max(1, W)
            sy = (ymax - ymin) / max(1, H)

            # 注意 origin='lower' 时，像素 y=0 在下；'upper' 时在上，需要翻转
            if self.origin == 'lower':
                # 直接线性映射
                x_m = xmin + x0_pix * sx
                y_m = ymin + y0_pix * sy
                w_m = (x1_pix - x0_pix + 1) * sx
                h_m = (y1_pix - y0_pix + 1) * sy
            else:
                # 上原点：y 轴翻转
                x_m = xmin + x0_pix * sx
                # y=0 在上，对应 ymax；像素增长向下 => 减少米坐标
                y_m = ymax - (y1_pix + 1) * sy
                w_m = (x1_pix - x0_pix + 1) * sx
                h_m = (y1_pix - y0_pix + 1) * sy

            # 复用/创建矩形 artist（绿色虚线框，细线，半透明填充可关）
            if self._roi_artist is None:
                self._roi_artist = mpatches.Rectangle(
                    (x_m, y_m), w_m, h_m,
                    fill=False, linewidth=1.5, linestyle='--', edgecolor='lime', zorder=3, alpha=0.95
                )
                self.ax.add_patch(self._roi_artist)
            else:
                self._roi_artist.set_xy((x_m, y_m))
                self._roi_artist.set_width(w_m)
                self._roi_artist.set_height(h_m)
                self._roi_artist.set_visible(True)
        else:
            # 没有 ROI 就把旧的隐藏（避免残影）
            if self._roi_artist is not None:
                self._roi_artist.set_visible(False)

        try:
            self.fig.canvas.draw_idle()
            plt.pause(0.01)
            return True
        except Exception:
            return False


# 下面是一个最小示例（可删）
if __name__ == "__main__":
    N, M = 800, 32.0
    grid = [[200 for _ in range(N)] for _ in range(N)]
    for y in range(150, 650):
        for x in range(395, 405):
            grid[y][x] = 20

    drawer = Drawer(N, M, title="Demo", show_trajectory=True, origin_lower_left=True)
    x_pix, y_pix, theta = 100.0, 100.0, 45.0
    for _ in range(200):
        ok = drawer.display(grid, (x_pix, y_pix, theta))
        if not ok: break
        x_pix += 2.0; y_pix += 2.0
