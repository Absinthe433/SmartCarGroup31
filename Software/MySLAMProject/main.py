import threading
grid_MAP_SIZE_PIXELS = 800
grid_MAP_SIZE_METERS = 6

sim_MAP_SIZE_METERS = 2

SEED = 9999
from breezyslam.algorithms import  RMHC_SLAM
from breezyslam.sensors import Laser
from slamCar.navigation.navigator import *
from slamCar.navigation.config import *
from slamCar.SerialIO.btlink import BTLink, WheelGeom as BTWheels, to_slam_inputs, ticks_to_motion
import argparse
import logging
import math # <--- MODIFICATION 1: 导入 math 库

p = Pose(0,0,0)



class threadfunc(threading.Thread):
    def __init__(self, link:BTLink, bt_geom:BTWheels, slam:RMHC_SLAM, lgm, planner:state, pose_vec, mapbytes, max_range_mm, report_box:dict):
        super().__init__(daemon=True)
        self.link = link
        self.bt_geom = bt_geom
        self.slam = slam
        self.lgm = lgm
        self.planner = planner
        self.pose_vec = pose_vec
        self.mapbytes = mapbytes
        self.max_range = max_range_mm
        self._stop_evt = threading.Event()

        # 控制状态
        self.in_flight = False  # 是否已有命令在执行
        self.prev_state = None  # (cumL, cumR, ts_us)
        self.last_cmd_id = 0
        self.report = report_box
        
        # 规划器返回的 (turn, dist) 指令
        self.turn_deg_cmd = 0.0
        self.distance_m_cmd = 0.0
        self.cmd_step = 0 # 0=待规划, 1=待执行转弯, 2=待执行直行

    def stop(self):
        self._stop_evt.set()

    def run(self):
        while not self._stop_evt.is_set():
            # 1. 始终获取最新帧
            frm = self.link.get_frame(timeout=1.0)
            if not frm:
                continue

            # 2. 始终使用最新帧更新SLAM和位姿
            # (移除了 update_triger 和 begin 逻辑)
            scan360, velocities, self.prev_state = to_slam_inputs(
                self.prev_state, frm, self.bt_geom, 
                max_range_mm=self.max_range, quality_min=0
            )

            # 只有在小车实际移动时才更新SLAM (防止静止时漂移)
            # ( velocities[0] = dxy_mm, velocities[1] = dtheta_deg )
            if abs(velocities[0]) > 1e-6 or abs(velocities[1]) > 1e-6:
                self.slam.update(scan360, velocities)
                
            self.pose_vec[0], self.pose_vec[1], self.pose_vec[2] = self.slam.getpos()
            self.slam.getmap(self.mapbytes)
            self.lgm.UpdateMap(self.mapbytes)
            self.lgm.SetCarPose(self.pose_vec[0] / 1000., self.pose_vec[1] / 1000., self.pose_vec[2])

            p.x_m = self.pose_vec[0] / 1000.0
            p.y_m = self.pose_vec[1] / 1000.0
            p.theta_deg = self.pose_vec[2]

            # 3. 检查是否可以发送下一条命令
            # 只有当 (A)小车已停止 且 (B)上一条命令ID匹配时
            if frm.status == 0 and frm.cmd_id == self.last_cmd_id:
                
                # 如果命令正在执行中，标记为完成
                if self.in_flight:
                    self.in_flight = False

                # -----------------
                # 状态机逻辑
                # -----------------
                
                # 状态 0: 空闲，需要调用规划器
                if self.cmd_step == 0:
                    command = self.planner.plan_next(p)
                    self.turn_deg_cmd, self.distance_m_cmd, self.report["report"] = command
                    self.cmd_step = 1 # 进入状态1
                
                # 状态 1: 执行转弯
                elif self.cmd_step == 1:
                    self.last_cmd_id += 1
                    
                    # ==========================================================
                    # MODIFICATION 2: 修复单位错误 (角度 -> 弧度)
                    # ==========================================================
                    turn_rad_cmd = math.radians(self.turn_deg_cmd)
                    
                    self.link.send_cmd(self.last_cmd_id, turn_rad=turn_rad_cmd, distance_m=0)
                    
                    self.in_flight = True
                    self.cmd_step = 2 # 进入状态2
                
                # 状态 2: 执行直行
                elif self.cmd_step == 2:
                    self.last_cmd_id += 1
                    
                    self.link.send_cmd(self.last_cmd_id, turn_rad=0, distance_m=self.distance_m_cmd)
                    
                    self.in_flight = True
                    self.cmd_step = 0 # 回到状态0 (待规划)


parser = argparse.ArgumentParser(description="PC↔STM32")
parser.add_argument("--port", type=str, default="COM5", help="串口名，如 COM7 或 /dev/rfcomm0；留空自动探测")
parser.add_argument("--baud", type=int, default=921600, help="波特率，默认 115200")
# 以下是您确认的正确默认值
parser.add_argument("--wheel-radius", type=float, default=0.0325, help="轮半径(米)")
parser.add_argument("--half-wheelbase", type=float, default=0.0825, help="半轮距(米)")
parser.add_argument("--cpr", type=int, default=770, help="编码器CPR (基于默认值推测)")
parser.add_argument("--latest", action="store_true", help="只取最新帧（丢旧保新）")
parser.add_argument("--csv", type=str, default=None, help="将关键字段记录到此CSV文件")
parser.add_argument("--log", type=str, default="INFO", help="日志等级：DEBUG/INFO/WARNING/ERROR")
args = parser.parse_args()

def main():
    # geom (用于 SLAM 里程计) 将使用正确的默认值 (1540 / 0.0325 / 0.084)
    geom = BTWheels(args.wheel_radius, args.half_wheelbase, args.cpr)
    link = BTLink(port=args.port, baud=args.baud, logger=logging.getLogger("BTLink"))
    link.start()
    mapbytes = bytearray(grid_MAP_SIZE_PIXELS * grid_MAP_SIZE_PIXELS)
    laser = Laser(360,5.0,360.0,8000,0,0)
    slam = RMHC_SLAM(laser, grid_MAP_SIZE_PIXELS, grid_MAP_SIZE_METERS,hole_width_mm=100)
    lgm = ListGridMap(grid_MAP_SIZE_PIXELS, grid_MAP_SIZE_METERS)

    # Pose will be modified in our threaded code
    pose = [0, 0, 0]

    # ==========================================================
    # MODIFICATION 3: 确保 planner 内部参数与 SLAM/args 一致
    # ==========================================================
    wheel_geom = WheelGeom(
        wheel_radius_mm=args.wheel_radius * 1000,   # e.g., 32.5
        half_axle_len_mm=args.half_wheelbase * 1000,# e.g., 84
        ticks_per_cycle=args.cpr                    # e.g., 1540
    )
    # 使用您日志中的目标点
    GOAL_WORLD_COORDS = (2.5,2.5 ) 
    planner = state(lgm, wheel_geom, goal_world=GOAL_WORLD_COORDS, vel_limits=VelocityLimits(), tol=CmdTolerance())

    report = PlannerStateReport(
        phase=MissionPhase.EXPLORE,
        curr_pose=Pose(0.0, 0.0, 0.0),
        next_waypoint=None,
        frontier_count=0,
        known_maze_rect=None,
        has_found_exit=False,
        path_len_nodes=0,
        note="未初始化"
    )

    report_box = {"report": report}

    ctrl = threadfunc(link, geom, slam, lgm, planner, pose, mapbytes,max_range_mm=8000,report_box=report_box)
    ctrl.start()

    while True:
        # 读取最新 Planner 报告（若线程还未写入，用初值兜底）
        rep = report_box.get("report", None)

        # 1) SLAM 栅格与位姿（lgm 已被线程实时更新）
        slam_pose = lgm.CurrCarPose  # (x_pix, y_pix, theta_deg) or None
        hud = ""
        if rep is not None:
            hud = (
                f"Phase: {rep.phase.name}\n"
                f"Frontiers: {rep.frontier_count}\n"
                f"Known ROI: {rep.known_maze_rect}\n"
                f"Path nodes: {rep.path_len_nodes}\n"
                f"Found exit: {rep.has_found_exit}\n"
                f"Note: {rep.note}"
            )
        # 传入前沿点信息
        frontier_points = rep.frontier_points if rep is not None else None
        best_frontier = rep.best_frontier_point if rep is not None else None
        spacialty = rep.Spatially_sampled
        lgm.drawer.display(lgm.grid, slam_pose, extra_text=None,roi_rect=(rep.known_maze_rect if rep is not None else None),frontier_points=frontier_points,best_frontier_point=best_frontier,spatially_sampled=spacialty,exit_pose_pix=rep.exit_pose)  # <<< 传入 HUD

if __name__ == "__main__":
    main()