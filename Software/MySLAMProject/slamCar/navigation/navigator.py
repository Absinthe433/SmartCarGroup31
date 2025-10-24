from dataclasses import dataclass
from slamCar.Mymap.grid_map import ListGridMap
from slamCar.navigation.converter import *
from slamCar.navigation.mapRec import  *
from slamCar.navigation.frontierExp import *
from slamCar.navigation.goalCheck import *
from slamCar.navigation.pathPlanner import *
from slamCar.navigation.config import *

@dataclass
class PlannerStateReport:
    """供上层可视化/调试的状态报告"""
    phase: MissionPhase
    curr_pose: Pose
    next_waypoint: Optional[Pose]
    frontier_count: int
    known_maze_rect: Optional[Tuple[int, int, int, int]]
    has_found_exit: bool
    path_len_nodes: int
    note: str = ""
    # 新增字段
    frontier_points: List[Tuple[int, int]] = None  # 所有前沿点的像素坐标列表
    best_frontier_point: Optional[Tuple[int, int]] = None  # 当前选择的最优前沿点像素坐标
    Spatially_sampled : List[Tuple[int, int]] = None
    exit_pose : Tuple[int,int] = None


class state:
    def __init__(self,
                 gridmap: ListGridMap,
                 wheel_geom: WheelGeom,
                 vel_limits: VelocityLimits = VelocityLimits(),
                 tol: CmdTolerance = CmdTolerance(),
                 goal_world: Optional[Tuple[float, float]] = None,
                 ctol: CTolerance = CTolerance()):
        self.map = gridmap
        self.phase = MissionPhase.EXPLORE

        self.coord_converter = Converter(
            map_size_pixels=gridmap.map_size_pixels,
            map_size_meters=gridmap.map_scale_meters_per_pixel * gridmap.map_size_pixels
        )

        self.rectifier = MapRec()
        self.explorer = FrontierExplorer()
        self.goal_checker = GoalChecker()
        self.global_planner = PathPlanner()
        self.tol = tol

        self.goal_world_m = goal_world  # 终点世界坐标（米）
        self.goal_pix: Optional[Tuple[int, int]] = None  # 终点像素坐标

        if goal_world:
            # 转换终点坐标
            self.goal_pix = self.coord_converter.world_to_pixel(goal_world[0], goal_world[1])
            print(f"终点设置: 世界坐标{goal_world} -> 像素坐标{self.goal_pix}")


        self.start_pose_world: Optional[Pose] = None   # 起点（世界系）
        self.exit_pose_world: Optional[Pose] = None    # 终点（世界系）
        self.exit_pose_pix = None #终点（像素系系）
        self.goal_path_pix: List[Tuple[int,int]] = []  # 目前在执行的像素路径
        self.current_waypoint: Optional[Pose] = None   # 即将驱动到的世界系路标（用于产生编码器命令）
        self.best_frontier = [] #按照分数从高往低排列的前沿点列表
        self.goal_frontier = None #当前选择的最优前沿点

        # 统计/可视化
        self.frontier_cache: List[Tuple[int,int]] = []
        self.maze_rect: Optional[Tuple[int,int,int,int]] = None
        self.note: str = ""

        #原地摇摆建图
        self._wiggle_toggle = 2  # 在左/右之间切换
        self._wiggle_yaw_deg = 1  # 每次原地转动角度（可调 10~30）

        self.spatially_sampled = None
        self.repick = False
        self.ctol = ctol
    # ------------------ 生命周期入口 ------------------

    def set_start_pose(self, pose_world: Pose) -> None:
        """在比赛起点初始化时调用一次，记录起点。"""
        self.start_pose_world = pose_world

    def set_exit_pose(self, pose_world: Pose) -> None:
        """如果你有外部“终点”提示，也可以直接注入；否则由 ExitDetector 在探索中发现。"""
        self.exit_pose_world = pose_world

    def waypoint_to_turn_and_go(self,
                                curr_pose: Pose,
                                waypoint: Pose,
                                tol: CTolerance
                                ) -> Tuple[float, float]:

        def wrap_deg(a: float) -> float:
            # 归一化到 (-180, +180]
            w = (a + 180.0) % 360.0 - 180.0
            return 180.0 if w == -180.0 else w

        # 位置差与指向角
        dx = waypoint.x_m - curr_pose.x_m
        dy = waypoint.y_m - curr_pose.y_m
        dist = math.hypot(dx, dy)
        tgt_yaw = math.degrees(math.atan2(dy, dx))  # 面向路标的期望朝向（度，逆时针为正）

        cmds: List[Tuple[float, float]] = []

        if dist <= tol.pos_tol_m:
            # 已在路标邻域：可以只做角度微调（按需要保留/注释）
            d_yaw = wrap_deg(waypoint.theta_deg - curr_pose.theta_deg)
            if abs(d_yaw) < 3 :
                return 0 ,0
            else :
                return d_yaw, 0


        # 第一步：把车头转到指向路标（顺时针为负、逆时针为正）
        d_yaw1 = wrap_deg(tgt_yaw - curr_pose.theta_deg)
        if abs(d_yaw1) < 2:
            return 0, dist
        else:
            return d_yaw1, dist


    def swing_in_place(self) :

        # 可配置参数
        swing_angle_deg = 15.0  # 每次转动 ±15度
        distance_m = 0.05  # 原地旋转不直行

        # 如果还没有状态，就初始化一个标志
        if not hasattr(self, "_swing_dir"):
            self._swing_dir = 1  # 1 表示先左转（逆时针）

        # 根据方向生成指令
        turn_deg = swing_angle_deg * self._swing_dir

        # 翻转方向（下一次反向转）
        self._swing_dir *= -1


        return turn_deg, distance_m

    # ------------------ 规划主循环接口 ------------------

    def plan_next(self,
                  curr_pose_world: Pose) -> Tuple[float,float, PlannerStateReport]:
        self._update_derived_state()

        # 状态机切换 & 目标生成
        self._ensure_waypoint(curr_pose_world)

        commands = None
        #self.current_waypoint = Pose(curr_pose_world.x_m+0.05,curr_pose_world.y_m,curr_pose_world.theta_deg)
        if self.current_waypoint:
            commands = self.waypoint_to_turn_and_go(curr_pose_world,self.current_waypoint,self.ctol)

        else:

            # 没有路标 -> 做原地摆头，保证主循环有命令可执行、SLAM可持续更新
            commands = self.swing_in_place()

        # 生成报告
        report = PlannerStateReport(
            phase=self.phase,
            curr_pose=curr_pose_world,
            next_waypoint=self.current_waypoint,
            frontier_count=len(self.frontier_cache),
            known_maze_rect=self.maze_rect,
            has_found_exit=(self.exit_pose_world is not None),
            path_len_nodes=len(self.goal_path_pix),
            note=self.note,
            frontier_points=self.frontier_cache.copy(),
            best_frontier_point=self.goal_frontier,
            Spatially_sampled = self.explorer.SPATIALLY,
            exit_pose=self.goal_pix

        )
        turn_deg, distance_m = commands
        #print(turn_deg,distance_m, report)
        return turn_deg,distance_m, report

    def _bootstrap_scan(self,
                        curr_pose_world: Pose,
                        curr_cum_ticks: Tuple[int, int]) -> List[EncoderTarget]:
        """当当前无路标/无路径时：原地小幅左右摆头，多扫几帧让SLAM更快成图。"""
        yaw = self._wiggle_toggle * self._wiggle_yaw_deg
        self._wiggle_toggle *= 1  # 下次换方向
        # 目标姿态仅用于显示：位置不变，朝向临时加 yaw
        tgt = Pose(curr_pose_world.x_m, curr_pose_world.y_m,
                   curr_pose_world.theta_deg + yaw)
        cmd = self.motion.turn_in_place(curr_cum_ticks, yaw, tgt, self.tol)
        self.note = "Bootstrap scan: wiggle to enrich map"
        return [cmd]

    # ------------------ 内部：每次规划前刷新派生信息 ------------------

    def _update_derived_state(self) -> None:

        # 1) 更新“迷宫有效矩形”
        cx_pix,cy_pix,_ = self.map.GetCarPose()
        self.maze_rect = self.rectifier.update_from_grid(self.map.grid,
                                                         ensure_contains=(int(round(cx_pix)), int(round(cy_pix))))

        # ==========================================================
        # MODIFICATION 1: 修复因地图空白导致的 NoneType 崩溃
        # ==========================================================
        if self.maze_rect is None:
            # 如果 mapRec 未能找到矩形 (地图为空)，则使用全地图作为默认值
            H = len(self.map.grid)
            W = len(self.map.grid[0]) if H > 0 else 0
            if W > 0 and H > 0:
                self.maze_rect = (0, 0, W - 1, H - 1) # 默认使用全地图
            else:
                # 地图尚未初始化，无法继续
                self.frontier_cache = []
                self.note = "Map grid not initialized"
                return # 提前退出
        # ==========================================================


        # 2) 若在探索阶段，则提取前沿与检测出口
        if self.phase == MissionPhase.EXPLORE:
            self.frontier_cache = self.explorer.detect_frontiers(self.map.grid, self.maze_rect)
            self.note = "exploring the map......"

            # 【新逻辑】检查终点是否可达
            if self.goal_pix is not None:
                goal_reachable = self.goal_checker.is_goal_reachable(
                    self.map.grid,
                    (cx_pix, cy_pix),
                    self.goal_pix,
                    self.maze_rect
                )

                if goal_reachable:
                    # 终点可达，直接进入返回阶段

                    self.phase = MissionPhase.GO_TO_EXIT
                    self.goal_path_pix.clear()
                    self.current_waypoint = None
                    self.note = "Goal reachable -> GO_TO_EXIT"
                    return

            self.note = "Exploring towards goal..."

    # ------------------ 内部：选择/生成“下一个路标” ------------------

    def _ensure_waypoint(self, curr_pose_world: Pose) -> None:

        # 1) 若没有起点记录，则初始化
        if self.start_pose_world is None:
            self.start_pose_world = Pose(curr_pose_world.x_m, curr_pose_world.y_m, curr_pose_world.theta_deg)

        # 2) 若已有路标且仍未到达（上层通过重复调用 plan_next() 会微调），则保持不变
        if self.current_waypoint and not self._reached(curr_pose_world, self.current_waypoint, self.tol):

            return

        # 3) 生成/刷新路径与路标
        if self.phase == MissionPhase.EXPLORE:
            cx_pix,cy_pix,_= self.map.GetCarPose()
            # 若已到达最优前沿点附近或者path长度小于2 就重新找最优前沿点并规划路径
            if self._near_without_deg(cx_pix,cy_pix,self.goal_frontier) or len(self.goal_path_pix)<=3:

                self._gen_waypoint_explore(curr_pose_world)
            else:
                self.current_waypoint = self._pop_next_waypoint(curr_pose_world)

        elif self.phase == MissionPhase.RETURN_HOME:
            if self._plan_or_refresh_path_to_world(curr_pose_world, self.start_pose_world,False):
                self.current_waypoint = self._pop_next_waypoint(curr_pose_world)
            else:
                # 到家或路径空
                if self._near(curr_pose_world, self.start_pose_world, self.tol):
                    # 下一阶段：从起点 -> 终点
                    if self.exit_pose_world:
                        self.phase = MissionPhase.GO_TO_EXIT
                        self.goal_path_pix.clear()
                        self.current_waypoint = None
                        self.note = "At start -> GO_TO_EXIT"
                    else:
                        self.note = "Waiting for exit_pose_world"
                else:
                    self.note = "No path to home (check map/inflation)"

        elif self.phase == MissionPhase.GO_TO_EXIT:
            # 从“起点”发起最短路（你也可以选择从“当前位置”直接到exit再回到起点后走最短路）
            goal_world_pose = Pose(self.goal_world_m[0],self.goal_world_m[1],0)
            if self._plan_or_refresh_path_to_world(curr_pose_world, goal_world_pose,True):
                self.current_waypoint = self._pop_next_waypoint(curr_pose_world)
            else:
                # 若路径已全部消耗/到终点
                cx_pix, cy_pix,_ = self.map.GetCarPose()

                if self._near_without_deg(cx_pix, cy_pix, self.goal_pix):
                    self.phase = MissionPhase.RETURN_HOME
                    self.current_waypoint = None
                    self.note = "Return home"

        elif self.phase == MissionPhase.DONE:
            self.current_waypoint = None
            self.note = "All done."

    # ------------------ 内部：阶段细节 ------------------

    def _gen_waypoint_explore(self, curr_pose_world: Pose) -> None:
        """挑选前沿并规划到前沿，再从路径中取一小段转为waypoint。"""
        if not self.frontier_cache:
            self.note = "No frontier candidates"

            self.current_waypoint = None
            self.best_frontier = []
            return
        if len(self.frontier_cache)==0:

            self.best_frontier = []
        # 当前像素
        cx_pix, cy_pix, _ = self.map.GetCarPose()
        print(f"current position{cx_pix},{cy_pix}")
        if len(self.best_frontier)<1:
            self.repick = False

        if self.repick == False:
            #重新生成一批前沿点按照分数从大到小的排序
            goal_pix_s  = self.explorer.pick_goal((cx_pix, cy_pix), self.frontier_cache,self.map.grid,goal_pix=self.goal_pix,maze_rect=self.maze_rect)
            self.best_frontier = goal_pix_s
            if len(goal_pix_s)<=0:
                return
            goal_pix = self.best_frontier.pop(0)
            self.goal_frontier = goal_pix
            self.repick = True
        else:
            goal_pix = self.best_frontier.pop(0)
            self.goal_frontier = goal_pix

        if goal_pix is None:
            self.note = "No valid frontier after filtering"
            self.current_waypoint = None
            return

        # A* 到前沿
        self.goal_path_pix = self.global_planner.plan_path(self.map.grid,
                                                           (int(round(cx_pix)), int(round(cy_pix))),
                                                           goal_pix, self.maze_rect)

        if len(self.goal_path_pix) < 2:
            self.note = "Path to frontier empty"
            self.current_waypoint = None
            self.repick = True
            return
        else:
            self.repick = False

        # ==========================================================
        # MODIFICATION 2: 修复 IndexError (goal_path_pix[3])
        # ==========================================================
        # 取“下一跳”为局部waypoint
        
        # 1. 路径最短也有2个点 (e.g., [start, end])
        # 2. 我们取索引 [1] 作为下一跳 (与 _pop_next_waypoint 一致)
        # 3. 原来的 [3] 太大了，会导致 Index Error
        
        if len(self.goal_path_pix) >= 2:
            self.current_waypoint = self._pixnode_to_local_waypoint(curr_pose_world, self.goal_path_pix[1])
        else:
            # 理论上不会到这里，但作为保险
            self.current_waypoint = None
            self.repick = True
        # ==========================================================


    def _plan_or_refresh_path_to_world(self, curr_pose_world: Pose, goal_world: Pose, go_to_exiit:bool) -> bool:
        """从当前像素到世界坐标goal的像素点规划；有路径则缓存。"""
        cx_pix, cy_pix, _ = self.map.GetCarPose()
        if go_to_exiit:
            gx_pix, gy_pix = self.coord_converter.world_to_pixel(goal_world.x_m,goal_world.y_m)
        else:
            gx_pix, gy_pix = self.map.m2pix(goal_world.x_m,goal_world.y_m)

        self.goal_path_pix = self.global_planner.plan_path(self.map.grid,
                                                           (int(round(cx_pix)), int(round(cy_pix))),
                                                           (int(round(gx_pix)), int(round(gy_pix))),
                                                           self.maze_rect)

        return len(self.goal_path_pix) >= 3

    def _pop_next_waypoint(self, curr_pose_world: Pose) -> Optional[Pose]:

        if len(self.goal_path_pix) < 2:
            return None

        nxt = self.goal_path_pix.pop(1)

        return self._pixnode_to_local_waypoint(curr_pose_world, nxt)

    def _pixnode_to_local_waypoint(self, curr_pose_world: Pose, node_pix: Tuple[int,int]) -> Pose:
        """像素节点 -> 世界路标；航向指向该节点。"""
        x_m, y_m = self.map.pix2m(node_pix[0], node_pix[1])
        dx, dy = x_m - curr_pose_world.x_m, y_m - curr_pose_world.y_m
        yaw = math.degrees(math.atan2(dy, dx))

        return Pose(x_m, y_m, yaw)

    # ------------------ 内部：容差判断 ------------------

    @staticmethod
    def _near(a: Pose, b: Pose, tol: CmdTolerance) -> bool:
        return (math.hypot(a.x_m - b.x_m, a.y_m - b.y_m) <= tol.pos_tol_m)

    @staticmethod
    def _reached(curr: Pose, goal: Pose, tol: CmdTolerance) -> bool:

        return state._near(curr, goal, tol)

    @staticmethod
    def _near_without_deg(cur_x_pix,cur_y_pix,best_frontier,max_distance=20):
        if best_frontier is None:
            return False
        goal_x_pix = best_frontier[0]
        goal_y_pix = best_frontier[1]
        distance = math.sqrt((cur_x_pix-goal_x_pix)**2+(cur_y_pix-goal_y_pix)**2)
        return distance<max_distance