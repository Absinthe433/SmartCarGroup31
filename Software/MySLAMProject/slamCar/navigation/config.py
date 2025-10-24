from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Optional, Iterable, Dict, Any,Set

class MissionPhase(Enum):
    """高层任务阶段"""
    EXPLORE = auto()       # 自主探索（基于frontier等）
    RETURN_HOME = auto()   # 从任意处回到起点（全局最短路）
    GO_TO_EXIT = auto()    # 从起点沿最短路到终点（比赛需要的最终阶段）
    DONE = auto()          # 任务完成

@dataclass
class Pose:
    """世界系（米/度）位姿，theta度，逆时针为正。"""
    x_m: float
    y_m: float
    theta_deg: float

@dataclass
class WheelGeom:
    """用于里程与编码器换算的底盘标定参数"""
    wheel_radius_mm: float            # 轮半径
    half_axle_len_mm: float           # 半轮距（半轴长）
    ticks_per_cycle: int              # 每圈编码器脉冲数（例如 Rover.ticks_per_cycle=2000）

@dataclass
class CmdTolerance:
    """到达目标点/角的误差容许"""
    pos_tol_m: float = 0.1          # 位置容差
    yaw_tol_deg: float = 20          # 航向角容差

@dataclass
class CTolerance:
    """到达目标点/角的误差容许"""
    pos_tol_m: float = 0.07          # 位置容差
    yaw_tol_deg: float = 40          # 航向角容差


@dataclass
class VelocityLimits:
    """速度/时间限制，用于把目标位姿离散成若干段命令"""
    max_lin_mps: float = 0.25
    max_ang_dps: float = 90.0
    min_segment_time_s: float = 0.2
    max_segment_time_s: float = 1.0

@dataclass
class EncoderTarget:
    """
    一条可下发的控制指令：指定“目标时刻的左右轮**累计**计数”+“应在本段内耗时多久”。
    上层应将这条指令发给底层（或仿真），在该时长内驱动左右轮“累计值”达到此计数。
    """
    cum_left_ticks: int
    cum_right_ticks: int
    duration_ms: int
    target_waypoint: Pose              # 期望抵达的“局部目标”位姿（用于调试/显示）
    tolerance: CmdTolerance


FREE_thresh = 254
OBSTACLE_val = 40
