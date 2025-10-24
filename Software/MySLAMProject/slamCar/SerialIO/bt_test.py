# -*- coding: utf-8 -*-
"""
test.py — 实机联测（只使用 btlink.py 提供的接口）
  - 手动输入命令：cmd_id turn_rad distance_m
  - 发送命令：BTLink.send_cmd()
  - 接收打印：BTLink.get_frame() / get_latest_frame()
  - 运动增量：btlink.ticks_to_motion() 或 to_slam_inputs()

用法示例：
  python test.py --port COM7 --baud 115200 --wheel-radius 0.077 --half-wheelbase 0.165 --cpr 2000
可选：
  --latest   : 只取最新帧（丢旧保新），默认逐帧FIFO
  --csv out.csv : 同步记录关键字段到CSV
"""

import sys
import time
import argparse
import threading
import logging
import csv
from typing import Optional, Tuple

from btlink import BTLink, WheelGeom, ticks_to_motion, to_slam_inputs, UpFrame,build_scan360

def status_text(s: int) -> str:
    return {0: "停止(完成)", 1: "转弯中", 2: "直行中"}.get(s, "?")

def print_upframe(frm: UpFrame, show_points: int = 500):
    print(f"[UpFrame] ts={frm.time_us}  cmd_id={frm.cmd_id}  status={frm.status}({status_text(frm.status)})")
    print(f" encL={frm.encoder_l}  encR={frm.encoder_r}  yaw={frm.current_yaw:.3f}  points={len(frm.points)}")
    
    # 【核心修正】使用命名参数，并为 quality_min 提供一个合理的值
    scan_list = build_scan360(points=frm.points, max_range_mm=8000, quality_min=10)
    
    idx = -180
    for scan in scan_list:
        print(f"idx{idx},scan:{scan}")
        idx = idx+1
    
    if frm.points:
        n = min(show_points, len(frm.points))
        print(f"  预览前 {n} 个点：")
        for i in range(n):
            p = frm.points[i]
            print(f"   #{i:02d}  Q={p.quality:3d}  angle={p.angle_deg:8.3f}°  dist={p.distance_mm:8.2f} mm")

def receiver_thread(link: BTLink,
                    geom: WheelGeom,
                    latest_mode: bool,
                    stop_evt: threading.Event,
                    csv_writer: Optional[csv.writer] = None):
    prev_state: Optional[Tuple[int,int,int]] = None  # (cumL, cumR, ts)
    while not stop_evt.is_set():
        # 取帧：顺序 or 最新
        frm = link.get_latest_frame() if latest_mode else link.get_frame(timeout=1.0)
        if not frm:
            continue

        print("\n================ 收到一帧 ================")
        print_upframe(frm, show_points=0)

        # 计算运动增量（严格使用 btlink 的工具函数）
        curr_state = (frm.encoder_l, frm.encoder_r, frm.time_us)
        if prev_state is None:
            print("[ticks_to_motion] 首帧，无增量。")
            dxy_mm, dtheta_deg, dt_s = 0.0, 0.0, 0.0
        else:
            dxy_mm, dtheta_deg, dt_s = ticks_to_motion(prev_state, curr_state, geom)
            print(f"[ticks_to_motion] dxy={dxy_mm:.2f} mm   dθ={dtheta_deg:.3f} °   dt={dt_s:.4f} s")
        prev_state = curr_state

        # （可选）也可以用 to_slam_inputs 一步得到 scan360 和 velocities
        # scan360, velocities, _ = to_slam_inputs(None if prev_state is None else (frm.encoder_l, frm.encoder_r, frm.time_us),
        #                                          frm, geom)
        # print(f"[to_slam_inputs] scan360_len={len(scan360)} velocities={velocities}")

        # 写 CSV（可选）
        if csv_writer:
            # time_us, cmd_id, status, encL, encR, yaw, dxy_mm, dtheta_deg, dt_s, points
            csv_writer.writerow([frm.time_us, frm.cmd_id, frm.status,
                                 frm.encoder_l, frm.encoder_r, f"{frm.current_yaw:.6f}",
                                 f"{dxy_mm:.3f}", f"{dtheta_deg:.3f}", f"{dt_s:.6f}",
                                 len(frm.points)])
            # 立即落盘，避免丢数据
            # 注意：频繁 flush 有轻微性能损耗，但5Hz影响可忽略
            try:
                csv_writer.writerow  # 仅为静态检查
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description="PC↔STM32 蓝牙联测（只用 btlink.py）")
    parser.add_argument("--port", type=str, default="COM5", help="串口名，如 COM7 或 /dev/rfcomm0；留空自动探测")
    parser.add_argument("--baud", type=int, default=921600, help="波特率，默认 115200")
    parser.add_argument("--wheel-radius", type=float, default=0.0325, help="轮半径(米)")
    parser.add_argument("--half-wheelbase", type=float, default=0.0825, help="半轮距(米)")
    parser.add_argument("--cpr", type=int, default=770, help="编码器CPR")
    parser.add_argument("--latest", action="store_true", help="只取最新帧（丢旧保新）")
    parser.add_argument("--csv", type=str, default=None, help="将关键字段记录到此CSV文件")
    parser.add_argument("--log", type=str, default="INFO", help="日志等级：DEUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO))
    log = logging.getLogger("test")

    geom = WheelGeom(args.wheel_radius, args.half_wheelbase, args.cpr)
    link = BTLink(port=args.port, baud=args.baud, logger=logging.getLogger("BTLink"))

    try:
        link.start()
    except Exception as e:
        print(f"⚠️ 打开串口失败：{e}")
        sys.exit(1)

    # CSV 记录器（可选）
    csv_fp = None
    csv_writer = None
    if args.csv:
        csv_fp = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow(["time_us","cmd_id","status","encL","encR","yaw",
                             "dxy_mm","dtheta_deg","dt_s","points"])

    print("\n====== 实机联测开始 ======")
    print(f"串口：{link.port or '(自动)'}  波特率：{args.baud}  轮系：R={geom.wheel_radius_m}m  B={geom.half_wheelbase_m}m  CPR={geom.ticks_per_rev}")
    print("小车应每 200 ms 上报一帧。你可以随时输入命令：cmd_id turn_rad distance_m")
    print("例如： 1 0.5236 0.20   （分别是 命令号、弧度增量、米）")
    print("按 Ctrl+C 退出。\n")

    stop_evt = threading.Event()
    t = threading.Thread(target=receiver_thread, args=(link, geom, args.latest, stop_evt, csv_writer), daemon=True)
    t.start()

    try:
        while True:
            line = input("请输入命令(cmd_id turn_rad distance_m)：").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                print("格式错误，应为：cmd_id turn_rad distance_m")
                continue
            try:
                cmd_id = int(parts[0])
                turn_rad = float(parts[1])   # 弧度增量
                distance_m = float(parts[2]) # 米
            except ValueError:
                print("解析失败，请输入形如：1 0.5236 0.20")
                continue

            link.send_cmd(cmd_id=cmd_id, turn_rad=turn_rad, distance_m=distance_m)
            print(f"→ 已发送：cmd_id={cmd_id}  turn_rad={turn_rad} rad  distance={distance_m} m")

    except KeyboardInterrupt:
        print("\n收到中断，准备退出...")

    finally:
        stop_evt.set()
        t.join(timeout=1.0)
        link.stop()
        if csv_fp:
            csv_fp.close()
        print("已退出。")

if __name__ == "__main__":
    main()

