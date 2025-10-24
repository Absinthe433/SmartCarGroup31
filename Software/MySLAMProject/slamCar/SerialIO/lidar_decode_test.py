# -*- coding: utf-8 -*-
"""
lidar_decode_test.py — 固件雷达数据解码与输出测试

功能：
  - 连接到 STM32 设备。
  - 持续接收上行数据帧 (UpFrame)。
  - 将每帧数据中的雷达点阵 (points) 解码出来。
  - 以 "角度: [val]°, 距离: [val] mm" 的格式清晰地打印每一个点。
  - 帮助验证数据链路和基础解码的正确性。

用法：
  python lidar_decode_test.py --port COM7 --baud 921600
  (请将 COM7 替换为您实际的蓝牙串口号)
"""

import sys
import time
import argparse
import logging
from typing import Optional

# 确保 btlink.py 和此脚本在同一个目录下
from btlink import BTLink, UpFrame

def print_lidar_points(frm: UpFrame):
    """格式化打印单个数据帧中的所有雷达点"""
    print(f"\n--- [ 新一帧雷达数据 | 时间戳: {frm.time_us} us | 点数: {len(frm.points)} ] ---")
    
    if not frm.points:
        print("  (该帧无雷达数据)")
        return
        
    # 为了屏幕输出不过于杂乱，可以选择只打印部分点，例如前20个
    # 如果想打印所有点，请将 max_points_to_print 设置为一个很大的数或者移除这个限制
    max_points_to_print = 20 
    
    print(f"  (仅显示前 {max_points_to_print} 个点)")

    for i, point in enumerate(frm.points):
        if i >= max_points_to_print:
            print("  ...")
            break
        
        # 核心输出：打印每个点的角度和距离
        print(f"  点 #{i+1:03d} | 角度: {point.angle_deg:8.3f}°  |  距离: {point.distance_mm:8.2f} mm  |  质量: {point.quality}")


def main():
    # --- 参数解析 ---
    parser = argparse.ArgumentParser(description="STM32 固件雷达数据解码测试")
    parser.add_argument("--port", type=str, required=True, help="蓝牙串口名，例如 COM7 或 /dev/ttyS0")
    parser.add_argument("--baud", type=int, default=921600, help="串口波特率")
    parser.add_argument("--log", type=str, default="INFO", help="日志等级: DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    # --- 日志配置 ---
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger("LidarTest")

    # --- 初始化 BTLink ---
    # 假设 btlink.py 中 BTLink 类的构造函数接受 logger 参数
    link = BTLink(port=args.port, baud=args.baud, logger=logging.getLogger("BTLink"))

    try:
        log.info(f"正在启动连接，目标串口: {args.port}, 波特率: {args.baud}...")
        link.start()
        log.info("连接成功，开始接收数据... (按 Ctrl+C 退出)")

        # --- 主循环：接收并处理数据 ---
        while True:
            # 等待并获取一个完整的数据帧，超时时间为1秒
            frame = link.get_frame(timeout=1.0)
            
            if frame:
                # 如果成功获取到数据帧，就调用打印函数
                print_lidar_points(frame)
            else:
                # 如果超时仍未收到数据，打印一条等待信息
                log.warning("在1秒内未收到完整数据帧，仍在等待...")

    except KeyboardInterrupt:
        log.info("\n检测到用户中断 (Ctrl+C)，正在关闭程序...")
    except Exception as e:
        log.error(f"发生致命错误: {e}")
        # 在实际使用中，可以根据错误类型决定是否需要重连
    finally:
        log.info("正在停止 BTLink 并关闭串口...")
        link.stop()
        print("程序已安全退出。")

if __name__ == "__main__":
    main()