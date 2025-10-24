# -*- coding: utf-8 -*-
"""
btlink.py — PC 上位机与 STM32（HC-04）蓝牙通信（最新版协议）

[MODIFIED] 增加了 CRC16 校验，以解决数据帧同步/解析错误问题。

下行（PC→STM32）命令帧：
  0xAA 0x55 | uint16 len=10 | uint16 cmd_id | float32 turn_rad | float32 distance_m | uint16 CRC16

上行（STM32→PC）周期数据帧：
  0x55 0xAA |
    uint32 time_us |
    uint16 cmd_id |
    uint8  status |
    int32  encoder_l |
    int32  encoder_r |
    float32 current_yaw |
    uint16 data_count |
    [ data_count * { uint8 Quality, uint16 angle_q6, uint16 distance_q2 } ] |
  (无CRC) <---- [最终修正] 确认固件没有发送CRC

角度解码：angle_deg = angle_q6 / 64.0
距离解码：distance_mm = distance_q2 / 4.0

提供：
- BTLink: 起停、收发、自动重连、取帧
- send_cmd(cmd_id, turn_rad, distance_m)
- get_frame()/get_latest_frame()
- to_slam_inputs(prev_state, frame, geom, max_range_mm, quality_min): 生成 (scan360_mm, (dxy_mm, dtheta_deg, dt_s))

依赖：pyserial
"""

from __future__ import annotations
import struct
import threading
import time
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple
from queue import Queue, Full, Empty
import numpy as np

import serial
import serial.tools.list_ports


# ================= 常量与工具 =================

# 帧头常量（方向不同，头字节顺序不同）
HEADER_DOWN = b'\xAA\x55'  # PC → STM32
HEADER_UP   = b'\x55\xAA'  # STM32 → PC

DEFAULT_BAUD = 921600
DEFAULT_TIMEOUT = 0.1       # 串口 read 超时(s)
RECONNECT_INTERVAL = 2.0    # 自动重连间隔(s)


# CRC16(Modbus) — init 0xFFFF, poly 0xA001
# (保留此函数，因为它仍被下行指令 send_cmd 使用)
def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if (crc & 1) != 0:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


# ================= 数据结构 =================

@dataclass
class WheelGeom:
    wheel_radius_m: float      # 轮半径 R（米）
    half_wheelbase_m: float    # 半轮距 B（米）
    ticks_per_rev: int         # 编码器 CPR

@dataclass
class LidarPoint:
    quality: int
    angle_deg: float
    distance_mm: float

@dataclass
class UpFrame:  # STM32 → PC
    time_us: int
    cmd_id: int
    status: int        # 0=完成, 1=转弯, 2=直行
    encoder_l: int
    encoder_r: int
    current_yaw: float
    points: List[LidarPoint]   # data_count 个点（原始角距）

# ================ BTLink 主类 =================

class BTLink:
    """
    管理串口：起停、收发、自动重连、解析上行帧、发送命令帧。
    """

    def __init__(self,
                 port: Optional[str],
                 baud: int = DEFAULT_BAUD,
                 timeout: float = DEFAULT_TIMEOUT,
                 auto_reconnect: bool = True,
                 max_queue: int = 1,
                 logger: Optional[logging.Logger] = None):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.auto_reconnect = auto_reconnect
        self.log = logger or logging.getLogger("BTLink")
        self._ser: Optional[serial.Serial] = None
        self._stop_evt = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None
        self._rx_buf = bytearray()
        self._queue: Queue[UpFrame] = Queue(maxsize=max_queue)

    # ---------- 生命周期 ----------
    def start(self):
        self._stop_evt.clear()
        self._open_serial()
        self._rx_thread = threading.Thread(target=self._rx_loop, name="BTLink-RX", daemon=True)
        self._rx_thread.start()
        self.log.info("BTLink started.")

    def stop(self):
        self._stop_evt.set()
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.5)
        self._close_serial()
        self.log.info("BTLink stopped.")

    # ---------- 下行发送（PC→STM32） ----------
    def send_cmd(self, cmd_id: int, turn_rad: float, distance_m: float):
        """
        发送命令帧：
          AA 55 | len=10 | cmd_id(u16) | turn_rad(f32) | distance_m(f32) | CRC16(Modbus)
        """
        payload = struct.pack('<Hff', int(cmd_id), float(turn_rad), float(distance_m))
        body_wo_crc = HEADER_DOWN + struct.pack('<H', 10) + payload
        crc = struct.pack('<H', crc16_modbus(body_wo_crc))
        frame = body_wo_crc + crc
        self._write(frame)
        self.log.debug(f"send_cmd: id={cmd_id} turn={turn_rad:.4f}rad dist={distance_m:.4f}m")

    # ---------- 上行取帧（STM32→PC） ----------
    def get_frame(self, timeout: Optional[float] = None) -> Optional[UpFrame]:
        """FIFO 取一帧（最旧帧）。"""
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def get_latest_frame(self) -> Optional[UpFrame]:
        """只取最新一帧（丢弃队列里较旧的）。"""
        try:
            frm = self._queue.get(timeout=0.0)
        except Empty:
            return None
        while True:
            try:
                frm = self._queue.get_nowait()
            except Empty:
                break
        return frm

    # ---------- 私有：串口 ----------
    def _open_serial(self):
        while not self._stop_evt.is_set():
            try:
                if self._ser and self._ser.is_open:
                    return
                if self.port is None:
                    selected_port = self._auto_pick_port()
                    if selected_port is None:
                        self.log.error("No suitable serial port found.")
                        if not self.auto_reconnect:
                            raise serial.SerialException("Auto-detection failed and auto-reconnect disabled.")
                        time.sleep(RECONNECT_INTERVAL)
                        continue
                    self.port = selected_port

                self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
                self.log.info(f"Serial opened: {self.port} @ {self.baud}")
                return
            except serial.SerialException as e:
                self.log.warning(f"Open serial failed on {self.port}: {e}")
                self._close_serial()
                self.port = None
                if not self.auto_reconnect:
                    raise
                time.sleep(RECONNECT_INTERVAL)
            except Exception as e:
                self.log.warning(f"Unexpected error opening serial port: {e}")
                self._close_serial()
                self.port = None
                if not self.auto_reconnect:
                    raise
                time.sleep(RECONNECT_INTERVAL)


    def _close_serial(self):
        try:
            if self._ser and self._ser.is_open:
                self._ser.close()
                self.log.info(f"Serial port {self.port} closed.")
        except Exception as e:
            self.log.error(f"Error closing serial port {self.port}: {e}")
        finally:
            self._ser = None

    def _write(self, data: bytes):
        try:
            if not self._ser or not self._ser.is_open:
                self.log.warning("Serial port not open, attempting to reopen...")
                self._open_serial()
                if not self._ser or not self._ser.is_open:
                     self.log.error("Failed to reopen serial port for writing.")
                     return

            bytes_written = self._ser.write(data)
            if bytes_written != len(data):
                 self.log.warning(f"Serial write incomplete: expected {len(data)}, wrote {bytes_written}")

        except serial.SerialTimeoutException:
             self.log.error("Serial write timed out.")
             if self.auto_reconnect:
                 self._close_serial()
                 self.port = None
        except serial.SerialException as e:
            self.log.error(f"Serial write failed: {e}")
            if self.auto_reconnect:
                self._close_serial()
                self.port = None
        except Exception as e:
             self.log.error(f"Unexpected error during serial write: {e}")
             if self.auto_reconnect:
                 self._close_serial()
                 self.port = None


    # ---------- RX 线程 ----------
    def _rx_loop(self):
        buf = self._rx_buf
        while not self._stop_evt.is_set():
            if not self._ser or not self._ser.is_open:
                if not self.auto_reconnect:
                    self.log.warning("Serial port closed and auto-reconnect disabled. Stopping RX loop.")
                    break
                self.log.info("Serial port closed, attempting to reopen...")
                self._open_serial()
                if not self._ser or not self._ser.is_open:
                    self.log.warning("Failed to reopen serial port, retrying later.")
                    time.sleep(RECONNECT_INTERVAL)
                    continue

            try:
                bytes_to_read = self._ser.in_waiting or 1
                chunk = self._ser.read(min(bytes_to_read, 1024))

                if chunk:
                    buf += chunk
                    while True:
                        parse_result = self._try_parse_upframe(buf)
                        if parse_result is None:
                            break

                        frm = parse_result
                        try:
                            self._queue.put_nowait(frm)
                        except Full:
                            try:
                                _ = self._queue.get_nowait()
                                self._queue.put_nowait(frm)
                            except Empty:
                                pass

                    max_buf_size = 8192
                    if len(buf) > max_buf_size:
                         self.log.warning(f"RX buffer size ({len(buf)}) exceeded limit ({max_buf_size}), discarding oldest {len(buf) - max_buf_size} bytes.")
                         del buf[:len(buf) - max_buf_size]

                else:
                    time.sleep(0.01)

            except serial.SerialException as e:
                self.log.error(f"Serial read error: {e}")
                if self.auto_reconnect:
                    self._close_serial()
                    self.port = None
                    time.sleep(RECONNECT_INTERVAL)
                else:
                    self.log.error("Auto-reconnect disabled. Stopping RX loop.")
                    break
            except Exception as e:
                 self.log.error(f"Unexpected error in RX loop: {e}", exc_info=True)
                 time.sleep(0.1)


    # ---------- [最终修正] 上行帧解析（移除CRC校验） ----------
    def _try_parse_upframe(self, buf: bytearray) -> Optional[UpFrame]:
        """
        从 buf 开头解析一帧 STM32→PC 上行数据帧（无CRC校验）：
          头 0x55 0xAA
          固定 21 字节 + 5*data_count
        成功则从 buf 移除该帧，并返回 UpFrame；否则返回 None。
        """
        # 1. 寻找帧头
        start = buf.find(HEADER_UP)
        if start == -1:
            if len(buf) > 1:
                del buf[:-1]
            return None
        
        # 2. 移除帧头前的所有无效数据
        if start > 0:
            self.log.debug(f"Discarding {start} bytes of junk data before header.")
            del buf[:start]

        # 3. 检查是否有足够的数据来读取固定长度部分 (头 + 固定字段 = 23字节)
        if len(buf) < 23:
            return None

        # 4. 解析固定字段以获取点云数量(data_count)
        try:
            data_count,  = struct.unpack_from('<H', buf, 21)
        except (struct.error, IndexError):
            self.log.warning("Failed to unpack data_count, header is likely corrupt. Discarding.")
            del buf[:2]
            return None

        # 5. 合理性检查
        max_expected_points = 600
        if data_count > max_expected_points:
            self.log.warning(f"Invalid data_count {data_count} > {max_expected_points}. Discarding header to re-sync.")
            del buf[:2]
            return None

        # 6. 计算不含CRC的完整帧长度，并检查数据是否已收全
        total_len_without_crc = 23 + data_count * 5
        if len(buf) < total_len_without_crc:
             return None

        # 7. [最终修正] 移除CRC校验逻辑

        # 8. 直接解析完整数据帧
        try:
            time_us, = struct.unpack_from('<I', buf, 2)
            cmd_id,  = struct.unpack_from('<H', buf, 6)
            status   = buf[8]
            encoder_l, = struct.unpack_from('<i', buf, 9)
            encoder_r, = struct.unpack_from('<i', buf, 13)
            current_yaw, = struct.unpack_from('<f', buf, 17)
            
            points: List[LidarPoint] = []
            off = 23
            for _ in range(data_count):
                q = buf[off]
                angle_q6, = struct.unpack_from('<H', buf, off + 1)
                dist_q2,  = struct.unpack_from('<H', buf, off + 3)
                off += 5
                
                quality = q >> 2
                angle_deg = angle_q6 / 64.0
                distance_mm = dist_q2 / 4.0
                points.append(LidarPoint(quality, angle_deg, distance_mm))
        
        except (struct.error, IndexError) as e:
            self.log.error(f"Parsing failed after header, data is corrupt: {e}. Discarding header.")
            del buf[:2]
            return None

        # 9. 解析成功，从缓冲区移除已处理的完整帧
        del buf[:total_len_without_crc]
        
        return UpFrame(
            time_us=time_us,
            cmd_id=cmd_id,
            status=status,
            encoder_l=encoder_l,
            encoder_r=encoder_r,
            current_yaw=current_yaw,
            points=points
        )


    @staticmethod
    def _auto_pick_port() -> Optional[str]:
        ports = list(serial.tools.list_ports.comports())
        selected_port = None
        for p in ports:
            name = (p.description or '') + ' ' + (p.device or '')
            if 'Bluetooth' in name or 'HC' in name or 'SPP' in name or 'BTHENUM' in p.hwid:
                selected_port = p.device
                logging.info(f"Auto-selected Bluetooth/SPP port: {selected_port}")
                return selected_port
        
        if ports:
             selected_port = ports[0].device
             logging.info(f"No specific Bluetooth port found, using first available: {selected_port}")
             return selected_port
        else:
             logging.warning("No serial ports detected.")
             return None



# ================= 辅助：上行帧 → SLAM 输入 =================

def ticks_to_motion(prev: Tuple[int, int, int],
                    curr: Tuple[int, int, int],
                    geom: WheelGeom) -> Tuple[float, float, float]:
    """
    由两帧 (cumL, cumR, ts_us) 计算 (dxy_mm, dtheta_deg, dt_s)
    """
    if geom.ticks_per_rev <= 0:
        logging.error("Invalid ticks_per_rev <= 0 in WheelGeom. Cannot calculate motion.")
        return 0.0, 0.0, 1e-6
    if geom.half_wheelbase_m <= 0:
        logging.error("Invalid half_wheelbase_m <= 0 in WheelGeom. Cannot calculate rotation.")
        dtheta_rad = 0.0
    
    cumL0, cumR0, ts0 = prev
    cumL1, cumR1, ts1 = curr

    if ts1 < ts0:
        if ts0 - ts1 > (0xFFFFFFFF // 2):
            dt_us = (0xFFFFFFFF - ts0) + ts1 + 1
        else:
            dt_us = 200000
            logging.warning(f"Timestamp decreased: {ts0} -> {ts1}. Assuming reorder, using fallback dt={dt_us/1e6}s.")
    else:
        dt_us = ts1 - ts0

    dt_s = max(dt_us / 1e6, 1e-6)

    dL = cumL1 - cumL0
    dR = cumR1 - cumR0

    revL = dL / float(geom.ticks_per_rev)
    revR = dR / float(geom.ticks_per_rev)
    dist_per_rev = 2.0 * np.pi * geom.wheel_radius_m
    sL = dist_per_rev * revL
    sR = dist_per_rev * revR
    
    ds = 0.5 * (sL + sR)

    if geom.half_wheelbase_m > 0:
        dtheta_rad = (sR - sL) / (2.0 * geom.half_wheelbase_m)
    else:
        dtheta_rad = 0.0

    dxy_mm = ds * 1000.0
    dtheta_deg = np.degrees(dtheta_rad)

    return (dxy_mm, dtheta_deg, dt_s)


def build_scan360(points: List[LidarPoint],
                  max_range_mm: int = 8000,
                  quality_min: int = 10) -> List[int]:
    """
    将 {angle_deg, distance_mm} 点集映射为 0..359 的距离数组（单位 mm）。
    """
    if not points:
        return [max_range_mm] * 360

    scan_raw = np.full(360, np.nan, dtype=float)
    valid_points_count = 0

    for p in points:
        if p.quality < quality_min or p.distance_mm <= 0 or p.distance_mm > max_range_mm:
            continue

        valid_points_count += 1
        angle = p.angle_deg % 360.0
        idx = int(np.round(angle)) % 360
        dist = p.distance_mm

        if np.isnan(scan_raw[idx]) or dist < scan_raw[idx]:
            scan_raw[idx] = dist

    if valid_points_count == 0:
        return [max_range_mm] * 360

    valid_indices = np.where(~np.isnan(scan_raw))[0]

    if len(valid_indices) < 2:
        final_scan_float = np.nan_to_num(scan_raw, nan=float(max_range_mm))
        final_scan = np.clip(np.round(final_scan_float), 0, max_range_mm).astype(int).tolist()
        return reorder_scan_ccw_minus180_to_plus179(final_scan)

    xp = valid_indices
    fp = scan_raw[valid_indices]
    x_interp = np.where(np.isnan(scan_raw))[0]

    if len(x_interp) > 0:
        interp_vals = np.interp(x_interp, xp, fp, period=360)
        scan_raw[x_interp] = interp_vals

    final_scan = np.clip(np.round(scan_raw), 0, max_range_mm).astype(int).tolist()

    return reorder_scan_ccw_minus180_to_plus179(final_scan)


def reorder_scan_ccw_minus180_to_plus179(scan: List[int]) -> List[int]:
    """
    将 [0..359] 顺时针扫描重排为 [-180..179] 逆时针扫描。
    """
    max_range_mm = 8000
    if len(scan) != 360:
         logging.error(f"reorder_scan input length is {len(scan)} not 360.")
         return [max_range_mm] * 360

    out = [max_range_mm] * 360
    try:
        out = scan[::-1]
        out = out[180:] + out[:180]
        
    except Exception as e:
         logging.error(f"Error during scan reordering: {e}", exc_info=True)
         return [max_range_mm] * 360
    return out


def to_slam_inputs(prev_state: Optional[Tuple[int, int, int]],
                   frame: UpFrame,
                   geom: WheelGeom,
                   max_range_mm: int = 8000,
                   quality_min: int = 10
                   ) -> Tuple[List[int], Tuple[float, float, float], Tuple[int, int, int]]:
    """
    把一帧上行数据转换为 SLAM 输入。
    """
    scan360 = build_scan360(frame.points, max_range_mm=max_range_mm, quality_min=quality_min)
    curr_state = (frame.encoder_l, frame.encoder_r, frame.time_us)

    if prev_state is None:
        velocities = (0.0, 0.0, 0.2)
    else:
        velocities = ticks_to_motion(prev_state, curr_state, geom)
        if velocities[2] <= 1e-6 or velocities[2] > 1.0:
             dt_fallback = 0.2
             logging.warning(f"Unreasonable dt calculated = {velocities[2]:.6f}s. Using fallback dt={dt_fallback}s.")
             velocities = (velocities[0], velocities[1], dt_fallback)

    return scan360, velocities, curr_state