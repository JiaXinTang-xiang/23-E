# -*- coding: utf-8 -*-
"""
23E题 - 红色激光运动目标控制系统 视觉功能模块
硬件: Jetson Orin NX + 海康/USB相机
"""
import sys, time, cv2, numpy as np, os, struct
from enum import Enum, auto

IS_JETSON = os.path.exists("/etc/nv_tegra_release")
SDK_OK = False
try:
    if IS_JETSON:
        sdk_paths = [
            "/opt/MVS/Samples/aarch64/Python/MvImport",
            "/usr/local/MVS/Samples/aarch64/Python/MvImport",
            "/home/nvidia/MVS/Samples/aarch64/Python/MvImport"
        ]
    else:
        sdk_paths = [
            "/opt/MVS/Samples/Python/MvImport",
            "/usr/local/MVS/Samples/Python/MvImport"
        ]
    for path in sdk_paths:
        if os.path.exists(path):
            sys.path.append(path)
            break
    else:
        sys.path.append("./MvImport")
    from MvCameraControl_class import *
    SDK_OK = True
except ImportError:
    pass

# Camera params
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 60
CAMERA_EXPOSURE_US = 3000
CAMERA_GAIN   = 8.0

# Red laser HSV ranges (red wraps around 0 in HSV)
RED_LOWER_1 = np.array([0,   120, 180])
RED_UPPER_1 = np.array([10,  255, 255])
RED_LOWER_2 = np.array([170, 120, 180])
RED_UPPER_2 = np.array([179, 255, 255])

RED_SPOT_MIN_AREA = 5       # minimum spot area (pixels)
RED_SPOT_MAX_AREA = 500     # maximum spot area (pixels)

# Screen calibration
SCREEN_SIZE_MM = 500.0      # 0.5m square side length
SCREEN_CORNERS_WORLD = np.array([
    [-250.0, -250.0], [250.0, -250.0],
    [250.0,  250.0], [-250.0,  250.0]
], dtype=np.float32)

# =============================================================================
# 相机管理类 - 支持海康SDK和USB摄像头
# =============================================================================
class CameraManager:
    """相机管理器, 自动选择海康工业相机或USB摄像头"""
    
    def __init__(self):
        self.cam = None
        self.data_buf = None
        self.payload_size = 0
        self.connected = False
        self.device_list = None
        self.use_hikvision = False
        self.usb_cap = None      # OpenCV VideoCapture backup
    
    def init(self):
        """初始化相机"""
        if SDK_OK:
            if self._init_hikvision():
                self.use_hikvision = True
                return True
        
        # Fallback to USB camera
        print("[Camera] 尝试USB摄像头...")
        return self._init_usb()
    
    def _init_hikvision(self):
        """初始化海康工业相机"""
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(
            MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        if ret != 0 or device_list.nDeviceNum == 0:
            print("[Camera] 未找到海康设备")
            return False
        
        print(f"[Camera] 找到 {device_list.nDeviceNum} 个海康设备")
        self.device_list = device_list
        
        self.cam = MvCamera()
        device_info = cast(device_list.pDeviceInfo[0],
                          POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(device_info)
        if ret != 0:
            return False
        
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Monitor, 0)
            if ret != 0:
                return False
        
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        
        param = MVCC_INTVALUE()
        memset(byref(param), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", param)
        self.payload_size = param.nCurValue if ret == 0 else CAMERA_WIDTH * CAMERA_HEIGHT * 3
        
        self.connected = True
        print("[Camera] 海康相机初始化成功")
        return True
    
    def _init_usb(self):
        """初始化USB摄像头"""
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                self.usb_cap = cap
                self.connected = True
                print(f"[Camera] USB摄像头 /dev/video{idx} 初始化成功")
                return True
            cap.release()
        print("[Camera] 未找到USB摄像头")
        return False
    
    def set_exposure(self, exposure_us):
        if self.use_hikvision and self.cam:
            self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us))
    
    def set_gain(self, gain):
        if self.use_hikvision and self.cam:
            self.cam.MV_CC_SetFloatValue("Gain", float(gain))
    
    def start_grabbing(self):
        if self.use_hikvision and self.cam:
            return self.cam.MV_CC_StartGrabbing() == 0
        return self.connected
    
    def get_frame(self):
        """获取一帧图像, 返回BGR numpy数组或None"""
        if self.use_hikvision:
            return self._get_hikvision_frame()
        else:
            return self._get_usb_frame()
    
    def _get_hikvision_frame(self):
        if not self.connected:
            return None
        if self.data_buf is None:
            self.data_buf = (c_ubyte * self.payload_size)()
        
        frame_info = MV_FRAME_OUT_INFO_EX()
        memset(byref(frame_info), 0, sizeof(frame_info))
        ret = self.cam.MV_CC_GetOneFrameTimeout(
            byref(self.data_buf), self.payload_size, frame_info, 2000)
        if ret != 0:
            return None
        
        image_data = np.frombuffer(self.data_buf,
                                   count=int(frame_info.nFrameLen),
                                   dtype=np.uint8)
        if len(image_data) == 0:
            return None
        
        bayer = image_data.reshape((frame_info.nHeight, frame_info.nWidth))
        return cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)
    
    def _get_usb_frame(self):
        if self.usb_cap is None:
            return None
        ret, frame = self.usb_cap.read()
        return frame if ret else None
    
    def stop_grabbing(self):
        if self.use_hikvision and self.cam:
            self.cam.MV_CC_StopGrabbing()
    
    def release(self):
        self.stop_grabbing()
        if self.use_hikvision and self.cam:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.connected = False
        if self.usb_cap:
            self.usb_cap.release()
            self.usb_cap = None
        print("[Camera] 相机已释放")


# =============================================================================
# 红色光斑检测
# =============================================================================
def detect_red_spot(image, debug=False):
    """
    检测图像中的红色激光光斑中心, 返回 (cx, cy, area) 或 (None, None, 0)
    
    算法: HSV双范围掩膜 + 形态学开运算去噪 + 轮廓分析取最大
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 红色跨0度, 双范围合并
    mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    mask  = cv2.bitwise_or(mask1, mask2)
    
    # 形态学开运算 - 去除小噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0, mask
    
    # 按面积排序, 取面积在范围内的最大轮廓
    best_cx, best_cy, best_area = None, None, 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if RED_SPOT_MIN_AREA <= area <= RED_SPOT_MAX_AREA:
            if area > best_area:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    best_cx   = M["m10"] / M["m00"]
                    best_cy   = M["m01"] / M["m00"]
                    best_area = area
    
    return best_cx, best_cy, best_area, mask


# =============================================================================
# 屏幕标定 - 检测0.5m正方形边框, 建立像素-世界坐标映射
# =============================================================================
def find_screen_corners(image, binary_threshold=80):
    """
    检测屏幕上0.5m正方形边框的四个角点 (用铅笔画的线)
    返回排序后的角点 [(x,y),...] 顺序: TL, TR, BR, BL  或 None
    
    策略: 灰度 + 自适应二值化 + 轮廓筛选 + 最大四边形
    """
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, binary_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 找轮廓
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # 按面积排序, 取最大轮廓 (应该是正方形边框)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for cnt in contours[:5]:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            # 面积应该合理 (边框不会太小)
            if area > 5000:
                pts = approx.reshape(4, 2).astype(np.float32)
                pts = _sort_corners(pts)  # TL, TR, BR, BL
                return pts, bw
    
    return None, bw


def _sort_corners(pts):
    """角点排序: 左上-右上-右下-左下"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL - sum最小
    rect[2] = pts[np.argmax(s)]  # BR - sum最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR - diff最小
    rect[3] = pts[np.argmax(diff)]  # BL - diff最大
    return rect


def compute_screen_homography(screen_corners_px,
                               screen_size_mm=SCREEN_SIZE_MM):
    """
    计算屏幕像素坐标 → 物理世界坐标(mm)的单应矩阵
    
    screen_corners_px: 屏幕四角的像素坐标 [TL, TR, BR, BL]
    世界坐标原点: 屏幕中心, X右 Y下 (图像坐标系方向)
    
    返回: H (3x3单应矩阵), H_inv (逆矩阵)
    """
    half = screen_size_mm / 2.0
    world_corners = np.array([
        [-half, -half],   # TL
        [ half, -half],   # TR
        [ half,  half],   # BR
        [-half,  half],   # BL
    ], dtype=np.float32)
    
    src = screen_corners_px.reshape(4, 2).astype(np.float32)
    dst = world_corners
    
    H = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    return H, H_inv


def pixel_to_world(px, H):
    """像素坐标 → 世界坐标(mm)"""
    p = np.array([px[0], px[1], 1.0])
    w = H @ p
    return (w[0] / w[2], w[1] / w[2])


def world_to_pixel(world_xy, H_inv):
    """世界坐标(mm) → 像素坐标"""
    w = np.array([world_xy[0], world_xy[1], 1.0])
    p = H_inv @ w
    return (int(p[0] / p[2]), int(p[1] / p[2]))


# =============================================================================
# 路径生成器
# =============================================================================
class PathGenerator:
    """生成竞赛所需的运动轨迹航点序列"""
    
    def __init__(self, screen_size_mm=500.0):
        self.screen_half = screen_size_mm / 2.0
        self.a4_w = 210.0  # A4宽度 mm
        self.a4_h = 297.0  # A4高度 mm
        self.tape_w = 18.0 # 电工胶带宽度 mm
    
    def generate_square_border(self, num_pts_per_side=50):
        """
        生成0.5m正方形边框的航点 (顺时针)
        
        边框路径: 左上角→右上角→右下角→左下角→左上角
        光斑中心距边线≤2cm, 即沿边框内侧走
        """
        half = self.screen_half
        offset = 5.0  # 略偏内侧, 用视觉反馈修正
        
        # 四个角点 (顺时针): TL→TR→BR→BL→TL
        corners = np.array([
            [-half + offset, -half + offset],  # TL
            [ half - offset, -half + offset],  # TR
            [ half - offset,  half - offset],  # BR
            [-half + offset,  half - offset],  # BL
            [-half + offset, -half + offset],  # TL (闭合)
        ])
        
        waypoints = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[i + 1]
            for j in range(num_pts_per_side):
                t = j / num_pts_per_side
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                waypoints.append((x, y))
        
        return waypoints
    
    def generate_a4_border(self, a4_center_mm, a4_angle_deg=0.0,
                           num_pts_per_side=40):
        """
        生成A4靶纸胶带边框的航点 (顺时针)
        
        a4_center_mm: A4纸中心在世界坐标的位置 (cx, cy) mm
        a4_angle_deg: A4纸旋转角度 (度)
        """
        hw = self.a4_w / 2.0
        hh = self.a4_h / 2.0
        
        # A4纸四角 (未旋转, 中心在原点)
        local_corners = np.array([
            [-hw, -hh],  # TL
            [ hw, -hh],  # TR
            [ hw,  hh],  # BR
            [-hw,  hh],  # BL
            [-hw, -hh],  # TL (闭合)
        ])
        
        # 旋转
        rad  = np.radians(a4_angle_deg)
        cosA = np.cos(rad)
        sinA = np.sin(rad)
        R = np.array([[cosA, -sinA], [sinA, cosA]])
        
        rotated = (R @ local_corners.T).T  # (5, 2)
        
        # 平移到指定中心
        corners_world = rotated + np.array(a4_center_mm)
        
        # 胶带中心路径 (略偏内侧)
        waypoints = []
        for i in range(4):
            p1 = corners_world[i]
            p2 = corners_world[i + 1]
            for j in range(num_pts_per_side):
                t = j / num_pts_per_side
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                waypoints.append((x, y))
        
        return waypoints


# =============================================================================
# 串口通信
# =============================================================================
class SerialComm:
    """与MCU的串口通信"""
    
    def __init__(self, port="/dev/ttyTHS1", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.enabled = False
    
    def init(self):
        try:
            import serial
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
            self.enabled = True
            print(f"[Serial] {self.port} @ {self.baudrate} 初始化成功")
            return True
        except Exception as e:
            print(f"[Serial] 初始化失败: {e}, 将以无串口模式运行")
            self.enabled = False
            return False
    
    def send_angle_command(self, pan_angle, tilt_angle, laser_on=True):
        """
        发送云台角度指令
        
        协议帧格式 (参考原代码):
            帧头 0xA5 (1B)
            长度       (1B) = 10 (2+2+2+4)
            命令ID     (2B) = 0x0200 (红色云台控制)
            标志位     (2B) = laser_on
            Pan角度    (2B) = int16, 0.01度单位
            Tilt角度   (2B) = int16, 0.01度单位
            校验       (2B) = 简单累加
        """
        if not self.enabled or self.ser is None:
            return False
        
        pan  = int(pan_angle * 100)   # 转为0.01度单位
        tilt = int(tilt_angle * 100)
        
        flags = 0x0001 if laser_on else 0x0000
        
        # 打包
        data = struct.pack("<hh", pan, tilt)
        checksum = (0xA5 + 10 + 0x0200 + flags + pan + tilt) & 0xFFFF
        
        frame = bytearray([0xA5, 10])
        frame += struct.pack("<HH", 0x0200, flags)
        frame += data
        frame += struct.pack("<H", checksum)
        
        try:
            self.ser.write(bytes(frame))
            return True
        except Exception as e:
            print(f"[Serial] 发送失败: {e}")
            return False
    
    def close(self):
        if self.ser:
            self.ser.close()
            self.ser = None
            self.enabled = False


# =============================================================================
# 可视化绘制
# =============================================================================
def draw_red_spot(image, cx, cy, area, color=(0, 255, 0)):
    """在图像上绘制检测到的红色光斑"""
    if cx is not None and cy is not None:
        cx_i, cy_i = int(cx), int(cy)
        cv2.circle(image, (cx_i, cy_i), 12, color, 2)
        cv2.circle(image, (cx_i, cy_i), 4, (0, 0, 255), -1)
        cv2.line(image, (cx_i - 15, cy_i), (cx_i + 15, cy_i), color, 1)
        cv2.line(image, (cx_i, cy_i - 15), (cx_i, cy_i + 15), color, 1)
        cv2.putText(image, f"Red:({cx_i},{cy_i}) A:{area:.0f}",
                    (cx_i + 18, cy_i - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def draw_screen_frame(image, corners_px, color=(255, 0, 0)):
    """绘制检测到的屏幕边框"""
    if corners_px is None:
        return
    pts = corners_px.reshape(4, 2).astype(np.int32)
    for i in range(4):
        cv2.line(image, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, 2)
    # 标注原点
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    cv2.drawMarker(image, (cx, cy), (0, 255, 255),
                   cv2.MARKER_CROSS, 15, 1)


def draw_waypoints(image, waypoints_world, H_inv, color=(255, 255, 0)):
    """在世界坐标中绘制航点路径"""
    if len(waypoints_world) < 2:
        return
    prev_px = None
    for wp in waypoints_world[:200]:  # 采样避免太密
        px = world_to_pixel(wp, H_inv)
        if prev_px is not None:
            cv2.line(image, prev_px, px, color, 1)
        prev_px = px


def draw_status_panel(image, text_lines, start_y=10, line_h=22):
    """绘制状态信息面板"""
    for i, text in enumerate(text_lines):
        y = start_y + i * line_h
        cv2.putText(image, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)


print("[red_vision.py] 功能模块加载完成")
