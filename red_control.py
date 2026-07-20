# -*- coding: utf-8 -*-
"""
23E题 - 红色激光运动目标控制系统 上位机 (Jetson)
======================================================
功能:
  1. 复位:     鼠标点击屏幕原点 -> 发送给下位机
  2. 0.5m正方形: 鼠标点击4个角点 -> 发送给下位机
  3. A4靶纸:   视觉识别内外矩形(8个角点) -> 发送给下位机
  持续:        实时检测红色激光笔光斑位置 -> 发送给下位机

操作说明:
  鼠标点击图像  - 标记点 (原点/正方形角点)
  键盘 '0'      - 标记为原点
  键盘 '1'~'4'  - 标记为正方形第1~4个角点
  键盘 'S'      - 发送全部数据到下位机
  键盘 'R'      - 重置所有标记点
  键盘 'C'      - 切换A4矩形检测
  键盘 '+/-'    - 调整二值化阈值
  键盘 'B'      - 切换二值化预览
  键盘 'Q'      - 退出

参考: 视觉代码带注释.py (江南大学团队)
"""

import sys, time, cv2, numpy as np, os, struct
from ctypes import *

# =============================================================================
# 海康相机SDK导入
# =============================================================================
IS_JETSON = os.path.exists('/etc/nv_tegra_release')

try:
    if IS_JETSON:
        sdk_paths = [
            "/opt/MVS/Samples/aarch64/Python/MvImport",
            "/usr/local/MVS/Samples/aarch64/Python/MvImport",
            "/home/nvidia/MVS/Samples/aarch64/Python/MvImport"
        ]
    else:
        sdk_paths = [
            "/opt/MVS/Samples/64/Python/MvImport",       # x86_64 标准路径
            "/opt/MVS/Samples/32/Python/MvImport",       # x86 标准路径
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
    SDK_OK = False

# =============================================================================
# 全局参数
# =============================================================================

# --- 图像处理阈值 ---
BINARY_THRESHOLD = 25       # 二值化阈值 (0-255), 越小检测越多黑色区域
CANNY_LOWER = 50             # Canny下阈值
CANNY_UPPER = 150            # Canny上阈值
GAUSSIAN_BLUR_SIZE = 5       # 高斯模糊核大小

# --- 矩形检测参数 ---
ANGLE_TOLERANCE = 30         # 矩形角度容差(度)
SIDE_RATIO_TOLERANCE = 0.4   # 对边长度比容差
MIN_CONTOUR_AREA = 1000      # 最小轮廓面积
MIN_AREA_RATIO = 0.7         # 内外矩形最小面积比

# --- 红色激光HSV检测 (可运行时调整) ---
RED_H_TOL  = 15      # H容差 (0-15范围, 另一半自动 180-RED_H_TOL ~ 179)
RED_S_LOW  = 100     # S下阈值, 越低越容易检测到不够饱和的光斑
RED_V_LOW  = 150     # V下阈值, 越低越容易检测到较暗的光斑
RED_SPOT_MIN_AREA = 2
RED_SPOT_MAX_AREA = 500

def _build_red_hsv_ranges():
    """根据当前参数构建HSV检测范围"""
    lower1 = np.array([0,            RED_S_LOW, RED_V_LOW])
    upper1 = np.array([RED_H_TOL,   255,        255])
    lower2 = np.array([180-RED_H_TOL, RED_S_LOW, RED_V_LOW])
    upper2 = np.array([179,          255,        255])
    return lower1, upper1, lower2, upper2

# --- 串口参数 ---
SERIAL_PORT = "/dev/ttyTHS1"
SERIAL_BAUDRATE = 921600

# --- 相机参数 ---
CAM_WIDTH, CAM_HEIGHT = 640, 480
EXPOSURE_US = 5000     # 曝光(us), 参考代码5000
GAIN_VAL = 12.0        # 增益, 参考代码12

# --- 用户标记点 ---
origin_point = None           # 原点 (鼠标标记)
square_corners = [None, None, None, None]  # 正方形4角点

# --- 检测结果缓存 ---
detected_outer_rect = None    # A4外矩形角点
detected_inner_rect = None    # A4内矩形角点
red_spot_pos = None           # 红色光斑位置
enable_a4_detect = True       # A4检测开关

# 调试开关
show_binary = False
show_edges = False
show_red_mask = False       # 红色光斑HSV掩膜预览

# 窗口状态跟踪 (防止cv2.destroyWindow报错, 参考视觉代码带注释.py)
_windows_created = {}

def _safe_show(name, img):
    _windows_created[name] = True
    cv2.imshow(name, img)

def _safe_hide(name):
    if _windows_created.get(name, False):
        cv2.destroyWindow(name)
        _windows_created[name] = False

print("[INFO] 参数初始化完成")

# =============================================================================
# 相机管理
# =============================================================================
class CameraManager:
    """海康工业相机 / USB摄像头 自动切换"""
    def __init__(self):
        self.cam = None; self.data_buf = None
        self.payload_size = 0; self.connected = False
        self.use_hik = False; self.usb_cap = None

    def init(self):
        if SDK_OK:
            try:
                if self._init_hik():
                    self.use_hik = True
                    return True
            except:
                pass
        return self._init_usb()

    def _init_hik(self):
        dl = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, dl)
        if ret != 0 or dl.nDeviceNum == 0:
            return False
        print(f"[CAM] 海康设备: {dl.nDeviceNum}个")
        self.cam = MvCamera()
        info = cast(dl.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        if self.cam.MV_CC_CreateHandle(info) != 0:
            return False
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Monitor, 0)
            if ret != 0: return False
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        param = MVCC_INTVALUE()
        memset(byref(param), 0, sizeof(MVCC_INTVALUE))
        r = self.cam.MV_CC_GetIntValue("PayloadSize", param)
        self.payload_size = param.nCurValue if r == 0 else CAM_WIDTH * CAM_HEIGHT * 3
        self.connected = True
        print("[CAM] 海康相机就绪")
        return True

    def _init_usb(self):
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                self.usb_cap = cap; self.connected = True
                print(f"[CAM] USB摄像头 /dev/video{idx}")
                return True
            cap.release()
        return False

    def start(self):
        if self.use_hik and self.cam:
            return self.cam.MV_CC_StartGrabbing() == 0
        return self.connected

    def get_frame(self):
        if self.use_hik:
            return self._get_hik()
        else:
            if self.usb_cap is None: return None
            ret, frame = self.usb_cap.read()
            return frame if ret else None

    def _get_hik(self):
        if not self.connected: return None
        if self.data_buf is None:
            self.data_buf = (c_ubyte * self.payload_size)()
        fi = MV_FRAME_OUT_INFO_EX()
        memset(byref(fi), 0, sizeof(fi))
        ret = self.cam.MV_CC_GetOneFrameTimeout(byref(self.data_buf), self.payload_size, fi, 2000)
        if ret != 0: return None
        data = np.frombuffer(self.data_buf, count=int(fi.nFrameLen), dtype=np.uint8)
        if len(data) == 0: return None
        bayer = data.reshape((fi.nHeight, fi.nWidth))
        return cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)

    def set_exposure(self, us):
        if self.use_hik and self.cam:
            self.cam.MV_CC_SetFloatValue("ExposureTime", float(us))

    def set_gain(self, g):
        if self.use_hik and self.cam:
            self.cam.MV_CC_SetFloatValue("Gain", float(g))

    def stop(self):
        if self.use_hik and self.cam:
            self.cam.MV_CC_StopGrabbing()

    def release(self):
        self.stop()
        if self.use_hik and self.cam:
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
        if self.usb_cap:
            self.usb_cap.release()
        print("[CAM] 已释放")


# =============================================================================
# 图像预处理 (对齐参考代码主循环的预处理管道)
# =============================================================================
def preprocess_image(image):
    """
    多模态预处理: 高斯模糊→灰度→二值化→闭运算→Canny边缘→融合
    参考: 视觉代码带注释.py 主循环预处理 (line 1916-1929)
    - 先用高斯模糊降噪
    - 小核(4x4)闭运算, 修复胶带细微断裂
    - 边缘+二值化OR融合
    """
    blurred = cv2.GaussianBlur(image, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(gray, CANNY_LOWER, CANNY_UPPER, apertureSize=3)
    combined = cv2.bitwise_or(binary_closed, edges)
    return combined, binary, edges

print("[INFO] Camera + preprocess OK")

# =============================================================================
# 矩形检测辅助函数 (from 视觉代码带注释.py)
# =============================================================================
def calculate_angle(p1, p2, p3):
    """三点角度, p2为顶点"""
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if len1 == 0 or len2 == 0: return 0
    cos_a = np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))

def calculate_side_lengths(corners):
    """四边形四边长"""
    sides = []
    for i in range(4):
        p1, p2 = corners[i], corners[(i+1)%4]
        sides.append(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))
    return sides

def check_rectangle_geometry(corners, angle_tol=ANGLE_TOLERANCE,
                              ratio_tol=SIDE_RATIO_TOLERANCE):
    """检查是否为矩形: 四角≈90度 + 对边长度相近"""
    if len(corners) != 4: return False
    angles = []
    for i in range(4):
        prev, curr, nxt = corners[(i-1)%4], corners[i], corners[(i+1)%4]
        angles.append(calculate_angle(prev, curr, nxt))
    if not all(abs(a - 90) <= angle_tol for a in angles):
        return False
    sides = calculate_side_lengths(corners)
    r1 = abs(sides[0]-sides[2])/max(sides[0], sides[2]) if max(sides[0], sides[2]) > 0 else 1
    r2 = abs(sides[1]-sides[3])/max(sides[1], sides[3]) if max(sides[1], sides[3]) > 0 else 1
    if r1 > ratio_tol or r2 > ratio_tol: return False
    return min(sides) >= 20 and max(sides)/min(sides) <= 10 if min(sides) > 0 else False

def sort_corners(corners):
    """角点排序: 左上-右上-右下-左下"""
    pts = corners.reshape(4, 2).astype(np.float32)
    centroid = np.mean(pts, axis=0)
    angles = [(np.arctan2(p[1]-centroid[1], p[0]-centroid[0]) + 2*np.pi) % (2*np.pi) for p in pts]
    sorted_idx = np.argsort(angles)
    # 找最接近225度(左下)的作为起点
    target = 5*np.pi/4
    best = 0; best_diff = float('inf')
    for i, idx in enumerate(sorted_idx):
        diff = abs(angles[idx] - target)
        diff = min(diff, 2*np.pi - diff)
        if diff < best_diff: best_diff = diff; best = i
    ordered = np.array([pts[sorted_idx[(best+i)%4]] for i in range(4)])
    # 二次验证: Y坐标分组
    y_mean = np.mean(ordered[:, 1])
    top = ordered[ordered[:, 1] <= y_mean]
    bot = ordered[ordered[:, 1] > y_mean]
    if len(top) == 2 and len(bot) == 2:
        top = top[np.argsort(top[:, 0])]   # 左上, 右上
        bot = bot[np.argsort(bot[:, 0])]    # 左下, 右下
        ordered = np.array([top[0], top[1], bot[1], bot[0]])
    return ordered.astype(np.int32)


# =============================================================================
# 嵌套矩形检测 (from 视觉代码带注释.py find_rectangles)
# =============================================================================
def find_rectangles(contours, hierarchy):
    """检测A4靶纸的内外嵌套矩形, 返回 [(contour, corners, area), ...]"""
    # 第一阶段: 筛选有效矩形
    valid = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area < 100 or peri < 20: continue  # 参考代码 line 1468
        epsilon = 0.02 * peri
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            sc = approx.reshape(4, 2)
            if check_rectangle_geometry(sc):
                x, y, w, h = cv2.boundingRect(cnt)
                parent = hierarchy[0][i][3] if hierarchy is not None else -1
                valid.append((cnt, approx, area, x, y, w, h, i, parent))

    if not valid: return []

    valid.sort(key=lambda x: x[2])  # 按面积升序

    # 第二阶段: 找父子嵌套对
    pairs = []
    for rect in valid:
        cnt, approx, area, x, y, w, h, idx, pid = rect
        if pid != -1:
            parent_rect = None
            for pc in valid:
                if pc[7] == pid: parent_rect = pc; break
            if parent_rect:
                ratio = area / parent_rect[2] if parent_rect[2] > 0 else 0
                if ratio >= MIN_AREA_RATIO:
                    pairs.append((parent_rect, rect))

    # 第三阶段: 选择最佳对
    if pairs:
        pairs.sort(key=lambda x: x[1][2])
        parent, child = pairs[0]
        selected = [child, parent]  # 内矩形, 外矩形
    else:
        # 几何嵌套检测
        nested = []
        for i, r1 in enumerate(valid):
            partners = []
            for j, r2 in enumerate(valid):
                if i == j: continue
                margin = 10
                if (r1[3] >= r2[3]-margin and r1[4] >= r2[4]-margin and
                    r1[3]+r1[5] <= r2[3]+r2[5]+margin and
                    r1[4]+r1[6] <= r2[4]+r2[6]+margin and
                    r1[2] < r2[2]*0.9):
                    ratio = r1[2] / r2[2] if r2[2] > 0 else 0
                    if ratio >= MIN_AREA_RATIO:
                        partners.append(r2)
            if partners:
                nested.append((r1, len(partners)))
        if len(nested) >= 2:
            nested.sort(key=lambda x: (-x[1], x[0][2]))
            selected = [nested[0][0][:3], nested[1][0][:3]]
        elif len(nested) >= 1:
            selected = [nested[0][0][:3]]
            for r in valid:
                if not any(r[0] is n[0][0] for n in nested):
                    selected.append(r[:3]); break
        else:
            if len(valid) >= 2:
                selected = [valid[0][:3], valid[-1][:3]]
            elif len(valid) == 1:
                selected = [valid[0][:3]]

    # 第四阶段: 角点排序 (参考代码 line 1577-1588)
    result = []
    for rect_data in selected:
        cnt = rect_data[0]
        approx = rect_data[1]
        area = rect_data[2]
        result.append((cnt, sort_corners(approx), area))
    return result


# =============================================================================
# 红色激光光斑检测
# =============================================================================
def detect_red_spot(image):
    """HSV双范围检测红色光斑, 返回 (cx, cy) 或 None"""
    lower1, upper1, lower2, upper2 = _build_red_hsv_ranges()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_cx, best_cy, best_area = None, None, 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if RED_SPOT_MIN_AREA <= area <= RED_SPOT_MAX_AREA and area > best_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                best_cx = M["m10"]/M["m00"]
                best_cy = M["m01"]/M["m00"]
                best_area = area
    return (int(best_cx), int(best_cy)) if best_cx is not None else None, mask

print("[INFO] Rectangle detection + red spot detection OK")

# =============================================================================
# 串口通信 (协议参考 视觉代码带注释.py pack_frame)
# =============================================================================
serial_port = None
serial_enabled = False

def init_serial(port=SERIAL_PORT, baudrate=SERIAL_BAUDRATE):
    """初始化串口"""
    global serial_port, serial_enabled
    try:
        import serial
        serial_port = serial.Serial(port, baudrate, timeout=0.05)
        serial_enabled = True
        print(f"[SER] {port} @ {baudrate} OK")
        return True
    except Exception as e:
        print(f"[SER] 失败: {e}, 无串口模式")
        serial_enabled = False
        return False

def pack_frame(cmd_id, flags, data_floats):
    """打包数据帧, 协议同参考代码"""
    n = len(data_floats)
    length = 2 + 2 + 4 * n  # cmd_id(2) + flags(2) + floats(4*n)
    buf = bytearray([0xA5, length & 0xFF])
    buf += struct.pack('<HH', cmd_id, flags)
    for f in data_floats:
        buf += struct.pack('<f', f)
    # 简单校验
    cs = sum(buf) & 0xFF
    buf.append(cs)
    return bytes(buf)

def send_points_to_mcu(origin, square_pts, outer_rect, inner_rect, red_spot):
    """发送全部数据到下位机"""
    global serial_port, serial_enabled
    if not serial_enabled:
        print("[SER] 串口未启用, 仅打印数据")
        _print_data(origin, square_pts, outer_rect, inner_rect, red_spot)
        return

    # 帧1: 原点 (cmd=0x0100)
    if origin is not None:
        data = [float(origin[0]), float(origin[1])] + [0.0]*10
        serial_port.write(pack_frame(0x0100, 0x0000, data))
        print(f"[SER] 发送原点: ({origin[0]}, {origin[1]})")

    # 帧2: 正方形4角点 (cmd=0x0101)
    if all(p is not None for p in square_pts):
        data = []
        for p in square_pts:
            data.extend([float(p[0]), float(p[1])])
        data += [0.0] * (12 - len(data))  # 补齐12个float
        serial_port.write(pack_frame(0x0101, 0x0000, data))
        print(f"[SER] 发送正方形4角点")

    # 帧3: A4外矩形 (cmd=0x0102)
    if outer_rect is not None:
        data = []
        for p in outer_rect:
            data.extend([float(p[0]), float(p[1])])
        data += [0.0] * (12 - len(data))
        serial_port.write(pack_frame(0x0102, 0x0000, data))
        print(f"[SER] 发送A4外矩形")

    # 帧4: A4内矩形 (cmd=0x0103)
    if inner_rect is not None:
        data = []
        for p in inner_rect:
            data.extend([float(p[0]), float(p[1])])
        data += [0.0] * (12 - len(data))
        serial_port.write(pack_frame(0x0103, 0x0000, data))
        print(f"[SER] 发送A4内矩形")

    # 帧5: 红色光斑 (cmd=0x0104) - 持续发送
    if red_spot is not None:
        data = [float(red_spot[0]), float(red_spot[1])] + [0.0]*10
        serial_port.write(pack_frame(0x0104, 0x0000, data))

def _print_data(origin, square_pts, outer_rect, inner_rect, red_spot):
    """无串口时打印数据到控制台"""
    print("=" * 50)
    if origin:
        print(f"原点: ({origin[0]}, {origin[1]})")
    if all(p is not None for p in square_pts):
        print(f"正方形角点: {square_pts}")
    if outer_rect is not None:
        print(f"A4外矩形: {outer_rect.tolist()}")
    if inner_rect is not None:
        print(f"A4内矩形: {inner_rect.tolist()}")
    if red_spot:
        print(f"红色光斑: ({red_spot[0]}, {red_spot[1]})")
    print("=" * 50)

def close_serial():
    global serial_port, serial_enabled
    if serial_port:
        serial_port.close()
        serial_enabled = False

print("[INFO] Serial communication OK")

# =============================================================================
# 鼠标回调
# =============================================================================
clicked_point = None
current_mouse_pos = None

def mouse_callback(event, x, y, flags, param):
    global clicked_point, current_mouse_pos
    current_mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"[CLICK] ({x}, {y})")

# =============================================================================
# 可视化绘制
# =============================================================================
def draw_marked_points(image):
    """绘制用户标记的原点和正方形角点"""
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255)]

    # 原点
    if origin_point is not None:
        cv2.circle(image, origin_point, 8, colors[0], 2)
        cv2.circle(image, origin_point, 3, colors[0], -1)
        cv2.putText(image, "Origin", (origin_point[0]+12, origin_point[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 1)

    # 正方形4角
    for i, pt in enumerate(square_corners):
        if pt is not None:
            cv2.circle(image, pt, 8, colors[i+1], 2)
            cv2.circle(image, pt, 3, colors[i+1], -1)
            cv2.putText(image, f"P{i+1}", (pt[0]+12, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i+1], 1)

    # 正方形连线
    valid_pts = [p for p in square_corners if p is not None]
    if len(valid_pts) >= 2:
        cv2.polylines(image, [np.array(valid_pts)], len(valid_pts) == 4,
                      (255, 255, 0), 2)


def draw_detected_rectangles(image):
    """绘制检测到的A4内外矩形"""
    # 外矩形 - 蓝色
    if detected_outer_rect is not None:
        pts = detected_outer_rect.reshape(-1, 1, 2)
        cv2.polylines(image, [pts], True, (255, 0, 0), 2)
        for i, p in enumerate(detected_outer_rect):
            cv2.circle(image, tuple(p), 5, (255, 0, 0), -1)
            cv2.putText(image, f"O{i+1}", (p[0]+8, p[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

    # 内矩形 - 绿色
    if detected_inner_rect is not None:
        pts = detected_inner_rect.reshape(-1, 1, 2)
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        for i, p in enumerate(detected_inner_rect):
            cv2.circle(image, tuple(p), 5, (0, 255, 0), -1)
            cv2.putText(image, f"I{i+1}", (p[0]+8, p[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)


def draw_red_spot_marker(image, spot):
    """绘制红色光斑位置"""
    if spot is None: return
    cv2.circle(image, spot, 12, (0, 255, 255), 2)
    cv2.circle(image, spot, 4, (0, 0, 255), -1)
    cv2.line(image, (spot[0]-15, spot[1]), (spot[0]+15, spot[1]), (0, 255, 255), 1)
    cv2.line(image, (spot[0], spot[1]-15), (spot[0], spot[1]+15), (0, 255, 255), 1)
    cv2.putText(image, f"Red({spot[0]},{spot[1]})",
                (spot[0]+18, spot[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def draw_status_bar(image, extra_lines=None):
    """绘制状态栏"""
    lines = [
        f"Binary:{BINARY_THRESHOLD} | Canny:[{CANNY_LOWER},{CANNY_UPPER}] | RedHSV: H±{RED_H_TOL} S>{RED_S_LOW} V>{RED_V_LOW}",
        f"Serial:{'ON' if serial_enabled else 'OFF'} | A4Detect:{'ON' if enable_a4_detect else 'OFF'}",
        f"Origin:{'OK' if origin_point else '--'} | "
        f"Square:{sum(1 for p in square_corners if p is not None)}/4"
    ]
    if extra_lines:
        lines.extend(extra_lines)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (10, CAM_HEIGHT - 20 - (len(lines)-1-i)*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

print("[INFO] Mouse callback + Drawing OK")

# =============================================================================
# 主程序
# =============================================================================
def main():
    global BINARY_THRESHOLD, enable_a4_detect, show_binary, show_edges, show_red_mask
    global CANNY_LOWER, CANNY_UPPER
    global RED_S_LOW, RED_V_LOW
    global origin_point, square_corners, clicked_point
    global detected_outer_rect, detected_inner_rect, red_spot_pos
    global serial_enabled

    print("=" * 55)
    print("  23E题 - 红色激光运动目标控制系统 上位机")
    print("=" * 55)

    # --- 初始化相机 ---
    cam = CameraManager()
    if not cam.init():
        print("[ERR] 相机初始化失败!")
        return
    cam.set_exposure(EXPOSURE_US)
    cam.set_gain(GAIN_VAL)
    if not cam.start():
        print("[ERR] 无法开始采集!")
        return

    # --- 初始化串口 ---
    init_serial()

    # --- OpenCV窗口 ---
    cv2.namedWindow("23E Red Control", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("23E Red Control", mouse_callback)
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)

    # --- 状态变量 ---
    frame_count = 0
    start_time = time.time()
    send_counter = 0           # 串口发送计数器
    SEND_INTERVAL = 5          # 每N帧发送一次红色光斑

    print("\n操作说明:")
    print("  鼠标点击 + 数字键标记:")
    print("    '0' -> 标记原点")
    print("    '1'~'4' -> 标记正方形角点")
    print("  'S' -> 发送全部数据到下位机")
    print("  'R' -> 重置所有标记点")
    print("  'C' -> 切换A4矩形检测")
    print("  调试窗口:")
    print("    'B' -> 二值化  'E' -> 边缘  'H' -> 红色光斑HSV掩膜")
    print("    '+/-' -> 调整二值化阈值")
    print("  'Q' -> 退出")
    print("-" * 55)

    # ====== 主循环 ======
    while True:
        # --- 采集 ---
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.001)
            continue
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        frame_count += 1

        # --- A4矩形检测 ---
        a4_rects = []
        combined_img = None
        binary_img = None
        edges_img = None

        if enable_a4_detect:
            combined_img, binary_img, edges_img = preprocess_image(frame)
            contours, hierarchy = cv2.findContours(
                combined_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            a4_rects = find_rectangles(contours, hierarchy)

            if len(a4_rects) >= 2:
                # 按面积排序, 小的为内矩形, 大的为外矩形
                a4_rects.sort(key=lambda x: x[2])
                detected_inner_rect = a4_rects[0][1]  # 内矩形角点
                detected_outer_rect = a4_rects[1][1]  # 外矩形角点
            elif len(a4_rects) == 1:
                detected_outer_rect = a4_rects[0][1]
                detected_inner_rect = None
            else:
                # 没有检测到, 保持上一次结果
                pass

        # --- 红色光斑检测 ---
        red_spot_pos, red_mask = detect_red_spot(frame)

        # --- 处理鼠标点击 ---
        # (在主循环中处理, 通过键盘数字键分配)

        # --- 串口发送红色光斑 (持续) ---
        if serial_enabled and red_spot_pos is not None:
            send_counter += 1
            if send_counter >= SEND_INTERVAL:
                send_points_to_mcu(origin_point, square_corners,
                                   detected_outer_rect, detected_inner_rect,
                                   red_spot_pos)
                send_counter = 0

        # --- 可视化 ---
        display = frame.copy()

        # 绘制用户标记点
        draw_marked_points(display)

        # 绘制A4检测结果
        draw_detected_rectangles(display)

        # 绘制红色光斑
        draw_red_spot_marker(display, red_spot_pos)

        # 鼠标当前位置
        if current_mouse_pos:
            cv2.putText(display, f"Mouse:{current_mouse_pos}",
                       (CAM_WIDTH - 140, 15), cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (200, 200, 200), 1)

        # 状态栏
        status_extra = []
        if detected_outer_rect is not None:
            status_extra.append(f"A4 Outer: {len(a4_rects)} rects found")
        draw_status_bar(display, status_extra)

        # 窗口标题
        cv2.putText(display, "23E Red Laser Control", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- 显示 ---
        cv2.imshow("23E Red Control", display)

        # 调试窗口
        if show_binary and binary_img is not None:
            _safe_show("Binary", binary_img)
        else:
            _safe_hide("Binary")

        if show_edges and edges_img is not None:
            _safe_show("Edges", edges_img)
        else:
            _safe_hide("Edges")

        if show_red_mask and red_mask is not None:
            _safe_show("Red Mask", red_mask)
        else:
            _safe_hide("Red Mask")

        # --- 键盘处理 ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("用户退出")
            break

        elif key == ord('0'):
            if clicked_point is not None:
                origin_point = clicked_point
                print(f"[MARK] 原点: {origin_point}")

        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            idx = key - ord('1')  # 0, 1, 2, 3
            if clicked_point is not None:
                square_corners[idx] = clicked_point
                print(f"[MARK] 正方形角点{idx+1}: {clicked_point}")

        elif key == ord('s') or key == ord('S'):
            # 发送全部数据
            send_points_to_mcu(origin_point, square_corners,
                               detected_outer_rect, detected_inner_rect,
                               red_spot_pos)
            print("[SEND] 数据已发送")

        elif key == ord('r') or key == ord('R'):
            # 重置
            origin_point = None
            square_corners = [None, None, None, None]
            clicked_point = None
            detected_outer_rect = None
            detected_inner_rect = None
            print("[RESET] 所有标记点已重置")

        elif key == ord('c') or key == ord('C'):
            enable_a4_detect = not enable_a4_detect
            print(f"[TOGGLE] A4检测: {'ON' if enable_a4_detect else 'OFF'}")

        elif key == ord('b') or key == ord('B'):
            show_binary = not show_binary
            print(f"[TOGGLE] 二值化预览: {'ON' if show_binary else 'OFF'}")

        elif key == ord('e') or key == ord('E'):
            show_edges = not show_edges
            print(f"[TOGGLE] 边缘检测预览: {'ON' if show_edges else 'OFF'}")

        elif key == ord('h') or key == ord('H'):
            show_red_mask = not show_red_mask
            print(f"[TOGGLE] 红色光斑HSV掩膜: {'ON' if show_red_mask else 'OFF'}")

        elif key == ord('+') or key == ord('='):
            BINARY_THRESHOLD = min(255, BINARY_THRESHOLD + 5)
            print(f"[ADJ] 二值化阈值: {BINARY_THRESHOLD}")

        elif key == ord('-'):
            BINARY_THRESHOLD = max(0, BINARY_THRESHOLD - 5)
            print(f"[ADJ] 二值化阈值: {BINARY_THRESHOLD}")

        elif key == ord('['):
            CANNY_LOWER = max(0, CANNY_LOWER - 10)
            print(f"[ADJ] Canny下阈值: {CANNY_LOWER} (上:{CANNY_UPPER})")

        elif key == ord(']'):
            CANNY_LOWER = min(CANNY_UPPER - 10, CANNY_LOWER + 10)
            print(f"[ADJ] Canny下阈值: {CANNY_LOWER} (上:{CANNY_UPPER})")

        elif key == ord('{'):
            CANNY_UPPER = max(CANNY_LOWER + 10, CANNY_UPPER - 10)
            print(f"[ADJ] Canny上阈值: {CANNY_UPPER} (下:{CANNY_LOWER})")

        elif key == ord('}'):
            CANNY_UPPER = min(255, CANNY_UPPER + 10)
            print(f"[ADJ] Canny上阈值: {CANNY_UPPER} (下:{CANNY_LOWER})")

        elif key == ord('k') or key == ord('K'):
            RED_S_LOW = max(0, RED_S_LOW - 10)
            print(f"[ADJ] 红色S下阈值: {RED_S_LOW} (V>{RED_V_LOW})")

        elif key == ord('l') or key == ord('L'):
            RED_S_LOW = min(255, RED_S_LOW + 10)
            print(f"[ADJ] 红色S下阈值: {RED_S_LOW} (V>{RED_V_LOW})")

        elif key == ord(',') or key == ord('<'):
            RED_V_LOW = max(0, RED_V_LOW - 10)
            print(f"[ADJ] 红色V下阈值: {RED_V_LOW} (S>{RED_S_LOW})")

        elif key == ord('.') or key == ord('>'):
            RED_V_LOW = min(255, RED_V_LOW + 10)
            print(f"[ADJ] 红色V下阈值: {RED_V_LOW} (S>{RED_S_LOW})")

        elif key == ord('p') or key == ord('P'):
            # 打印当前所有数据
            _print_data(origin_point, square_corners,
                       detected_outer_rect, detected_inner_rect, red_spot_pos)

        # --- FPS ---
        if frame_count % 60 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[FPS] {fps:.1f} | Thresh:{BINARY_THRESHOLD} | "
                  f"A4:{'ON' if enable_a4_detect else 'OFF'}")

    # ====== 清理 ======
    cv2.destroyAllWindows()
    cam.release()
    close_serial()
    print("系统已退出")


if __name__ == "__main__":
    main()
