"""用于视觉设备与嵌入式设备通信的通用二进制协议。

帧结构（所有多字节数据均使用小端字节序）：
    A5 5A | VER | FLAGS | SEQ | SRC | DST | CMD(u16) | LEN(u8)
          | PAYLOAD(LEN bytes) | CRC16(u16)

CRC16-CCITT-FALSE 的计算范围从 VER 开始，到 PAYLOAD 最后一个字节结束。
"""

import struct


SOF = b"\xA5\x5A"
VERSION = 0x01
MAX_PAYLOAD = 240
FIXED_HEADER_SIZE = 10
FRAME_OVERHEAD = 12

# 标志位
FLAG_NEED_ACK = 0x01
FLAG_IS_ACK = 0x02
FLAG_IS_ERROR = 0x04
FLAG_DATA_VALID = 0x08

# 设备地址
DEVICE_BROADCAST = 0x00
DEVICE_MAIN_MCU = 0x01
DEVICE_VISION = 0x02
DEVICE_GIMBAL = 0x03
DEVICE_CHASSIS = 0x04
DEVICE_ACTUATOR = 0x05
DEVICE_PC = 0xFE

# 系统命令
CMD_HEARTBEAT = 0x0001
CMD_DEVICE_INFO = 0x0002
CMD_ERROR_REPORT = 0x0003

# 视觉命令，保留原23E项目已使用的命令ID。
CMD_TARGET_POINT = 0x0101
CMD_OUTER_RECT = 0x0102
CMD_INNER_RECT = 0x0103
CMD_LASER_POINT = 0x0104
CMD_CLASS_RESULT = 0x0105
CMD_LINE_ERROR = 0x0106
CMD_TARGET_LIST = 0x0107

# 运动控制、参数配置和状态反馈命令。
CMD_CHASSIS_SPEED = 0x0201
CMD_GIMBAL_CONTROL = 0x0202
CMD_MOTOR_POSITION = 0x0203
CMD_SERVO_CONTROL = 0x0204
CMD_START_STOP = 0x0205
CMD_SET_MODE = 0x0301
CMD_SET_PID = 0x0302
CMD_SET_PARAMETER = 0x0303
CMD_SAVE_PARAMETER = 0x0304
CMD_SYSTEM_STATUS = 0x0401
CMD_MOTOR_STATUS = 0x0402
CMD_VISION_STATUS = 0x0403


class ProtocolError(ValueError):
    """协议帧格式错误。"""


class SequenceCounter:
    """生成0~255循环的8位无符号帧序号。"""

    def __init__(self, initial=0):
        self._value = initial & 0xFF

    def next(self):
        value = self._value
        self._value = (self._value + 1) & 0xFF
        return value


def crc16_ccitt_false(data):
    """计算CRC16-CCITT-FALSE（多项式0x1021，初始值0xFFFF）。"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def pack_frame(cmd, payload=b"", flags=0, seq=0,
               src=DEVICE_VISION, dst=DEVICE_MAIN_MCU, version=VERSION):
    """打包一个完整的V1协议帧。"""
    payload = bytes(payload)
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"有效数据过长: {len(payload)} > {MAX_PAYLOAD}")
    if flags & 0xF0:
        raise ProtocolError(f"保留标志位必须为0: 0x{flags:02X}")

    body = struct.pack(
        "<BBBBBHB",
        version & 0xFF,
        flags & 0xFF,
        seq & 0xFF,
        src & 0xFF,
        dst & 0xFF,
        cmd & 0xFFFF,
        len(payload),
    ) + payload
    crc = crc16_ccitt_false(body)
    return SOF + body + struct.pack("<H", crc)


def unpack_frame(frame):
    """校验并解析一个完整V1协议帧，返回字段字典。"""
    frame = bytes(frame)
    if len(frame) < FRAME_OVERHEAD:
        raise ProtocolError(f"数据帧过短: {len(frame)}")
    if frame[:2] != SOF:
        raise ProtocolError("帧头错误")

    version, flags, seq, src, dst, cmd, payload_len = struct.unpack_from(
        "<BBBBBHB", frame, 2)
    expected_size = FRAME_OVERHEAD + payload_len
    if len(frame) != expected_size:
        raise ProtocolError(
            f"帧长度不匹配: 期望{expected_size}，实际{len(frame)}")
    if payload_len > MAX_PAYLOAD:
        raise ProtocolError(f"有效数据长度非法: {payload_len}")

    expected_crc = struct.unpack_from("<H", frame, len(frame) - 2)[0]
    actual_crc = crc16_ccitt_false(frame[2:-2])
    if actual_crc != expected_crc:
        raise ProtocolError(
            f"CRC校验失败: 帧内0x{expected_crc:04X}，计算0x{actual_crc:04X}")

    return {
        "version": version,
        "flags": flags,
        "seq": seq,
        "src": src,
        "dst": dst,
        "cmd": cmd,
        "payload": frame[FIXED_HEADER_SIZE:-2],
        "crc": expected_crc,
    }


class ProtocolParser:
    """从任意拆包、粘包或带噪声的字节流中恢复完整V1帧。"""

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data):
        self._buffer.extend(data)
        frames = []

        while True:
            start = self._buffer.find(SOF)
            if start < 0:
                if self._buffer[-1:] == SOF[:1]:
                    self._buffer[:] = self._buffer[-1:]
                else:
                    self._buffer.clear()
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < FIXED_HEADER_SIZE:
                break

            payload_len = self._buffer[9]
            if payload_len > MAX_PAYLOAD:
                del self._buffer[0]
                continue

            frame_size = FRAME_OVERHEAD + payload_len
            if len(self._buffer) < frame_size:
                break

            candidate = bytes(self._buffer[:frame_size])
            try:
                frames.append(unpack_frame(candidate))
                del self._buffer[:frame_size]
            except ProtocolError:
                del self._buffer[0]

        return frames


def pack_rect_payload(points):
    """将四个(x, y)图像坐标按uint16打包。"""
    if points is None or len(points) != 4:
        raise ProtocolError("矩形数据必须恰好包含四个点")
    values = []
    for point in points:
        x, y = point
        values.extend((_clamp_u16(x), _clamp_u16(y)))
    return struct.pack("<B8H", 4, *values)


def unpack_rect_payload(payload):
    """解析矩形四角点数据。"""
    if len(payload) != 17:
        raise ProtocolError(f"矩形数据必须为17字节，实际{len(payload)}")
    values = struct.unpack("<B8H", payload)
    if values[0] != 4:
        raise ProtocolError(f"不支持的角点数量: {values[0]}")
    return [(values[i], values[i + 1]) for i in range(1, 9, 2)]


def pack_laser_payload(x=0, y=0, color=1, confidence=100):
    """打包激光点颜色、位置和置信度。"""
    return struct.pack(
        "<BHHB",
        color & 0xFF,
        _clamp_u16(x),
        _clamp_u16(y),
        max(0, min(100, int(confidence))),
    )


def unpack_laser_payload(payload):
    """解析激光点数据。"""
    if len(payload) != 6:
        raise ProtocolError(f"激光点数据必须为6字节，实际{len(payload)}")
    color, x, y, confidence = struct.unpack("<BHHB", payload)
    return {
        "color": color,
        "x": x,
        "y": y,
        "confidence": confidence,
    }


def pack_heartbeat_payload(uptime_ms, state=0, error_code=0):
    """打包心跳数据：运行时间、设备状态和错误码。"""
    return struct.pack(
        "<IBB",
        int(uptime_ms) & 0xFFFFFFFF,
        state & 0xFF,
        error_code & 0xFF,
    )


def pack_ack_payload(result=0):
    """打包ACK结果码。"""
    return struct.pack("<B", result & 0xFF)


def _clamp_u16(value):
    """将数值四舍五入并限制到uint16范围。"""
    return max(0, min(0xFFFF, int(round(float(value)))))
