"""通用串口协议底层。

本文件只处理固定帧结构、CRC、ACK和字节流解析，不包含任何具体题目的业务数据。
业务命令号和数据格式统一放在 protocol_commands.py 中。

帧结构（多字节数据均为小端序）：
    A5 5A | VER | FLAGS | SEQ | SRC | DST | CMD(u16) | LEN(u8)
          | PAYLOAD(LEN字节) | CRC16(u16)
"""

import struct


# ==================== 固定协议配置 ====================

FRAME_HEAD = b"\xA5\x5A"
VERSION = 0x01
MAX_PAYLOAD = 240
FIXED_HEADER_SIZE = 10
FRAME_OVERHEAD = 12

# FLAGS标志位
FLAG_NEED_ACK = 0x01
FLAG_IS_ACK = 0x02
FLAG_IS_ERROR = 0x04
FLAG_DATA_VALID = 0x08

# 常用设备地址
DEVICE_BROADCAST = 0x00
DEVICE_MAIN_MCU = 0x01
DEVICE_VISION = 0x02
DEVICE_GIMBAL = 0x03
DEVICE_CHASSIS = 0x04
DEVICE_ACTUATOR = 0x05
DEVICE_PC = 0xFE

_sequence = 0


# ==================== 发送端 ====================

def crc16(data):
    """计算CRC16-CCITT-FALSE。"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def make_frame(cmd, payload=b"", flags=0,
               src=DEVICE_VISION, dst=DEVICE_MAIN_MCU, seq=None):
    """填写固定字段、序号和CRC，生成一个完整数据帧。"""
    global _sequence

    payload = bytes(payload)
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"有效数据过长: {len(payload)} > {MAX_PAYLOAD}")

    if seq is None:
        seq = _sequence
        _sequence = (_sequence + 1) & 0xFF

    body = struct.pack(
        "<BBBBBHB",
        VERSION,
        flags & 0x0F,
        seq & 0xFF,
        src & 0xFF,
        dst & 0xFF,
        cmd & 0xFFFF,
        len(payload),
    ) + payload

    return FRAME_HEAD + body + struct.pack("<H", crc16(body))


def make_ack(request, result=0, src=None):
    """根据收到的命令生成ACK；SEQ和CMD与原命令保持一致。"""
    flags = FLAG_IS_ACK
    if result != 0:
        flags |= FLAG_IS_ERROR

    return make_frame(
        cmd=request["cmd"],
        payload=struct.pack("<B", result & 0xFF),
        flags=flags,
        seq=request["seq"],
        src=request["dst"] if src is None else src,
        dst=request["src"],
    )


# ==================== 接收端 ====================

def decode_frame(frame):
    """校验并解析一个已经完整收齐的数据帧。"""
    frame = bytes(frame)
    if len(frame) < FRAME_OVERHEAD or frame[:2] != FRAME_HEAD:
        return None

    version, flags, seq, src, dst, cmd, length = struct.unpack_from(
        "<BBBBBHB", frame, 2)
    if version != VERSION or len(frame) != FRAME_OVERHEAD + length:
        return None

    received_crc = struct.unpack_from("<H", frame, len(frame) - 2)[0]
    if crc16(frame[2:-2]) != received_crc:
        return None

    return {
        "version": version,
        "flags": flags,
        "seq": seq,
        "src": src,
        "dst": dst,
        "cmd": cmd,
        "payload": frame[FIXED_HEADER_SIZE:-2],
    }


def parse_frames(buffer, data=b""):
    """向接收缓存追加数据，并取出所有校验正确的完整帧。"""
    buffer.extend(data)
    frames = []

    while True:
        start = buffer.find(FRAME_HEAD)
        if start < 0:
            buffer[:] = buffer[-1:] if buffer[-1:] == FRAME_HEAD[:1] else b""
            break
        if start:
            del buffer[:start]
        if len(buffer) < FIXED_HEADER_SIZE:
            break

        length = buffer[9]
        frame_size = FRAME_OVERHEAD + length
        if length > MAX_PAYLOAD:
            del buffer[0]
            continue
        if len(buffer) < frame_size:
            break

        frame = decode_frame(buffer[:frame_size])
        if frame is None:
            del buffer[0]
            continue

        frames.append(frame)
        del buffer[:frame_size]

    return frames
