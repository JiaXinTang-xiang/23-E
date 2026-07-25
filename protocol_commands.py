"""通用协议的命令号和数据格式表。

以后更换题目时，优先在本文件中增加 CMD 和 FMT；协议底层 protocol_v1.py
通常不需要修改。FMT使用Python struct格式，所有多字节数据均为小端序。
"""


# ==================== 系统命令 0x00xx ====================

# 运行时间:u32，设备状态:u8，错误码:u8
CMD_HEARTBEAT = 0x0001
FMT_HEARTBEAT = "<IBB"

# ACK不单独占用命令号：回复沿用原命令CMD和SEQ，Payload只有结果码:u8
FMT_ACK = "<B"

# 设备类型:u8，主版本:u8，次版本:u8，功能标志:u16
CMD_DEVICE_INFO = 0x0002
FMT_DEVICE_INFO = "<BBBH"

# 错误来源:u8，错误码:u16，附加信息:i32
CMD_ERROR_REPORT = 0x0003
FMT_ERROR_REPORT = "<BHi"


# ==================== 视觉命令 0x01xx ====================

# 目标ID:u8，x/y:uint16，置信度:u8
CMD_TARGET_POINT = 0x0101
FMT_TARGET_POINT = "<BHHB"

# 23E A4靶纸：图像宽/高:uint16，外框四组x/y:uint16，内框四组x/y:uint16
# 角点顺序统一为：左上、右上、右下、左下
CMD_A4_TARGET = 0x0102
FMT_A4_TARGET = "<18H"

# 23E红色激光点：x/y:uint16；是否检测到由FLAG_DATA_VALID表示
CMD_RED_LASER = 0x0104
FMT_RED_LASER = "<HH"

# 类别ID:u8，置信度:u8
CMD_CLASS_RESULT = 0x0105
FMT_CLASS_RESULT = "<BB"

# 道路类型:u8，横向偏差:int16，角度x100:int16
CMD_LINE_ERROR = 0x0106
FMT_LINE_ERROR = "<Bhh"


# ==================== 运动控制 0x02xx ====================

# vx/vy:mm每秒，角速度x100，均为int16
CMD_CHASSIS_SPEED = 0x0201
FMT_CHASSIS_SPEED = "<hhh"

# 水平角x100:int16，俯仰角x100:int16，激光开关:u8，模式:u8
CMD_GIMBAL_CONTROL = 0x0202
FMT_GIMBAL_CONTROL = "<hhBB"

# 电机ID:u8，模式:u8，位置:int32，速度:u16，加速度:u16
CMD_MOTOR_POSITION = 0x0203
FMT_MOTOR_POSITION = "<BBiHH"

# 舵机ID:u8，角度x100:int16，动作时间ms:u16
CMD_SERVO_CONTROL = 0x0204
FMT_SERVO_CONTROL = "<BhH"

# 目标模块:u8，启动状态:u8
CMD_START_STOP = 0x0205
FMT_START_STOP = "<BB"


# ==================== 参数设置 0x03xx ====================

# 模块ID:u8，工作模式:u8
CMD_SET_MODE = 0x0301
FMT_SET_MODE = "<BB"

# 控制器ID:u8，kp/ki/kd均放大1000后按int32发送
CMD_SET_PID = 0x0302
FMT_SET_PID = "<Biii"

# 参数ID:u16，参数值:int32
CMD_SET_PARAMETER = 0x0303
FMT_SET_PARAMETER = "<Hi"

# 保存区域:u8
CMD_SAVE_PARAMETER = 0x0304
FMT_SAVE_PARAMETER = "<B"


# ==================== 状态反馈 0x04xx ====================

# 系统状态:u8，错误码:u8，供电电压mV:u16
CMD_SYSTEM_STATUS = 0x0401
FMT_SYSTEM_STATUS = "<BBH"

# 电机ID:u8，状态:u8，速度:int16，位置:int32
CMD_MOTOR_STATUS = 0x0402
FMT_MOTOR_STATUS = "<BBhi"

# 视觉状态:u8，帧率x10:u16，识别耗时ms:u16
CMD_VISION_STATUS = 0x0403
FMT_VISION_STATUS = "<BHH"
