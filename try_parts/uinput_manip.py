import uinput
import time
import struct

# # 定义事件格式和大小
# EVENT_FORMAT = 'IhBB'
# EVENT_SIZE = struct.calcsize(EVENT_FORMAT)

# # 定义事件类型
# JS_EVENT_BUTTON = 0x01  # 按钮事件
# JS_EVENT_AXIS = 0x02   # 轴事件
# JS_EVENT_INIT = 0x80   # 初始化事件

# # 创建一个uinput设备
# device = uinput.Device([
#     uinput.BTN_TRIGGER,  # 按钮
#     uinput.ABS_X,        # 摇杆X轴
#     uinput.ABS_Y,        # 摇杆Y轴
# ])

# # 模拟摇杆移动
# def move_joystick(x, y):
#     # 摇杆移动事件
#     device.emit(uinput.ABS_X, x)
#     device.emit(uinput.ABS_Y, y)

# # 模拟按钮按下和释放
# def press_button(button):
#     # 按钮按下事件
#     device.emit(uinput.BTN_TRIGGER, 1)
#     # 按钮释放事件
#     device.emit(uinput.BTN_TRIGGER, 0)

# # 示例：模拟摇杆向右移动和按钮按下
# move_joystick(50, 0)  # 向右移动摇杆
# press_button(uinput.BTN_TRIGGER)  # 按下按钮

# # 等待一段时间
# time.sleep(1)

# # 清理，释放资源
# device.close()

# 定义摇杆和按钮的事件代码
ABS_X = uinput.ABS_X
ABS_Y = uinput.ABS_Y
BTN_TRIGGER = uinput.BTN_TRIGGER
# 创建一个虚拟设备，包括摇杆和按钮
with uinput.Device([
    uinput.ABS_X,
    uinput.ABS_Y,
    uinput.BTN_TRIGGER,
    uinput.BTN_THUMB,
    uinput.BTN_TOP,
    uinput.BTN_A,
    uinput.BTN_B,
    uinput.BTN_C,
    uinput.BTN_X,
    uinput.BTN_Y,
    uinput.BTN_Z,
]) as device:
    try:
        while True:
            # 模拟摇杆移动到中间位置
            device.emit(ABS_X, 0)  # 将X轴移动到中心
            device.emit(ABS_Y, 0)  # 将Y轴移动到中心
            # 模拟按下按钮
            device.emit(BTN_TRIGGER, 1)  # 按下扳机按钮
            # device.emit(BTN_THUMB, 1)    # 按下拇指按钮
            # 发送同步事件
            device.emit(uinput.EV_SYN, uinput.SYN_REPORT)
            # 等待一段时间
            time.sleep(0.5)
            # 模拟摇杆移动到特定位置
            device.emit(ABS_X, 5000)  # 将X轴移动到特定值
            device.emit(ABS_Y, -5000) # 将Y轴移动到特定值
            # 模拟释放按钮
            device.emit(BTN_TRIGGER, 0)  # 释放扳机按钮
            # device.emit(BTN_THUMB, 0)    # 释放拇指按钮
            # 发送同步事件
            device.emit(uinput.EV_SYN, uinput.SYN_REPORT)
            # 等待一段时间
            time.sleep(0.5)
    except KeyboardInterrupt:
        # 如果用户中断程序（例如，按Ctrl+C），则退出循环
        print("程序被用户中断")







