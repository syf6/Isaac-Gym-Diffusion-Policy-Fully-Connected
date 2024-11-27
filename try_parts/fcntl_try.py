import os
import fcntl
import struct
# 定义输入事件结构体
class input_event(struct.Struct):
    _fields_ = [('time', 'I'),
               ('type', 'H'),
               ('code', 'H'),
               ('value', 'i')]
# 打开手柄设备
js_dev = '/dev/input/js0'
fd = os.open(js_dev, os.O_RDWR)
try:
    # 计算输入事件结构体的大小
    event_size = input_event.calcsize()
    # 读取手柄事件
    while True:
        data = os.read(fd, event_size)
        if not data:
            break
        ev = input_event.unpack(data)
        print(f"time: {ev.time}, type: {ev.type}, code: {ev.code}, value: {ev.value}")
    # 模拟按钮按下事件
    button_event = input_event()
    button_event.type = 1  # EV_KEY
    button_event.code = 0x130  # 假设这是你想要模拟的按钮，例如BTN_TRIGGER
    button_event.value = 1  # 1 表示按下
    os.write(fd, button_event.pack())
    # 模拟摇杆移动事件
    joystick_event = input_event()
    joystick_event.type = 3  # EV_ABS
    joystick_event.code = 0x00  # 假设这是你想要模拟的摇杆轴，例如ABS_X
    joystick_event.value = 5000  # 摇杆的值
    os.write(fd, joystick_event.pack())
finally:
    # 关闭手柄设备
    os.close(fd)