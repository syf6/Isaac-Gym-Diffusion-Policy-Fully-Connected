import time
from evdev import UInput, ecodes as e
# 创建一个UInput设备对象
ui = UInput()
# 模拟按下按钮
def press_button(button):
    ui.write(e.EV_KEY, button, 1)
    ui.syn()
# 模拟释放按钮
def release_button(button):
    ui.write(e.EV_KEY, button, 0)
    ui.syn()
# 模拟摇杆移动
def move_joystick(x, y):
    ui.write(e.EV_ABS, e.ABS_X, x)
    ui.write(e.EV_ABS, e.ABS_Y, y)
    ui.syn()
# 示例：模拟按下A按钮
press_button(e.BTN_A)
time.sleep(1)  # 等待1秒
release_button(e.BTN_A)
# 示例：模拟摇杆移动到中心点
move_joystick(0, 0)
time.sleep(1)  # 等待1秒
# 示例：模拟摇杆移动到左上角
move_joystick(-32767, 32767)  # 这些值可能需要根据你的设备进行调整
time.sleep(1)  # 等待1秒
# 示例：模拟摇杆回到中心点
move_joystick(0, 0)
time.sleep(1)  # 等待1秒
# 销毁虚拟设备
ui.destroy()