import ctypes
import time

try:
    from pynput.keyboard import Controller, Key
    _pynput_kb = Controller()
except Exception:
    _pynput_kb = None

user32 = ctypes.windll.user32
SendInput = user32.SendInput
MAPVK_VSC_TO_VK = 1

# Hardware Scan Codes
up_pressed = 0x48       # Up Arrow
down_pressed = 0x50     # Down Arrow
left_pressed = 0x4B     # Left Arrow
right_pressed = 0x4D    # Right Arrow
nitro_pressed = 0x39    # Spacebar
shift_pressed = 0x2A    # Left Shift (Nitro / Drift in web games)

# WASD Alternative Scan Codes
w_pressed = 0x11
s_pressed = 0x1F
a_pressed = 0x1E
d_pressed = 0x20

# Arrow keys require KEYEVENTF_EXTENDEDKEY (0x0001) in Windows SendInput
EXTENDED_KEYS = {0x48, 0x50, 0x4B, 0x4D}

# Mapping to pynput keys for 100% web browser (Chrome, Edge, Firefox) & WebGL compatibility
if _pynput_kb is not None:
    PYNPUT_MAP = {
        0x48: Key.up,
        0x50: Key.down,
        0x4B: Key.left,
        0x4D: Key.right,
        0x39: Key.space,
        0x2A: Key.shift,
        0x11: 'w',
        0x1F: 's',
        0x1E: 'a',
        0x20: 'd',
    }
else:
    PYNPUT_MAP = {}

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

def PressKey(hexKeyCode):
    """
    Sends key press across both pynput (web games / browsers) and SendInput (desktop/DirectX games).
    """
    # 1. Dispatch via pynput for web games in browser (Chrome/Edge/Firefox)
    if _pynput_kb is not None:
        pk = PYNPUT_MAP.get(hexKeyCode)
        if pk is not None:
            try:
                _pynput_kb.press(pk)
            except Exception:
                pass

    # 2. Dispatch via native Windows SendInput with mapped virtual key code
    try:
        vk = user32.MapVirtualKeyA(hexKeyCode, MAPVK_VSC_TO_VK)
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        flags = 0x0008  # KEYEVENTF_SCANCODE
        if hexKeyCode in EXTENDED_KEYS:
            flags |= 0x0001  # KEYEVENTF_EXTENDEDKEY
        ii_.ki = KeyBdInput(vk, hexKeyCode, flags, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    except Exception:
        pass

def ReleaseKey(hexKeyCode):
    """
    Sends key release across both pynput (web games / browsers) and SendInput (desktop/DirectX games).
    """
    # 1. Release via pynput
    if _pynput_kb is not None:
        pk = PYNPUT_MAP.get(hexKeyCode)
        if pk is not None:
            try:
                _pynput_kb.release(pk)
            except Exception:
                pass

    # 2. Release via native Windows SendInput
    try:
        vk = user32.MapVirtualKeyA(hexKeyCode, MAPVK_VSC_TO_VK)
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        flags = 0x0008 | 0x0002  # KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP
        if hexKeyCode in EXTENDED_KEYS:
            flags |= 0x0001  # KEYEVENTF_EXTENDEDKEY
        ii_.ki = KeyBdInput(vk, hexKeyCode, flags, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    except Exception:
        pass

if __name__ == '__main__':
    while True:
        PressKey(0x11)
        time.sleep(1)
        ReleaseKey(0x11)
        time.sleep(1)