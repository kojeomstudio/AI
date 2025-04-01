import win32gui
import win32con
import win32ui
import numpy as np
import cv2

def get_window_rect(title: str):
    hwnd = win32gui.FindWindow(None, title)
    if hwnd == 0:
        return None, None

    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    # win32gui.SetForegroundWindow(hwnd)  # 생략 권장
    rect = win32gui.GetWindowRect(hwnd)
    return hwnd, rect

def capture_window(hwnd, rect):
    left, top, right, bottom = rect
    width, height = right - left, bottom - top

    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitmap = win32ui.CreateBitmap()
    saveBitmap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitmap)

    saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

    bmpinfo = saveBitmap.GetInfo()
    bmpstr = saveBitmap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype='uint8').reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    win32gui.DeleteObject(saveBitmap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    return img_gray

def get_game_window_image(window_title: str):
    hwnd, rect = get_window_rect(window_title)
    if hwnd is None or rect is None:
        return None
    return capture_window(hwnd, rect)
