import os, sys, json
import cv2
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from datetime import datetime
from pynput import keyboard
from pynput.keyboard import Key

from capture import get_window_rect, capture_window  # 사용자 정의 캡처 함수

# ----------------------------
# 설정 및 상수
# ----------------------------

CONFIG_PATH = "./config/config.json"
OUTPUT_DIR_NAME = "screenshots"
HOTKEY = Key.f8  # 전역 단축키
WINDOW_TITLE_DEFAULT = "마비노기모바일"  # fallback

# ----------------------------
# 도우미 함수
# ----------------------------

def get_file_path(rel_path):
    base = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
    return os.path.join(base, rel_path)

def load_config(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        messagebox.showerror("설정 오류", f"설정 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

def log(message: str):
    print(message)
    log_var.set(message)

# ----------------------------
# 스크린샷 캡처
# ----------------------------

def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def capture_and_save():
    global capture_count
    hwnd, rect = get_window_rect(window_title)
    if hwnd is None:
        log(f"[오류] 창 '{window_title}' 을(를) 찾을 수 없습니다.")
        return

    img = capture_window(hwnd, rect)
    capture_count += 1
    filename = f"screenshot_{capture_count:03}.png"
    cv2.imwrite(str(output_dir / filename), img)
    log(f"[{capture_count}] 저장됨: {filename}")

# ----------------------------
# 전역 단축키 리스너
# ----------------------------

def on_press(key):
    if key == HOTKEY:
        capture_and_save()

def start_hotkey_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

# ----------------------------
# GUI 구성
# ----------------------------

def create_gui():
    root.title("스크린샷 수집기")
    root.geometry("400x160")
    root.resizable(False, False)

    tk.Label(root, text="🎯 [F8] 키를 누르면 게임 창을 캡처합니다.", font=("Arial", 12)).pack(pady=10)
    tk.Label(root, text=f"📁 저장 경로: {output_dir.resolve()}", fg="gray").pack()
    tk.Label(root, textvariable=log_var, fg="blue").pack(pady=10)

    root.mainloop()

# ----------------------------
# 실행
# ----------------------------

if __name__ == "__main__":
    config = load_config(get_file_path(CONFIG_PATH))
    window_title = config.get("window_title", WINDOW_TITLE_DEFAULT)

    output_dir = Path(get_file_path(OUTPUT_DIR_NAME))
    ensure_output_dir(output_dir)

    capture_count = 0
    root = tk.Tk()
    log_var = tk.StringVar()

    start_hotkey_listener()
    create_gui()
