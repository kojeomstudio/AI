import os, sys, json
import cv2
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from datetime import datetime
from pynput import keyboard
from pynput.keyboard import Key

from utils.capture import get_window_rect, capture_window
from utils.macro.macro_uitls import get_path

# ----------------------------
# 설정 및 상수
# ----------------------------

CONFIG_PATH = "../../config/config.json"
OUTPUT_DIR_NAME = "./screenshots"
HOTKEY = Key.f8  # 전역 단축키
WINDOW_TITLE_DEFAULT = "마비노기모바일"  # fallback

# ----------------------------
# 도우미 함수
# ----------------------------

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
    global capture_count, prev_prefix

    hwnd, rect = get_window_rect(window_title)
    if hwnd is None:
        log(f"[오류] 창 '{window_title}' 을(를) 찾을 수 없습니다.")
        return

    # 현재 접두사 가져오기 (공백이면 기본값 사용)
    current_prefix = entry_prefix.get().strip() or "screenshot"

    # 접두사가 변경되었으면 시작 번호를 재설정
    if current_prefix != prev_prefix:
        try:
            capture_count = int(entry_start_number.get())
            log(f"[초기화] 접두사 변경 감지 → 넘버링을 {capture_count}부터 시작합니다.")
        except ValueError:
            log("[오류] 시작 번호가 유효하지 않습니다. 0부터 시작합니다.")
            capture_count = 0
        prev_prefix = current_prefix  # 변경된 접두사를 저장

    # 캡처 및 저장
    img = capture_window(hwnd, rect)
    filename = f"{current_prefix}_{capture_count:03}.png"
    cv2.imwrite(str(output_dir / filename), img)
    log(f"[{capture_count}] 저장됨: {filename}")
    capture_count += 1

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
    global entry_prefix, entry_start_number

    root.title("스크린샷 수집기")
    root.geometry("400x220")
    root.resizable(False, False)

    tk.Label(root, text="🎯 [F8] 키를 누르면 게임 창을 캡처합니다.", font=("Arial", 12)).pack(pady=10)

    frame_input = tk.Frame(root)
    frame_input.pack(pady=5)

    tk.Label(frame_input, text="접두사:").grid(row=0, column=0, sticky="e")
    entry_prefix = tk.Entry(frame_input)
    entry_prefix.grid(row=0, column=1, padx=5)

    tk.Label(frame_input, text="시작 번호:").grid(row=1, column=0, sticky="e")
    entry_start_number = tk.Entry(frame_input)
    entry_start_number.grid(row=1, column=1, padx=5)
    entry_start_number.insert(0, "0")

    tk.Label(root, text=f"📁 저장 경로: {output_dir.resolve()}", fg="gray").pack(pady=5)
    tk.Label(root, textvariable=log_var, fg="blue").pack(pady=10)

    root.mainloop()

# ----------------------------
# 실행
# ----------------------------

if __name__ == "__main__":
    config = load_config(get_path(CONFIG_PATH))
    window_title = config.get("window_title", WINDOW_TITLE_DEFAULT)

    output_dir = Path(get_path(OUTPUT_DIR_NAME))
    ensure_output_dir(output_dir)

    # 전역 변수 초기화
    capture_count = 0
    prev_prefix = None

    root = tk.Tk()
    log_var = tk.StringVar()

    start_hotkey_listener()
    create_gui()
