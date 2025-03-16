import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import logging
import sys

# 로그 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_base_path():
    """실행 환경에 따라 경로를 반환"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)  # 바이너리 실행 시
    return os.path.dirname(os.path.abspath(__file__))  # VS Code 또는 스크립트 실행 시

OUTPUT_DIR = os.path.join(get_base_path(), "split_sprites")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def open_files():
    """여러 개의 이미지 파일 선택"""
    file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_paths:
        return
    for file_path in file_paths:
        process_image(file_path)

def process_image(file_path):
    """이미지를 로드하고 개별 스프라이트로 분리"""
    try:
        logging.info(f"Processing file: {file_path}")
        pil_img = Image.open(file_path).convert("RGBA")
        img_cv = np.array(pil_img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGRA)

        detect_and_save_sprites(img_cv, file_path)
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        messagebox.showerror("Error", f"Failed to process {os.path.basename(file_path)}")

def detect_and_save_sprites(img_cv, file_path):
    """스프라이트 감지 후 저장"""
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if img_cv.shape[2] == 3 else img_cv[:, :, 3]
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filename = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(save_path, exist_ok=True)
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        sprite = img_cv[y:y+h, x:x+w]
        sprite_filename = os.path.join(save_path, f"sprite_{i}.png")
        cv2.imwrite(sprite_filename, sprite)
        logging.info(f"Saved: {sprite_filename}")
    
    messagebox.showinfo("Success", f"Sprites saved in {save_path}/")

# GUI 설정
root = tk.Tk()
root.title("Sprite Sheet Auto Splitter")

btn_open = tk.Button(root, text="Open Images", command=open_files)
btn_open.pack()

canvas = tk.Canvas(root)
canvas.pack()

root.mainloop()
