import os
from tkinter import Tk, filedialog, Label, Button, Entry, messagebox, Listbox, Scrollbar, END
import tkinter as tk
from PIL import Image

def resize_images(image_paths, output_folder, width, height, rename):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for idx, image_path in enumerate(image_paths):
        try:
            with Image.open(image_path) as img:
                img_resized = img.resize((width, height))
                ext = os.path.splitext(image_path)[1]
                if rename:
                    new_name = f"image_{idx+1}{ext}"
                else:
                    new_name = os.path.basename(image_path)
                output_path = os.path.join(output_folder, new_name)
                img_resized.save(output_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process {image_path}: {e}")

def select_files():
    files = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
    if files:
        file_list.extend(files)
        update_file_list()

def select_folder():
    folder = filedialog.askdirectory()
    if folder:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                    file_list.append(os.path.join(root, file))
        update_file_list()

def update_file_list():
    listbox.delete(0, END)  # 기존 목록 초기화
    for file in file_list:
        listbox.insert(END, os.path.basename(file))
    file_label["text"] = f"Selected {len(file_list)} files."

def clear_file_list():
    """ 선택한 파일 목록 초기화 """
    file_list.clear()
    update_file_list()

def start_resize():
    try:
        width = int(width_entry.get())
        height = int(height_entry.get())
        output_folder = filedialog.askdirectory(title="Select Output Folder")
        rename_files = rename_var.get()

        if not output_folder:
            return

        if file_list:
            resize_images(file_list, output_folder, width, height, rename_files)
            messagebox.showinfo("Success", "Images resized successfully.")

            # 변환 완료 후 목록 초기화
            clear_file_list()
        else:
            messagebox.showwarning("No Files", "Please select files or folders first.")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid width and height.")

# GUI setup
root = Tk()
root.title("Image Resizer")

Label(root, text="Select images or folders containing images:").pack(pady=5)
Button(root, text="Select Files", command=select_files).pack(pady=5)
Button(root, text="Select Folder", command=select_folder).pack(pady=5)

file_list = []
file_label = Label(root, text="No files selected.")
file_label.pack(pady=5)

# 리스트박스 및 스크롤바 추가
frame = tk.Frame(root)
frame.pack(pady=5)

scrollbar = Scrollbar(frame)
scrollbar.pack(side="right", fill="y")

listbox = Listbox(frame, width=50, height=10, yscrollcommand=scrollbar.set)
listbox.pack(side="left")

scrollbar.config(command=listbox.yview)

Button(root, text="Clear List", command=clear_file_list).pack(pady=5)

Label(root, text="Enter new dimensions:").pack(pady=5)
width_entry = Entry(root, width=10)
width_entry.insert(0, "800")
width_entry.pack(pady=2)
Label(root, text="Width").pack()
height_entry = Entry(root, width=10)
height_entry.insert(0, "600")
height_entry.pack(pady=2)
Label(root, text="Height").pack()

rename_var = tk.BooleanVar()
rename_var.set(True)
Button(root, text="Start Resizing", command=start_resize).pack(pady=10)

root.mainloop()
