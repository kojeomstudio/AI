import asyncio
import threading
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, MarkdownFormatOption

# Logging 설정
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]  # stdout으로 로그 출력
)

class RedirectStdout:
    def __init__(self, log_callback):
        self.log_callback = log_callback
        self.default_stdout = sys.stdout  # 기본 stdout 저장

    def write(self, message):
        if message.strip():
            self.log_callback(message)
            self.default_stdout.write(message)  # 기본 stdout에도 출력

    def flush(self):
        pass


class LogHandler(logging.Handler):
    def __init__(self, log_callback):
        super().__init__()
        self.log_callback = log_callback

    def emit(self, record):
        log_entry = self.format(record)
        self.log_callback(log_entry)


def get_base_path():
    """스크립트 또는 바이너리 실행 경로를 반환."""
    if getattr(sys, 'frozen', False):  # PyInstaller로 패키징된 경우
        return Path(sys._MEIPASS).resolve()  # 바이너리 내부의 임시 디렉터리
    else:  # 일반 스크립트 실행
        return Path(__file__).resolve().parent

# 실행 경로 기반으로 디렉터리 설정
base_path = get_base_path()
output_path = base_path / "converted_files"
output_path.mkdir(parents=True, exist_ok=True)

print(f"Base path: {base_path}")
print(f"Output directory: {output_path}")

doc_converter = DocumentConverter(
    allowed_formats=[
        InputFormat.PDF,
        InputFormat.IMAGE,
        InputFormat.DOCX,
        InputFormat.HTML,
        InputFormat.PPTX,
        InputFormat.ASCIIDOC,
        InputFormat.MD,
    ],
    format_options={InputFormat.MD: MarkdownFormatOption()},
)


def convert_files(file_paths, update_progress_callback):
    results = []
    for idx, path in enumerate(file_paths):

        output_file_abs_path_str = ""

        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            res = doc_converter.convert_all([Path(path)])

            for conv_result in res:

                output_file_path = output_path.joinpath(f"{conv_result.input.file.stem}.md")
                output_file_abs_path_str = output_file_path.absolute()

                logger.info(f"convert -> output_file_path: {output_file_path.absolute()}")
                with output_file_path.open("w", encoding="utf-8", errors="replace") as fp:
                    writed_bytes = fp.write(conv_result.document.export_to_markdown())
                    logger.info(f"{output_file_path} is writed {writed_bytes} bytes")

                results.append(output_file_path)
                
        except FileNotFoundError as e:
            logger.error(f"File not found: {path} - {e}  output_file_path : {output_file_abs_path_str}")
        except RuntimeError as e:
            logger.error(f"PDF conversion error: {path} - {e}  output_file_path : {output_file_abs_path_str}")
        except Exception as e:
            logger.error(f"Unexpected error converting {path}: {e}, output_file_path : {output_file_abs_path_str}")
        finally:
            update_progress_callback(idx + 1, len(file_paths))
    return results


def start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


class App(tk.Tk):
    def __init__(self, loop):
        super().__init__()
        self.title("Markdown Document Converter")
        self.geometry("800x700")
        self.file_paths = []
        self.loop = loop

        self.loop_thread = threading.Thread(target=start_event_loop, args=(self.loop,), daemon=True)
        self.loop_thread.start()

        self.stop_event = threading.Event()

        self.create_widgets()

        sys.stdout = RedirectStdout(self.log_message)
        sys.stderr = RedirectStdout(self.log_message)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        log_handler = LogHandler(self.log_message)
        log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(log_handler)
        logger.setLevel(logging.INFO)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        tk.Label(self, text="Selected Files", font=("Arial", 12, "bold"), anchor="w").pack(padx=10, pady=5, fill=tk.X)
        self.file_listbox = tk.Listbox(self, height=10, selectmode=tk.SINGLE)
        self.file_listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        tk.Label(self, text="Conversion Log", font=("Arial", 12, "bold"), anchor="w").pack(padx=10, pady=5, fill=tk.X)
        self.log_text = scrolledtext.ScrolledText(self, height=10, state="disabled")
        self.log_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.status_frame = tk.Frame(self, relief=tk.RIDGE, borderwidth=2)
        self.status_frame.pack(padx=10, pady=10, fill=tk.X)

        self.status_label = tk.Label(self.status_frame, text="Idle", font=("Arial", 10, "italic"), anchor="w")
        self.status_label.pack(padx=10, pady=5, fill=tk.X)

        self.progress = ttk.Progressbar(self.status_frame, mode="determinate", maximum=100)
        self.progress.pack(padx=10, pady=5, fill=tk.X)

        button_frame = tk.Frame(self)
        button_frame.pack(pady=10)

        self.add_files_btn = ttk.Button(button_frame, text="Add Files", command=self.add_files)
        self.add_files_btn.pack(side=tk.LEFT, padx=5)

        self.add_folder_btn = ttk.Button(button_frame, text="Add Folder", command=self.add_folder)
        self.add_folder_btn.pack(side=tk.LEFT, padx=5)

        self.convert_btn = ttk.Button(button_frame, text="Convert to Markdown", command=self.start_conversion)
        self.convert_btn.pack(side=tk.LEFT, padx=5)

    def add_files(self):
        files = filedialog.askopenfilenames(title="Select Files")
        if files:
            self.file_paths.extend(files)
            self.update_file_listbox()
            self.log_message(f"Added files: {files}")

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select Folder")
        if folder:
            for root, _, files in os.walk(folder):
                self.file_paths.extend([os.path.join(root, file) for file in files])
            self.update_file_listbox()
            self.log_message(f"Added folder: {folder}")

    def update_file_listbox(self):
        self.file_listbox.delete(0, tk.END)
        for file_path in self.file_paths:
            self.file_listbox.insert(tk.END, file_path)

    def log_message(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    def lock_ui(self):
        self.add_files_btn.config(state=tk.DISABLED)
        self.add_folder_btn.config(state=tk.DISABLED)
        self.convert_btn.config(state=tk.DISABLED)

    def unlock_ui(self):
        self.add_files_btn.config(state=tk.NORMAL)
        self.add_folder_btn.config(state=tk.NORMAL)
        self.convert_btn.config(state=tk.NORMAL)

    def reset_progress(self):
        self.progress["value"] = 0
        self.status_label.config(text="Idle")

    def update_progress(self, current, total):
        percent = int((current / total) * 100)
        self.progress["value"] = percent
        self.status_label.config(text=f"Processing {current}/{total} files...")

    def on_close(self):
        self.log_message("Closing application...")
        self.stop_event.set()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join()
        self.destroy()

    def start_conversion(self):
        if not self.file_paths:
            messagebox.showerror("Error", "No files selected for conversion.")
            return

        self.lock_ui()
        self.progress.config(mode="indeterminate")
        self.progress.start(10)

        try:
            future = asyncio.run_coroutine_threadsafe(self.run_conversion(), self.loop)
            future.add_done_callback(lambda f: self.after(0, self.handle_conversion_result, f))
        except Exception as e:
            self.log_message(f"Exception during future creation: {e}")
            self.unlock_ui()
            self.progress.stop()
            self.progress.config(mode="determinate")

    async def run_conversion(self):
        try:
            with ThreadPoolExecutor() as executor:
                converted_files = await self.loop.run_in_executor(
                    executor,
                    convert_files,
                    self.file_paths,
                    lambda current, total: self.after(0, self.update_progress, current, total),
                )
            self.after(0, self.update_ui, converted_files)
        except Exception as e:
            self.log_message(f"Exception in run_conversion: {e}")
        finally:
            self.log_message("Exiting run_conversion()")

    def handle_conversion_result(self, future):
        try:
            future.result()
            self.log_message("Conversion task completed successfully.")
        except Exception as e:
            self.log_message(f"Error in conversion task: {e}")
            self.status_label.config(text="Error during conversion.")
        finally:
            self.progress.stop()
            self.progress.config(mode="determinate")
            self.unlock_ui()

    def update_ui(self, converted_files):
        self.unlock_ui()
        self.reset_progress()
        if converted_files:
            self.log_message(f"Converted {len(converted_files)} files successfully!")
            messagebox.showinfo("Success", f"Converted {len(converted_files)} files successfully!")
        else:
            self.log_message("No files were converted.")
            messagebox.showerror("Error", "No files were converted.")
        self.file_paths.clear()
        self.update_file_listbox()


def main():
    loop = asyncio.new_event_loop()
    app = App(loop)
    app.mainloop()


if __name__ == "__main__":
    main()
