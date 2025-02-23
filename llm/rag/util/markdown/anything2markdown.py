import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import asyncio
import threading
import os
from markitdown import MarkItDown

class MarkdownConverterUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Markdown Converter")
        self.root.geometry("600x500")
        self.file_paths = []
        
        self.label = tk.Label(root, text="파일을 선택하세요:")
        self.label.pack(pady=10)
        
        self.listbox = tk.Listbox(root, height=10, width=70)
        self.listbox.pack(pady=5)
        
        self.select_button = tk.Button(root, text="파일 선택", command=self.select_files)
        self.select_button.pack(pady=5)
        
        self.convert_button = tk.Button(root, text="변환 시작", command=self.start_conversion, state=tk.DISABLED)
        self.convert_button.pack(pady=5)
        
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=10)
        
        self.status_label = tk.Label(root, text="")
        self.status_label.pack(pady=10)
        
        self.text_output = tk.Text(root, wrap=tk.WORD, height=10, width=70)
        self.text_output.pack(pady=10)
    
    def select_files(self):
        file_types = [("Supported Files", "*.pdf;*.pptx;*.docx;*.xlsx;*.jpg;*.png;*.mp3;*.html;*.csv;*.json;*.xml"),
                      ("PDF", "*.pdf"), ("PowerPoint", "*.pptx"), ("Word", "*.docx"), ("Excel", "*.xlsx"),
                      ("Images", "*.jpg;*.png"), ("Audio", "*.mp3"), ("HTML", "*.html"),
                      ("Text-Based Formats", "*.csv;*.json;*.xml")]
        self.file_paths = filedialog.askopenfilenames(filetypes=file_types)
        if self.file_paths:
            self.listbox.delete(0, tk.END)
            for file_path in self.file_paths:
                self.listbox.insert(tk.END, file_path)
            self.convert_button.config(state=tk.NORMAL)
    
    def start_conversion(self):
        self.convert_button.config(state=tk.DISABLED)
        self.select_button.config(state=tk.DISABLED)
        self.status_label.config(text="파일 변환 중...")
        self.progress['value'] = 0
        self.progress['maximum'] = len(self.file_paths)
        
        thread = threading.Thread(target=self.run_conversion, daemon=True)
        thread.start()
    
    def run_conversion(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.convert_files())
    
    async def convert_files(self):
        markitdown = MarkItDown()
        self.text_output.delete(1.0, tk.END)
        output_dir = os.path.join(os.getcwd(), "generated")
        os.makedirs(output_dir, exist_ok=True)
        
        for index, file_path in enumerate(self.file_paths):
            try:
                result = await asyncio.to_thread(markitdown.convert, file_path)
                filename = os.path.basename(file_path).rsplit('.', 1)[0] + ".md"
                output_path = os.path.join(output_dir, filename)
                
                with open(output_path, "w", encoding="utf-8") as md_file:
                    md_file.write(result.text_content)
                
                self.text_output.insert(tk.END, f"{file_path}\n--------------------\n{result.text_content}\n\n")
            except Exception as e:
                self.text_output.insert(tk.END, f"Error processing {file_path}: {e}\n")
            finally:
                self.progress['value'] += 1
                self.root.update_idletasks()
        
        self.status_label.config(text="변환 완료")
        self.convert_button.config(state=tk.NORMAL)
        self.select_button.config(state=tk.NORMAL)
        self.listbox.delete(0, tk.END)
        
if __name__ == "__main__":
    root = tk.Tk()
    app = MarkdownConverterUI(root)
    root.mainloop()
