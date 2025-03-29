import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime
from rag_handler import send_query, get_font_config
from logger import get_logger

logger = get_logger()

class RAGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG 클라이언트 (ver1.0.0)")
        self.geometry("1280x720")
        self.minsize(800, 600)
        self.configure(bg="#f2f2f2")

        self.font_family, self.font_size = get_font_config()

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=(self.font_family, self.font_size), padding=6)
        style.configure("TEntry", font=(self.font_family, self.font_size))

        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.query_input = ttk.Entry(self)
        self.query_input.grid(row=0, column=0, padx=10, pady=(20, 5), sticky="ew")
        self.query_input.bind("<Return>", lambda event: self.on_send())
        self.query_input.focus_set()

        self.send_button = ttk.Button(self, text="질문하기", command=self.on_send)
        self.send_button.grid(row=0, column=1, padx=10, pady=(20, 5), sticky="e")

        self.loading_label = tk.Label(self, text="응답을 기다리는 중입니다...", font=(self.font_family, self.font_size - 1), bg="#f2f2f2", fg="gray")
        self.loading_label.grid(row=1, column=0, columnspan=2, pady=(0, 5))
        self.loading_label.grid_remove()

        # 응답 출력 영역
        response_label = tk.Label(self, text="응답 결과", font=(self.font_family, self.font_size, "bold"), bg="#f2f2f2")
        response_label.grid(row=2, column=0, columnspan=2, padx=10, sticky="w")

        text_frame = ttk.Frame(self)
        text_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="nsew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        self.response_output = tk.Text(text_frame, wrap="word", state='disabled', font=(self.font_family, self.font_size), bg="white")
        self.response_output.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.response_output.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.response_output.config(yscrollcommand=scrollbar.set)

        # 로그 출력 영역
        log_label = tk.Label(self, text="로그 기록", font=(self.font_family, self.font_size, "bold"), bg="#f2f2f2")
        log_label.grid(row=4, column=0, columnspan=2, padx=10, pady=(5, 0), sticky="w")

        self.log_listbox = tk.Listbox(self, height=8, font=(self.font_family, self.font_size - 1))
        self.log_listbox.grid(row=5, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")
        self.grid_rowconfigure(5, weight=0)

        with open("rag_ui_log.txt", "w", encoding="utf-8") as f:
            f.write("[시작] 로그 기록 시작\n")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_send(self):
        query = self.query_input.get().strip()
        if not query:
            messagebox.showwarning("입력 오류", "질문을 입력하세요.")
            return

        logger.info("[사용자 입력] 질문 전송 시작")
        self.query_input.configure(state='disabled')
        self.send_button.configure(state='disabled')
        self.clear_response()
        self.loading_label.grid()
        self.log("[시스템] 응답 대기 시작")

        threading.Thread(target=self.handle_query, args=(query,), daemon=True).start()

    def handle_query(self, query):
        try:
            for chunk in send_query(query):
                self.after(0, self.append_response, chunk)
        except Exception as e:
            self.after(0, self.append_response, f"[오류 발생] {str(e)}")
            self.after(0, self.log, f"[오류] {str(e)}")
        finally:
            self.after(0, self.query_input.configure, {'state': 'normal'})
            self.after(0, self.send_button.configure, {'state': 'normal'})
            self.after(0, self.loading_label.grid_remove)
            self.after(0, self.log, "[시스템] 응답 수신 완료")

    def append_response(self, text):
        self.response_output.configure(state='normal')
        self.response_output.insert(tk.END, text)
        self.response_output.see(tk.END)
        self.response_output.configure(state='disabled')

    def clear_response(self):
        self.response_output.configure(state='normal')
        self.response_output.delete("1.0", tk.END)
        self.response_output.configure(state='disabled')

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        self.log_listbox.insert(tk.END, full_message)
        self.log_listbox.yview_moveto(1)
        logger.info(full_message)
        with open("rag_ui_log.txt", "a", encoding="utf-8") as f:
            f.write(full_message + "\n")

    def on_close(self):
        self.log("[시스템] 프로그램 종료")
        self.destroy()

if __name__ == "__main__":
    app = RAGUI()
    app.mainloop()
