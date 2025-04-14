import tkinter as tk
from tkinter import ttk, messagebox
import threading
from datetime import datetime
from rag_handler import send_query, get_font_config
from logger import get_logger
import random

logger = get_logger()

class RAGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("R2R 클라이언트 (ver1.0.0)")
        self.geometry("1280x720")
        self.minsize(800, 600)
        self.configure(bg="#ffffff")

        self.font_family, self.font_size = get_font_config()

        hintText = [
            "input the text..."
        ]

        self.placeholder = random.choice(hintText)
        self.placeholder_color = 'grey'

        style = ttk.Style()
        style.theme_use('clam')
        self.default_fg_color = style.lookup("TEntry", "foreground")

        style.configure("TButton", font=(self.font_family, self.font_size), padding=8)
        style.configure("TEntry", font=(self.font_family, self.font_size))
        style.configure("TCombobox", font=(self.font_family, self.font_size))
        style.configure("TLabel", font=(self.font_family, self.font_size))

        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # top_frame의 column 구성
        top_frame = tk.Frame(self, bg="#ffffff")
        top_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        top_frame.grid_columnconfigure(0, weight=0)  # 모드 라벨
        top_frame.grid_columnconfigure(1, weight=0)  # 콤보박스
        top_frame.grid_columnconfigure(2, weight=1)  # Entry 입력 필드 확장
        top_frame.grid_columnconfigure(3, weight=0)  # 질문 버튼

        # 대화 초기화 버튼 (agent 모드에서만 활성화)
        self.clear_button = ttk.Button(self, text="대화 초기화", command=self.clear_response)
        self.clear_button.grid(row=1, column=1, padx=(0, 20), pady=(5, 10), sticky="e")
        self.clear_button.configure(state='disabled')  # 기본은 비활성화

        # 쿼리 모드 콤보박스
        mode_label = tk.Label(top_frame, text="모드", font=(self.font_family, self.font_size), bg="#ffffff")
        mode_label.grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.query_mode = tk.StringVar(value="rag")
        self.mode_selector = ttk.Combobox(top_frame, textvariable=self.query_mode, values=["rag", "agent"], state="readonly", width=10)
        self.mode_selector.grid(row=0, column=1, sticky="w")
        self.mode_selector.bind("<<ComboboxSelected>>", self.on_mode_change)

        # ✅ 입력 필드 - weight=1에 따라 가장 넓게 확장됨
        self.query_input = ttk.Entry(top_frame)
        self.query_input.grid(row=0, column=2, sticky="ew", padx=(20, 10))
        self.query_input.bind("<Return>", lambda event: self.on_send())

        # 질문하기 버튼
        self.send_button = ttk.Button(top_frame, text="질문하기", command=self.on_send)
        self.send_button.grid(row=0, column=3)

        # 로딩 메시지
        self.loading_label = tk.Label(self, text="⏳ 응답을 기다리는 중입니다...", font=(self.font_family, self.font_size - 1), bg="#ffffff", fg="gray")
        self.loading_label.grid(row=1, column=0, pady=(5, 10))
        self.loading_label.grid_remove()

        # 응답 결과 영역
        response_label = tk.Label(self, text="📘 응답 결과", font=(self.font_family, self.font_size, "bold"), bg="#ffffff")
        response_label.grid(row=2, column=0, padx=20, pady=(5, 5), sticky="w")

        text_frame = ttk.Frame(self)
        text_frame.grid(row=3, column=0, padx=20, pady=(0, 5), sticky="nsew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        self.response_output = tk.Text(text_frame, wrap="word", state='disabled', font=(self.font_family, self.font_size), bg="#fefefe")
        self.response_output.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.response_output.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.response_output.config(yscrollcommand=scrollbar.set)

        # 로그 출력
        log_label = tk.Label(self, text="📝 로그", font=(self.font_family, self.font_size, "bold"), bg="#ffffff")
        log_label.grid(row=4, column=0, padx=20, pady=(10, 5), sticky="w")

        self.log_listbox = tk.Listbox(self, height=8, font=(self.font_family, self.font_size - 1))
        self.log_listbox.grid(row=5, column=0, padx=20, pady=(0, 20), sticky="nsew")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_mode_change(self, event):
        selected_mode = self.query_mode.get()
        if selected_mode == "agent":
            self.clear_button.configure(state='normal')
        else:
            self.clear_button.configure(state='disabled')
            self.clear_response()

    def on_send(self):
        query = self.query_input.get().strip()
        if not query:
            messagebox.showwarning("입력 오류", "질문을 입력하세요.")
            return

        mode = self.query_mode.get()
        logger.info(f"[사용자 입력] 질문 전송 시작 - 모드: {mode}")
        self.query_input.configure(state='disabled')
        self.send_button.configure(state='disabled')
        self.mode_selector.configure(state='disabled')

        if mode == "rag":
            self.clear_response()
            self.clear_button.configure(state='disabled')
        else:
            self.clear_button.configure(state='normal')

        self.loading_label.grid()
        self.log(f"[시스템] '{mode}' 모드로 응답 대기 시작")

        threading.Thread(target=self.handle_query, args=(query, mode), daemon=True).start()

    def handle_query(self, query, mode):
        try:
            if mode == "agent":
                 self.append_response(f"\n🙋 사용자: {query}\n")

            for chunk in send_query(query, mode=mode):
                self.after(0, self.append_response, chunk)
                
        except Exception as e:
            self.after(0, self.append_response, f"[오류 발생] {str(e)}")
            self.after(0, self.log, f"[오류] {str(e)}")
        finally:
            self.after(0, self.query_input.configure, {'state': 'normal'})
            self.after(0, self.send_button.configure, {'state': 'normal'})
            self.after(0, self.mode_selector.configure, {'state': 'readonly'})
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

    def on_key_press(self, event):
        if self.query_input.get() == self.placeholder:
            self.query_input.delete(0, tk.END)
            self.query_input.configure(foreground=self.default_fg_color)

    def on_focus_out(self, event):
        if not self.query_input.get():
            self.query_input.insert(0, self.placeholder)
            self.query_input.configure(foreground=self.placeholder_color)

    def on_close(self):
        self.log("[시스템] 프로그램 종료")
        self.destroy()

if __name__ == "__main__":
    app = RAGUI()
    app.mainloop()
