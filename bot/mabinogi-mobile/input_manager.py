import time
import win32gui
import win32con
import win32api
import win32process
import ctypes
from ctypes import wintypes
import pyautogui
from logger_helper import get_logger

logger = get_logger()

# 지원되는 특수 키에 대한 가상 키 코드 매핑
KEY_MAP = {
    "space": win32con.VK_SPACE,
    "enter": win32con.VK_RETURN,
    "esc": win32con.VK_ESCAPE,
    "left": win32con.VK_LEFT,
    "up": win32con.VK_UP,
    "right": win32con.VK_RIGHT,
    "down": win32con.VK_DOWN,
}


def _get_vk_code(key: str):
    """문자 또는 키 이름을 가상 키 코드로 변환"""
    key = key.lower()
    if len(key) == 1:
        return ord(key.upper())
    return KEY_MAP.get(key)

class InputManager:
    """다른 프로세스에 입력을 전달하는 매니저 클래스"""
    
    def __init__(self, target_window_title="Mabinogi Mobile"):
        self.target_window_title = target_window_title
        self.target_hwnd = None
        self._find_target_window()
        
    def _find_target_window(self):
        """타겟 윈도우 핸들 찾기"""
        self.target_hwnd = win32gui.FindWindow(None, self.target_window_title)
        if self.target_hwnd == 0:
            logger.error(f"윈도우를 찾을 수 없습니다: {self.target_window_title}")
            return False
        logger.info(f"타겟 윈도우 찾음: {self.target_window_title} (HWND: {self.target_hwnd})")
        return True
    
    def _get_window_info(self):
        """윈도우 정보 가져오기"""
        if not self.target_hwnd:
            return None
            
        rect = win32gui.GetWindowRect(self.target_hwnd)
        client_rect = win32gui.GetClientRect(self.target_hwnd)
        
        return {
            'hwnd': self.target_hwnd,
            'window_rect': rect,
            'client_rect': client_rect,
            'left': rect[0],
            'top': rect[1],
            'right': rect[2],
            'bottom': rect[3],
            'width': rect[2] - rect[0],
            'height': rect[3] - rect[1],
            'client_width': client_rect[2],
            'client_height': client_rect[3]
        }
    
    def method1_pyautogui_with_window_focus(self, x, y, button='left'):
        """방법 1: pyautogui + 윈도우 포커스"""
        if not self.target_hwnd:
            return False
            
        try:
            # 윈도우를 포그라운드로 가져오기
            win32gui.SetForegroundWindow(self.target_hwnd)
            time.sleep(0.1)  # 포커스 대기
            
            # pyautogui로 클릭
            pyautogui.click(x, y, button=button)
            logger.debug(f"pyautogui 클릭: ({x}, {y}) - {button}")
            return True
            
        except Exception as e:
            logger.error(f"pyautogui 클릭 실패: {e}")
            return False
    
    def method2_win32api_sendinput(self, x, y, button='left'):
        """방법 2: win32api SendInput 사용"""
        if not self.target_hwnd:
            return False
            
        try:
            # 윈도우를 포그라운드로 가져오기
            win32gui.SetForegroundWindow(self.target_hwnd)
            time.sleep(0.1)
            
            # 마우스 이벤트 구조체 정의
            class MOUSEINPUT(ctypes.Structure):
                _fields_ = [
                    ("dx", wintypes.LONG),
                    ("dy", wintypes.LONG),
                    ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))
                ]
            
            class INPUT(ctypes.Structure):
                _fields_ = [
                    ("type", wintypes.DWORD),
                    ("mi", MOUSEINPUT)
                ]
            
            # 마우스 이동
            win32api.SetCursorPos((x, y))
            time.sleep(0.05)
            
            # 마우스 클릭
            if button == 'left':
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.05)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            elif button == 'right':
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                time.sleep(0.05)
                win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                
            logger.debug(f"win32api 클릭: ({x}, {y}) - {button}")
            return True
            
        except Exception as e:
            logger.error(f"win32api 클릭 실패: {e}")
            return False
    
    def method3_postmessage(self, x, y, button='left'):
        """방법 3: PostMessage 사용 (윈도우 메시지 직접 전송)"""
        if not self.target_hwnd:
            return False
            
        try:
            # 클라이언트 좌표로 변환
            window_info = self._get_window_info()
            if not window_info:
                return False
                
            client_x = x - window_info['left']
            client_y = y - window_info['top']
            
            # LPARAM 생성 (좌표 정보)
            lparam = win32api.MAKELONG(client_x, client_y)
            
            if button == 'left':
                # 마우스 다운
                win32gui.PostMessage(self.target_hwnd, win32con.WM_LBUTTONDOWN, 
                                   win32con.MK_LBUTTON, lparam)
                time.sleep(0.05)
                # 마우스 업
                win32gui.PostMessage(self.target_hwnd, win32con.WM_LBUTTONUP, 
                                   0, lparam)
            elif button == 'right':
                win32gui.PostMessage(self.target_hwnd, win32con.WM_RBUTTONDOWN, 
                                   win32con.MK_RBUTTON, lparam)
                time.sleep(0.05)
                win32gui.PostMessage(self.target_hwnd, win32con.WM_RBUTTONUP, 
                                   0, lparam)
                
            logger.debug(f"PostMessage 클릭: ({x}, {y}) -> ({client_x}, {client_y}) - {button}")
            return True
            
        except Exception as e:
            logger.error(f"PostMessage 클릭 실패: {e}")
            return False
    
    def method4_sendmessage(self, x, y, button='left'):
        """방법 4: SendMessage 사용 (동기식 메시지 전송)"""
        if not self.target_hwnd:
            return False
            
        try:
            # 클라이언트 좌표로 변환
            window_info = self._get_window_info()
            if not window_info:
                return False
                
            client_x = x - window_info['left']
            client_y = y - window_info['top']
            
            # LPARAM 생성
            lparam = win32api.MAKELONG(client_x, client_y)
            
            if button == 'left':
                # 마우스 다운
                win32gui.SendMessage(self.target_hwnd, win32con.WM_LBUTTONDOWN, 
                                   win32con.MK_LBUTTON, lparam)
                time.sleep(0.05)
                # 마우스 업
                win32gui.SendMessage(self.target_hwnd, win32con.WM_LBUTTONUP, 
                                   0, lparam)
            elif button == 'right':
                win32gui.SendMessage(self.target_hwnd, win32con.WM_RBUTTONDOWN, 
                                   win32con.MK_RBUTTON, lparam)
                time.sleep(0.05)
                win32gui.SendMessage(self.target_hwnd, win32con.WM_RBUTTONUP, 
                                   0, lparam)
                
            logger.debug(f"SendMessage 클릭: ({x}, {y}) -> ({client_x}, {client_y}) - {button}")
            return True
            
        except Exception as e:
            logger.error(f"SendMessage 클릭 실패: {e}")
            return False
    
    def send_key(self, key, method='pyautogui'):
        """키보드 입력 전송"""
        if not self.target_hwnd:
            return False

        try:
            if method == 'pyautogui':
                # 윈도우 포커스 후 키 전송
                win32gui.SetForegroundWindow(self.target_hwnd)
                time.sleep(0.1)
                pyautogui.press(key)
                logger.debug(f"pyautogui 키 전송: {key}")

            elif method == 'postmessage':
                vk_code = _get_vk_code(key)
                if vk_code is None:
                    logger.error(f"지원하지 않는 키: {key}")
                    return False
                # PostMessage로 키 이벤트 전송
                win32gui.PostMessage(self.target_hwnd, win32con.WM_KEYDOWN, vk_code, 0)
                time.sleep(0.05)
                win32gui.PostMessage(self.target_hwnd, win32con.WM_KEYUP, vk_code, 0)
                logger.debug(f"PostMessage 키 전송: {key}")

            elif method == 'sendmessage':
                vk_code = _get_vk_code(key)
                if vk_code is None:
                    logger.error(f"지원하지 않는 키: {key}")
                    return False
                # SendMessage로 키 이벤트 전송
                win32gui.SendMessage(self.target_hwnd, win32con.WM_KEYDOWN, vk_code, 0)
                time.sleep(0.05)
                win32gui.SendMessage(self.target_hwnd, win32con.WM_KEYUP, vk_code, 0)
                logger.debug(f"SendMessage 키 전송: {key}")

            else:
                logger.error(f"지원하지 않는 메서드: {method}")
                return False

            return True

        except Exception as e:
            logger.error(f"키 전송 실패: {e}")
            return False
    
    def click(self, x, y, button='left', method='pyautogui'):
        """통합 클릭 메서드"""
        methods = {
            'pyautogui': self.method1_pyautogui_with_window_focus,
            'win32api': self.method2_win32api_sendinput,
            'postmessage': self.method3_postmessage,
            'sendmessage': self.method4_sendmessage
        }
        
        if method not in methods:
            logger.error(f"지원하지 않는 메서드: {method}")
            return False
            
        return methods[method](x, y, button)
    
    def test_all_methods(self, x, y):
        """모든 메서드 테스트"""
        logger.info("=== 입력 메서드 테스트 시작 ===")
        
        methods = [
            ('pyautogui', self.method1_pyautogui_with_window_focus),
            ('win32api', self.method2_win32api_sendinput),
            ('postmessage', self.method3_postmessage),
            ('sendmessage', self.method4_sendmessage)
        ]
        
        for name, method in methods:
            logger.info(f"테스트 중: {name}")
            try:
                result = method(x, y, 'left')
                logger.info(f"{name}: {'성공' if result else '실패'}")
                time.sleep(1)  # 테스트 간격
            except Exception as e:
                logger.error(f"{name} 테스트 실패: {e}")
        
        logger.info("=== 입력 메서드 테스트 완료 ===") 
