from logger_helper import get_logger
import pyautogui
from enum import Enum

logger = get_logger()

class UIElementType(Enum):
    IRON_VEIN = 0
    COAL_VEIN = 1
    FELLING = 2
    MINING = 3
    COMPASS = 4
    WORKING = 5

class UIElement:
    def __init__(self, type : UIElementType, class_id):
        self.type = type
        self.class_id = class_id

    def match(self, results):
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == self.class_id:
                    return True, box.xyxy[0].tolist()
        return False, None

    def action(self, pos):
        x1, y1, x2, y2 = pos
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        pyautogui.click(center_x, center_y)
        logger.debug(f"[ACTION] {self.type.name} 클릭: ({center_x}, {center_y})")

    def get_type(self):
        return self.type
'''
import cv2
import time
from capture import get_window_rect
from logger_helper import get_logger
from ocr_helper import extract_text_from_image

class UIElement:
    def __init__(self, name, template_paths, threshold=0.85, offset=(10, 10),
                 window_title=None, roi_config=None, required_text=None):
        self.name = name
        self.template_paths = template_paths
        self.templates = [cv2.imread(path, 0) for path in self.template_paths]
        self.threshold = threshold
        self.offset = offset
        self.window_title = window_title
        self.roi_config = roi_config or {
            "top_ratio": 0.0, "bottom_ratio": 1.0, "left_ratio": 0.0, "right_ratio": 1.0
        }
        self.required_text = required_text or []

    def match(self, screen_gray, screen_color):
        hwnd, rect = get_window_rect(self.window_title)
        if not hwnd:
            return False, None

        left, top, right, bottom = rect
        width = right - left
        height = bottom - top

        roi_top = int(height * self.roi_config["top_ratio"])
        roi_bottom = int(height * self.roi_config["bottom_ratio"])
        roi_left = int(width * self.roi_config["left_ratio"])
        roi_right = int(width * self.roi_config["right_ratio"])

        roi_gray = screen_gray[roi_top:roi_bottom, roi_left:roi_right]
        roi_color = screen_color[roi_top:roi_bottom, roi_left:roi_right]

        scores = []
        positions = []

        for idx, template in enumerate(self.templates):
            for scale in [1.0, 0.9, 1.1]:
                resized = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                if resized.shape[0] > roi_gray.shape[0] or resized.shape[1] > roi_gray.shape[1]:
                    continue
                result = cv2.matchTemplate(roi_gray, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                logger.debug(f"[MATCH] {self.name} [{idx}] scale {scale:.1f} score: {max_val:.4f}")
                scores.append(max_val)
                positions.append((max_loc[0] + roi_left, max_loc[1] + roi_top))

        if not scores:
            return False, None

        avg_score = sum(scores) / len(scores)
        best_score = max(scores)
        best_pos = positions[scores.index(best_score)]
        logger.debug(f"[MATCH] {self.name} best: {best_score:.4f}, avg: {avg_score:.4f}")

        # OCR 조건 검사
        x, y = best_pos
        w, h = 150, 50  # 검사할 텍스트 영역 크기 (조정 가능)
        ocr_region = screen_color[y - 60:y - 10, x - 40:x + 110]
        ocr_text = extract_text_from_image(ocr_region)

        logger.debug(f"[OCR] {self.name} 텍스트: {ocr_text}")
        text_condition = any(keyword in ocr_text for keyword in self.required_text)

        # 최종 조건 통합
        match_condition = (
            best_score >= self.threshold and
            avg_score >= self.threshold * 0.9 and
            text_condition
        )

        if match_condition:
            logger.info(f"[MATCH+TEXT] {self.name} 인식됨, 텍스트: {ocr_text}")
            return True, best_pos

        cv2.imwrite(f"logs/fail_{self.name}_{int(time.time())}.png", screen_gray)
        return False, None
'''