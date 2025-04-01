import time
import numpy as np
import sys, os
from pathlib import Path
import cv2
from PIL import ImageGrab

from macro_uitls import get_path

from logger_helper import get_logger
from capture import get_game_window_image
from config_loader import load_config
from ui.vein import CoalNode, IronNode

logger = get_logger()

CLASS_MAP = {
    "CoalNode": CoalNode,
    "IronNode": IronNode
}

def collect_templates_from_dir(dir_path: str) -> list:
    dir = Path(dir_path)
    if not dir.exists():
        raise FileNotFoundError(f"템플릿 폴더가 존재하지 않음: {dir_path}")
    return [str(p) for p in sorted(dir.glob("*.png"))]

def build_ui_elements(config):
    elements = []
    window_title = config.get("window_title")
    roi_config = config.get("roi_config")

    for el in config["elements"]:
        cls = CLASS_MAP.get(el["type"])
        if not cls:
            logger.warning(f"[SKIP] 알 수 없는 타입: {el['type']}")
            continue

        template_paths = collect_templates_from_dir(get_path(el["template_dir"]))
        element = cls(
            name=el["name"],
            template_paths=template_paths,
            threshold=el.get("threshold", 0.85),
            offset=tuple(el.get("offset", [10, 10])),
            window_title=window_title,
            roi_config=roi_config,
            required_text=el.get("required_text", [])
        )
        elements.append(element)
    return elements

def main_loop(ui_elements, config):
    tick = config.get("tick_interval", 0.5)
    delay = config.get("action_delay", 2.0)

    try:
        while True:
            # ✅ 컬러 스크린샷
            screenshot = np.array(ImageGrab.grab())
            screen_color = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
            screen_gray = cv2.cvtColor(screen_color, cv2.COLOR_BGR2GRAY)

            for element in ui_elements:
                matched, pos = element.match(screen_gray, screen_color)
                if matched:
                    logger.info(f"[MATCHED] {element.name} at {pos}")
                    element.action(pos)
                    time.sleep(delay)
                    break

            time.sleep(tick)
    except KeyboardInterrupt:
        logger.info("[EXIT] 매크로 종료됨")

if __name__ == "__main__":
    config = load_config(get_path("./config/config.json"))
    elements = build_ui_elements(config)
    logger.info("[START] 매크로 실행 중...")
    main_loop(elements, config)
