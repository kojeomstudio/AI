import time
import json
import sys
import os

from ultralytics import YOLO

from ui.object import *
from ui.action import *
from ui.base.element import *
from logger_helper import get_logger
from utils.capture import *
from input_manager import InputManager

logger = get_logger()
config = None


def get_file_path(in_origin):
    """실행 환경에 따라 상대 경로 처리"""
    base_dir = (
        os.path.dirname(sys.executable)
        if getattr(sys, "frozen", False)
        else os.path.dirname(os.path.abspath(__file__))
    )
    return os.path.join(base_dir, str(in_origin))


def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_elements(results, elements):
    """모든 요소와 YOLO 결과 매칭"""
    matched = {}
    for element in elements:
        is_match, pos = element.match(results)
        if is_match:
            matched[element.get_type()] = (element, pos)
    return matched


def process_logic(matched, input_manager):
    """매칭된 요소 기반 동작 처리"""
    # COMPASS 또는 WORKING 상태면 아무것도 하지 않음
    if ElementType.UI_COMPASS in matched:
        logger.debug("대기 상태이므로 동작하지 않음")
        return False
    if ElementType.UI_WORKING in matched:
        logger.debug("작업 상태이므로 동작하지 않음")
        return False

    # 채굴 조건: 채굴 UI + 광맥 중 하나
    if ElementType.UI_MINING in matched:
        if ElementType.COAL_VEIN in matched or ElementType.IRON_VEIN in matched:
            element, pos = matched[ElementType.UI_MINING]
            logger.info("→ 채굴 조건 만족, 채굴 실행")
            element.action(pos)
            x1, y1, x2, y2 = pos
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_manager.click(cx, cy)
            input_manager.send_key("space")
            return True
        else:
            logger.debug("채굴 UI 감지됨, 그러나 광맥 없음")

    # 벌채 조건: 벌채 UI (추후 나무 노드 존재 여부도 체크 가능)
    elif ElementType.UI_FELLING in matched:
        if ElementType.TREE in matched:
            element, pos = matched[ElementType.UI_FELLING]
            logger.info("→ 벌채 조건 만족, 벌채 실행")
            element.action(pos)
            x1, y1, x2, y2 = pos
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_manager.click(cx, cy)
            input_manager.send_key("space")
            return True
        else:
            logger.debug("벌채 UI 감지됨, 그러나 나무 없음")

    return False


def main_loop(model, elements, input_manager, tick=0.5):
    try:
        while True:
            screen_np = get_game_window_image(config["window_title"])
            results = model.predict(screen_np, conf=0.5, verbose=False)
            matched = match_elements(results, elements)

            if process_logic(matched, input_manager):
                time.sleep(1)
            else:
                logger.debug("처리된 액션 없음")

            time.sleep(tick)

    except KeyboardInterrupt:
        logger.info("[EXIT] 매크로 종료됨")


if __name__ == "__main__":
    config = load_config(get_file_path("./config/config.json"))

    model_path = get_file_path("ml/training_output/mabinogi_model/weights/best.pt")
    model = YOLO(model_path)

    elements = [
        CoalVeinNode(ElementType.COAL_VEIN, class_id=0),
        IronVeinNode(ElementType.IRON_VEIN, class_id=7),
        NormalVeinNode(ElementType.NORMAL_VEIN, class_id=8),
        TreeNode(ElementType.TREE, class_id=9),
        UI_Attack(ElementType.UI_ATTACK, class_id=1),
        UI_Inventory(ElementType.UI_INVENTORY, class_id=2),
        UI_Riding(ElementType.UI_RIDING, class_id=3),
        UI_Riding_Out(ElementType.UI_RIDING_OUT, class_id=11),
        UI_Mining(ElementType.UI_MINING, class_id=4),
        UI_Craft(ElementType.UI_CRAFT, class_id=5),
        UI_Compass(ElementType.UI_COMPASS, class_id=6),
        UI_Felling(ElementType.UI_FELLING, class_id=12),
        UI_Working(ElementType.UI_WORKING, class_id=10),
        UI_Wing(ElementType.UI_WING, class_id=13),
    ]

    logger.info("[START] YOLO 매크로 실행 중...")
    input_manager = InputManager(config["window_title"])
    main_loop(model, elements, input_manager, tick=config.get("tick", 1.0))
