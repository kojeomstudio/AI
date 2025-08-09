import time
import json
import sys
import os
import signal
from typing import Dict, List

from ultralytics import YOLO

from ui.object import *
from ui.action import *
from ui.base.element import *
from logger_helper import get_logger
from utils.capture import *
from input_manager import InputManager
from action_processor import ActionProcessor

logger = get_logger()
config = None
running = True

def signal_handler(signum, frame):
    """시그널 핸들러 - 프로그램 종료"""
    global running
    logger.info("종료 신호 수신, 프로그램을 종료합니다...")
    running = False

def get_file_path(in_origin):
    """실행 환경에 따라 상대 경로 처리"""
    base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, str(in_origin))

def load_config(path="config.json"):
    """설정 파일 로드"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}")
        return {}

def match_elements(results, elements):
    """모든 요소와 YOLO 결과 매칭"""
    matched = {}
    for element in elements:
        is_match, pos = element.match(results)
        if is_match:
            logger.debug(f"감지됨: {element.get_type().name} at {pos}")
            matched[element.get_type()] = (element, pos)
        else:
            logger.debug(f"미감지: {element.get_type().name}")
    return matched

def create_elements():
    """YOLO 요소 리스트 생성"""
    return [
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

def main_loop_improved(model, elements, action_processor, tick=0.5):
    """개선된 메인 루프"""
    global running
    
    try:
        while running:
            # 타겟 프로세스 감시
            if not action_processor.input_manager.monitor_process():
                logger.warning("타겟 프로세스를 찾지 못했습니다. 5초 후 재시도...")
                time.sleep(5)
                continue

            # 화면 캡처
            screen_np = get_game_window_image(config["window_title"])
            if screen_np is None:
                logger.warning("게임 창을 찾을 수 없습니다. 5초 후 재시도...")
                time.sleep(5)
                continue

            # YOLO 예측
            logger.debug("YOLO 예측 수행")
            results = model.predict(screen_np, conf=0.5, verbose=False)
            
            # 요소 매칭
            matched = match_elements(results, elements)
            
            # 액션 처리
            if matched:
                action_processor.process_detected_elements(matched)
            else:
                logger.debug("감지된 요소 없음")
            
            # 대기
            time.sleep(tick)
            
    except KeyboardInterrupt:
        logger.info("[EXIT] 매크로 종료됨")
    except Exception as e:
        logger.error(f"메인 루프 오류: {e}")

def test_mode(model, elements, action_processor):
    """테스트 모드 - 입력 메서드 테스트"""
    logger.info("=== 테스트 모드 시작 ===")
    
    # 화면 중앙 좌표 계산
    screen_np = get_game_window_image(config["window_title"])
    if screen_np is not None:
        height, width = screen_np.shape[:2]
        center_x = width // 2
        center_y = height // 2
        
        logger.info(f"화면 크기: {width}x{height}, 중앙: ({center_x}, {center_y})")
        
        # 입력 메서드 테스트
        action_processor.test_input_methods(center_x, center_y)
    else:
        logger.error("게임 창을 찾을 수 없어 테스트를 진행할 수 없습니다.")

def print_stats(action_processor):
    """통계 출력"""
    stats = action_processor.get_action_stats()
    logger.info("=== 매크로 실행 통계 ===")
    logger.info(f"총 실행된 액션 수: {stats['total_actions']}")
    
    if stats['action_counts']:
        logger.info("액션별 실행 횟수:")
        for action, count in stats['action_counts'].items():
            logger.info(f"  {action}: {count}회")

if __name__ == "__main__":
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 설정 로드
    config = load_config(get_file_path("./config/config.json"))
    if not config:
        logger.error("설정 파일을 로드할 수 없습니다.")
        sys.exit(1)
    
    # YOLO 모델 로드
    model_path = get_file_path("ml/training_output/mabinogi_model/weights/best.pt")
    if not os.path.exists(model_path):
        logger.error(f"YOLO 모델 파일을 찾을 수 없습니다: {model_path}")
        sys.exit(1)
    
    model = YOLO(model_path)
    logger.info(f"YOLO 모델 로드 완료: {model_path}")
    
    # 요소 리스트 생성
    elements = create_elements()
    logger.info(f"YOLO 요소 {len(elements)}개 로드 완료")
    
    # 입력 매니저 초기화
    input_manager = InputManager(config["window_title"])
    if not input_manager.target_hwnd:
        logger.error("타겟 윈도우를 찾을 수 없습니다.")
        sys.exit(1)
    
    # 액션 프로세서 초기화
    action_config_path = get_file_path("./config/action_config.json")
    action_processor = ActionProcessor(action_config_path, input_manager)
    
    # 명령행 인수 처리
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_mode(model, elements, action_processor)
    else:
        logger.info("[START] 개선된 YOLO 매크로 실행 중...")
        logger.info(f"타겟 윈도우: {config['window_title']}")
        logger.info(f"틱 간격: {config.get('tick_interval', 0.5)}초")
        
        try:
            main_loop_improved(
                model, 
                elements, 
                action_processor, 
                tick=config.get('tick_interval', 0.5)
            )
        finally:
            print_stats(action_processor)
            logger.info("[END] 매크로 종료") 
