import json
import time
from typing import Dict, List, Tuple, Optional
from input_manager import InputManager
from logger_helper import get_logger
from ui.base.element import ElementType

logger = get_logger()

class ActionProcessor:
    """개선된 액션 처리 시스템"""
    
    def __init__(self, action_config_path: str, input_manager: InputManager):
        self.input_manager = input_manager
        self.action_config = self._load_action_config(action_config_path)
        self.last_action_time = {}  # 액션별 마지막 실행 시간 추적
        
    def _load_action_config(self, config_path: str) -> dict:
        """액션 설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"액션 설정 로드 완료: {config_path}")
            return config
        except Exception as e:
            logger.error(f"액션 설정 로드 실패: {e}")
            return {}
    
    def _check_cooldown(self, action_name: str) -> bool:
        """액션 쿨다운 체크"""
        if action_name not in self.last_action_time:
            return True
            
        delay = self.action_config.get('actions', {}).get(action_name, {}).get('delay', 
                self.action_config.get('default_delay', 0.5))
        
        time_since_last = time.time() - self.last_action_time[action_name]
        return time_since_last >= delay
    
    def _update_action_time(self, action_name: str):
        """액션 실행 시간 업데이트"""
        self.last_action_time[action_name] = time.time()
    
    def _check_conditions(self, detected_elements: List[str], required_conditions: List[str]) -> bool:
        """조건 충족 여부 확인"""
        if not required_conditions:
            return True
            
        detected_set = set(detected_elements)
        required_set = set(required_conditions)
        
        return required_set.issubset(detected_set)
    
    def _execute_action(self, action_name: str, position: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """액션 실행"""
        action_config = self.action_config.get('actions', {}).get(action_name)
        if not action_config:
            logger.warning(f"알 수 없는 액션: {action_name}")
            return False
        
        action_type = action_config.get('type')
        input_method = self.action_config.get('input_method', 'pyautogui')
        
        try:
            if action_type == 'click':
                if position:
                    x1, y1, x2, y2 = position
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    result = self.input_manager.click(center_x, center_y, method=input_method)
                    if result:
                        logger.info(f"클릭 액션 실행: {action_name} ({center_x}, {center_y})")
                        self._update_action_time(action_name)
                        return True
                else:
                    logger.warning(f"클릭 액션에 위치 정보가 없음: {action_name}")
                    
            elif action_type == 'key':
                key = action_config.get('key', 'space')
                result = self.input_manager.send_key(key, method=input_method)
                if result:
                    logger.info(f"키 액션 실행: {action_name} (키: {key})")
                    self._update_action_time(action_name)
                    return True
                    
        except Exception as e:
            logger.error(f"액션 실행 실패 {action_name}: {e}")
            
        return False
    
    def process_detected_elements(self, matched_elements: Dict[ElementType, Tuple]) -> bool:
        """감지된 요소들을 처리하고 적절한 액션 실행"""
        if not matched_elements:
            return False
            
        # 감지된 요소 이름 리스트 생성
        detected_element_names = [element_type.name for element_type in matched_elements.keys()]
        logger.debug(f"감지된 요소: {detected_element_names}")
        
        # 우선순위 규칙 확인
        priority_rules = self.action_config.get('priority_rules', [])
        
        for rule in priority_rules:
            rule_conditions = rule.get('conditions', [])
            rule_action = rule.get('action')
            
            if self._check_conditions(detected_element_names, rule_conditions):
                logger.debug(f"우선순위 규칙 매칭: {rule.get('name')}")
                
                if rule_action == 'wait':
                    logger.info(f"대기 상태: {rule.get('description')}")
                    return True
                    
                elif rule_action in self.action_config.get('actions', {}):
                    # 조건을 만족하는 요소의 위치 찾기
                    target_element_type = None
                    for element_type in matched_elements.keys():
                        if element_type.name == rule_action:
                            target_element_type = element_type
                            break
                    
                    if target_element_type:
                        element, position = matched_elements[target_element_type]
                        if self._check_cooldown(rule_action):
                            return self._execute_action(rule_action, position)
                        else:
                            logger.debug(f"쿨다운 중: {rule_action}")
                            return True
                    else:
                        logger.warning(f"규칙 액션에 해당하는 요소를 찾을 수 없음: {rule_action}")
                        
        # 우선순위 규칙에 매칭되지 않은 경우, 개별 액션 처리
        for element_type, (element, position) in matched_elements.items():
            element_name = element_type.name
            
            # UI 요소는 조건 없이 클릭
            if element_name.startswith('UI_') and element_name not in ['UI_WORKING', 'UI_COMPASS']:
                if self._check_cooldown(element_name):
                    if self._execute_action(element_name, position):
                        return True
                        
        return False
    
    def test_input_methods(self, x: int, y: int):
        """입력 메서드 테스트"""
        logger.info("=== 입력 메서드 테스트 시작 ===")
        self.input_manager.test_all_methods(x, y)
        
    def get_action_stats(self) -> Dict:
        """액션 실행 통계 반환"""
        stats = {
            'total_actions': len(self.last_action_time),
            'action_counts': {},
            'last_action_times': {}
        }
        
        for action_name, last_time in self.last_action_time.items():
            stats['action_counts'][action_name] = 1  # 간단한 카운트
            stats['last_action_times'][action_name] = last_time
            
        return stats 