import json
import time
from typing import Dict, List, Tuple, Optional
from jsonschema import validate, ValidationError

from input_manager import InputManager
from logger_helper import get_logger
from ui.base.element import ElementType

logger = get_logger()

class ActionProcessor:
    """Processes detected elements and executes corresponding actions."""

    ACTION_CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "input_method": {"type": "string"},
            "default_delay": {"type": "number"},
            "actions": {"type": "object"},
            "priority_rules": {"type": "array"},
        },
        "required": ["actions", "priority_rules"],
    }

    def __init__(self, action_config_path: str, input_manager: InputManager):
        self.input_manager = input_manager
        self.action_config = self._load_action_config(action_config_path)
        self.last_action_time: Dict[str, float] = {}
        self.action_counts: Dict[str, int] = {}

    def _load_action_config(self, config_path: str) -> dict:
        """Loads and validates the action configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            validate(instance=config, schema=self.ACTION_CONFIG_SCHEMA)
            logger.info(f"Action configuration loaded and validated: {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Action config file not found: {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse action config file: {e}")
        except ValidationError as e:
            logger.error(f"Action config validation failed: {e.message}")
        except Exception as e:
            logger.error(f"Failed to load action config: {e}")
        return {'actions': {}, 'priority_rules': []}

    def _check_cooldown(self, action_name: str) -> bool:
        """Checks if the cooldown for a given action has passed."""
        if action_name not in self.last_action_time:
            return True
        
        action_info = self.action_config.get('actions', {}).get(action_name, {})
        delay = action_info.get('delay', self.action_config.get('default_delay', 0.5))
        
        return (time.time() - self.last_action_time[action_name]) >= delay

    def _update_action_time(self, action_name: str):
        """Updates the last execution time and count for an action."""
        self.last_action_time[action_name] = time.time()
        self.action_counts[action_name] = self.action_counts.get(action_name, 0) + 1

    def _check_conditions(self, detected_elements: List[ElementType], required_conditions: List[str]) -> bool:
        """Checks if all required conditions are met."""
        if not required_conditions:
            return True
        
        detected_names = {el.name for el in detected_elements}
        return set(required_conditions).issubset(detected_names)

    def _execute_action(self, action_name: str, position: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """Executes a specific action."""
        action_info = self.action_config.get('actions', {}).get(action_name)
        if not action_info:
            logger.warning(f"Action '{action_name}' not found in config.")
            return False

        action_type = action_info.get('type')
        input_method = self.action_config.get('input_method', 'pyautogui')

        try:
            if action_type == 'click' and position:
                x, y = int((position[0] + position[2]) / 2), int((position[1] + position[3]) / 2)
                if self.input_manager.click(x, y, method=input_method):
                    logger.info(f"Executed click action: {action_name} at ({x}, {y})")
                    self._update_action_time(action_name)
                    return True
            elif action_type == 'key':
                key = action_info.get('key', 'space')
                if self.input_manager.send_key(key, method=input_method):
                    logger.info(f"Executed key action: {action_name} (key: {key})")
                    self._update_action_time(action_name)
                    return True
        except Exception as e:
            logger.error(f"Failed to execute action '{action_name}': {e}", exc_info=True)
        
        return False

    def process_detected_elements(self, matched_elements: Dict[ElementType, Tuple]) -> bool:
        """Processes detected elements and executes actions based on priority rules."""
        if not matched_elements:
            return False

        detected_element_types = list(matched_elements.keys())
        logger.debug(f"Detected elements: {[el.name for el in detected_element_types]}")

        if self._handle_priority_rules(detected_element_types, matched_elements):
            return True

        if self._handle_individual_actions(matched_elements):
            return True

        return False

    def _handle_priority_rules(self, detected_element_types: List[ElementType], matched_elements: Dict[ElementType, Tuple]) -> bool:
        """Handles priority rules."""
        for rule in self.action_config.get('priority_rules', []):
            if self._check_conditions(detected_element_types, rule.get('conditions', [])):
                logger.debug(f"Matched priority rule: {rule.get('name')}")
                action_name = rule.get('action')

                if action_name == 'wait':
                    logger.info(f"State: Wait - {rule.get('description')}")
                    return True
                
                if action_name in self.action_config.get('actions', {}):
                    target_element_type = next((et for et in matched_elements if et.name == action_name), None)
                    if target_element_type and self._check_cooldown(action_name):
                        _, position = matched_elements[target_element_type]
                        return self._execute_action(action_name, position)
        return False

    def _handle_individual_actions(self, matched_elements: Dict[ElementType, Tuple]) -> bool:
        """Handles individual element actions if no priority rule was matched."""
        for element_type, (_, position) in matched_elements.items():
            action_name = element_type.name
            if action_name.startswith('UI_') and action_name not in ['UI_WORKING', 'UI_COMPASS']:
                if self._check_cooldown(action_name):
                    return self._execute_action(action_name, position)
        return False

    def test_input_methods(self, x: int, y: int):
        """Tests all available input methods."""
        logger.info("=== Starting Input Method Test ===")
        self.input_manager.test_all_methods(x, y)

    def get_action_stats(self) -> Dict:
        """Returns statistics on executed actions."""
        return {
            'total_actions': sum(self.action_counts.values()),
            'action_counts': self.action_counts,
            'last_action_times': self.last_action_time
        }