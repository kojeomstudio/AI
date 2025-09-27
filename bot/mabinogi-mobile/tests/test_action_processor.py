
import unittest
from unittest.mock import MagicMock, patch, mock_open
import os
import json
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock modules that are not available in the test environment
sys.modules['pyautogui'] = MagicMock()
sys.modules['win32gui'] = MagicMock()
sys.modules['win32con'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['pywin32'] = MagicMock()
sys.modules['win32api'] = MagicMock()
sys.modules['win32process'] = MagicMock()

from action_processor import ActionProcessor
from ui.base.element import ElementType

class TestActionProcessor(unittest.TestCase):

    def setUp(self):
        # Create a mock InputManager
        self.mock_input_manager = MagicMock()

        # Create a dummy action_config.json for testing
        self.action_config = {
            "input_method": "test",
            "default_delay": 0.1,
            "actions": {
                "UI_ATTACK": {"type": "click"},
                "COAL_VEIN": {"type": "key", "key": "space"}
            },
            "priority_rules": [
                {
                    "name": "Attack if enemy detected",
                    "conditions": ["UI_ATTACK"],
                    "action": "UI_ATTACK"
                }
            ]
        }
        self.action_config_path = "test_action_config.json"
        with open(self.action_config_path, "w") as f:
            json.dump(self.action_config, f)

        self.action_processor = ActionProcessor(self.action_config_path, self.mock_input_manager)

    def tearDown(self):
        os.remove(self.action_config_path)

    def test_load_action_config(self):
        self.assertIsNotNone(self.action_processor.action_config)
        self.assertEqual(self.action_processor.action_config["input_method"], "test")

    def test_execute_click_action(self):
        self.mock_input_manager.click.return_value = True
        result = self.action_processor._execute_action("UI_ATTACK", (10, 20, 30, 40))
        self.assertTrue(result)
        self.mock_input_manager.click.assert_called_once_with(20, 30, method="test")

    def test_execute_key_action(self):
        self.mock_input_manager.send_key.return_value = True
        result = self.action_processor._execute_action("COAL_VEIN")
        self.assertTrue(result)
        self.mock_input_manager.send_key.assert_called_once_with("space", method="test")

    def test_priority_rule_match(self):
        matched_elements = {ElementType.UI_ATTACK: (MagicMock(), (10, 20, 30, 40))}
        with patch.object(self.action_processor, '_execute_action', return_value=True) as mock_execute:
            result = self.action_processor.process_detected_elements(matched_elements)
            self.assertTrue(result)
            mock_execute.assert_called_once_with("UI_ATTACK", (10, 20, 30, 40))

if __name__ == '__main__':
    unittest.main()
