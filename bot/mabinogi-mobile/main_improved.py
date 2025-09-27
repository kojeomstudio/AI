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
from utils.capture import get_game_window_image
from input_manager import InputManager
from action_processor import ActionProcessor

class MacroApp:
    def __init__(self, config_path="config/config.json", elements_path="config/elements.json"):
        self.logger = get_logger()
        self.running = True
        self.config = self._load_json(config_path)
        self.elements_config = self._load_json(elements_path)
        
        if not self.config or not self.elements_config:
            self.logger.error("Failed to load configuration files. Exiting.")
            sys.exit(1)

        self.model = self._load_model()
        self.elements = self._load_elements()
        self.input_manager = InputManager(self.config["window_title"])
        self.action_processor = ActionProcessor("config/action_config.json", self.input_manager)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_json(self, path: str) -> dict:
        """Loads a JSON file."""
        try:
            with open(self._get_file_path(path), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file at {path}: {e}")
            return {}

    def _get_file_path(self, in_origin: str) -> str:
        """Gets the absolute path to a file."""
        base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, str(in_origin))

    def _load_model(self) -> YOLO:
        """Loads the YOLO model."""
        model_path = self._get_file_path(self.config["model_path"])
        if not os.path.exists(model_path):
            self.logger.error(f"YOLO model file not found: {model_path}")
            sys.exit(1)
        
        try:
            model = YOLO(model_path)
            self.logger.info(f"YOLO model loaded successfully: {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            sys.exit(1)

    def _load_elements(self) -> List:
        """Loads YOLO elements from the configuration."""
        elements = []
        element_classes = {
            "CoalVeinNode": CoalVeinNode,
            "IronVeinNode": IronVeinNode,
            "NormalVeinNode": NormalVeinNode,
            "TreeNode": TreeNode,
            "UI_Attack": UI_Attack,
            "UI_Inventory": UI_Inventory,
            "UI_Riding": UI_Riding,
            "UI_Riding_Out": UI_Riding_Out,
            "UI_Mining": UI_Mining,
            "UI_Craft": UI_Craft,
            "UI_Compass": UI_Compass,
            "UI_Felling": UI_Felling,
            "UI_Working": UI_Working,
            "UI_Wing": UI_Wing,
        }

        for element_data in self.elements_config.get("elements", []):
            element_name = element_data.get("name")
            class_id = element_data.get("class_id")
            
            # Find the corresponding class in element_classes based on the name
            element_class = next((cls for name, cls in element_classes.items() if name == element_name), None)

            if element_class:
                try:
                    element_type = ElementType[element_name]
                    elements.append(element_class(element_type, class_id=class_id))
                except KeyError:
                    self.logger.error(f"ElementType '{element_name}' not found.")
                except Exception as e:
                    self.logger.error(f"Failed to create element '{element_name}': {e}")
            else:
                self.logger.warning(f"Element class for '{element_name}' not found.")

        self.logger.info(f"Loaded {len(elements)} YOLO elements.")
        return elements

    def _signal_handler(self, signum, frame):
        """Handles signals to gracefully shut down the application."""
        self.logger.info("Termination signal received. Shutting down...")
        self.running = False

    def _match_elements(self, results) -> Dict:
        """Matches detected elements with YOLO results."""
        matched = {}
        for element in self.elements:
            is_match, pos = element.match(results)
            if is_match:
                self.logger.debug(f"Detected: {element.get_type().name} at {pos}")
                matched[element.get_type()] = (element, pos)
        return matched

    def run(self):
        """The main loop of the macro."""
        self.logger.info("Starting improved YOLO macro...")
        self.logger.info(f"Target window: {self.config['window_title']}")
        self.logger.info(f"Tick interval: {self.config.get('tick_interval', 0.5)}s")

        try:
            while self.running:
                try:
                    if not self.input_manager.monitor_process():
                        self.logger.warning("Target process not found. Retrying in 5 seconds...")
                        time.sleep(5)
                        continue

                    screen_np = get_game_window_image(self.config["window_title"])
                    if screen_np is None:
                        self.logger.warning("Game window not found. Retrying in 5 seconds...")
                        time.sleep(5)
                        continue

                    self.logger.debug("Performing YOLO prediction...")
                    results = self.model.predict(screen_np, conf=self.config.get("confidence_threshold", 0.5), verbose=False)
                    
                    matched = self._match_elements(results)

                    if matched:
                        self.action_processor.process_detected_elements(matched)
                    else:
                        self.logger.debug("No elements detected.")

                except Exception as e:
                    self.logger.error(f"An error occurred in the main loop: {e}", exc_info=True)

                time.sleep(self.config.get("tick_interval", 0.5))

        except KeyboardInterrupt:
            self.logger.info("Macro terminated by user.")
        finally:
            self._print_stats()
            self.logger.info("Macro has been shut down.")

    def test_mode(self):
        """Runs the macro in test mode to test input methods."""
        self.logger.info("=== Starting Test Mode ===")
        screen_np = get_game_window_image(self.config["window_title"])
        if screen_np is not None:
            height, width, _ = screen_np.shape
            center_x, center_y = width // 2, height // 2
            self.logger.info(f"Screen dimensions: {width}x{height}, center: ({center_x}, {center_y})")
            self.action_processor.test_input_methods(center_x, center_y)
        else:
            self.logger.error("Game window not found. Cannot proceed with the test.")

    def _print_stats(self):
        """Prints the action statistics."""
        stats = self.action_processor.get_action_stats()
        self.logger.info("=== Macro Execution Stats ===")
        self.logger.info(f"Total actions executed: {stats['total_actions']}")
        if stats['action_counts']:
            self.logger.info("Actions executed:")
            for action, count in stats['action_counts'].items():
                self.logger.info(f"  - {action}: {count} times")

if __name__ == "__main__":
    app = MacroApp()
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        app.test_mode()
    else:
        app.run()