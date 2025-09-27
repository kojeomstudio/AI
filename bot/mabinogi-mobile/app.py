
import time
import sys
import os
import signal
from typing import Dict, List

import typer
from ultralytics import YOLO

from ui.base.element import ElementType
from ui.yolo_element import YoloElement
from logger_helper import get_logger
from utils.capture import get_game_window_image
from input_manager import InputManager
from action_processor import ActionProcessor
from config_manager import ConfigManager

class MacroApp:
    def __init__(self, config_manager: ConfigManager):
        self.logger = get_logger()
        self.running = True
        self.config_manager = config_manager

        self.model = self._load_model()
        self.elements = self._load_elements()
        self.input_manager = InputManager(self.config_manager.get("config", "window_title"))
        self.action_processor = ActionProcessor("config/action_config.json", self.input_manager)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_model(self) -> YOLO:
        """Loads the YOLO model."""
        model_path = self.config_manager.get("config", "model_path")
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
        for element_data in self.config_manager.get_config("elements").get("elements", []):
            element_name = element_data.get("name")
            class_id = element_data.get("class_id")
            try:
                element_type = ElementType[element_name]
                elements.append(YoloElement(element_type, class_id=class_id))
            except KeyError:
                self.logger.error(f"ElementType '{element_name}' not found.")
            except Exception as e:
                self.logger.error(f"Failed to create element '{element_name}': {e}")

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
        self.logger.info(f"Target window: {self.config_manager.get('config', 'window_title')}")
        self.logger.info(f"Tick interval: {self.config_manager.get('config', 'tick_interval', 0.5)}s")

        try:
            while self.running:
                try:
                    if not self.input_manager.monitor_process():
                        self.logger.warning("Target process not found. Retrying in 5 seconds...")
                        time.sleep(5)
                        continue

                    screen_np = get_game_window_image(self.config_manager.get("config", "window_title"))
                    if screen_np is None:
                        self.logger.warning("Game window not found. Retrying in 5 seconds...")
                        time.sleep(5)
                        continue

                    self.logger.debug("Performing YOLO prediction...")
                    results = self.model.predict(screen_np, conf=self.config_manager.get("config", "confidence_threshold", 0.5), verbose=False)
                    
                    matched = self._match_elements(results)

                    if matched:
                        self.action_processor.process_detected_elements(matched)
                    else:
                        self.logger.debug("No elements detected.")

                except Exception as e:
                    self.logger.error(f"An error occurred in the main loop: {e}", exc_info=True)

                time.sleep(self.config_manager.get("config", "tick_interval", 0.5))

        except KeyboardInterrupt:
            self.logger.info("Macro terminated by user.")
        finally:
            self._print_stats()
            self.logger.info("Macro has been shut down.")

    def test_mode(self):
        """Runs the macro in test mode to test input methods."""
        self.logger.info("=== Starting Test Mode ===")
        screen_np = get_game_window_image(self.config_manager.get("config", "window_title"))
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

app = typer.Typer()

@app.command()
def run(test: bool = typer.Option(False, "--test", help="Run in test mode.")):
    """Runs the Mabinogi Mobile macro."""
    config_manager = ConfigManager()
    macro_app = MacroApp(config_manager)
    if test:
        macro_app.test_mode()
    else:
        macro_app.run()

if __name__ == "__main__":
    app()
