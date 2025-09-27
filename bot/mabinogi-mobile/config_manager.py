
import json
import os
import sys
from typing import Dict

from logger_helper import get_logger

class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.logger = get_logger()
        self.config_dir = self._get_file_path(config_dir)
        self.config = self._load_config()

    def _get_file_path(self, in_origin: str) -> str:
        """Gets the absolute path to a file."""
        base_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, str(in_origin))

    def _load_config(self) -> Dict:
        """Loads all JSON configuration files from the config directory."""
        config = {}
        for filename in os.listdir(self.config_dir):
            if filename.endswith(".json"):
                config_name = os.path.splitext(filename)[0]
                try:
                    with open(os.path.join(self.config_dir, filename), "r", encoding="utf-8") as f:
                        config[config_name] = json.load(f)
                        self.logger.info(f"Loaded configuration file: {filename}")
                except Exception as e:
                    self.logger.error(f"Failed to load configuration file {filename}: {e}")
        return config

    def get(self, config_name: str, key: str, default=None):
        """Gets a configuration value."""
        return self.config.get(config_name, {}).get(key, default)

    def get_config(self, config_name: str) -> Dict:
        """Gets a whole configuration dictionary."""
        return self.config.get(config_name, {})
