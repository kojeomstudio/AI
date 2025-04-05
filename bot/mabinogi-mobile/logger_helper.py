import logging
from pathlib import Path
from utils.macro_uitls import get_path

_logger_instance = None

def get_logger(name: str = "macro_logger", log_file: str = "./logs/macro.log") -> logging.Logger:
    global _logger_instance
    if _logger_instance:
        return _logger_instance

    Path(get_path("./logs")).mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

    if not logger.handlers:
        file_handler = logging.FileHandler(get_path(log_file), encoding='utf-8')
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    _logger_instance = logger
    return logger
