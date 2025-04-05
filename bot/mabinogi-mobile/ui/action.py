import pyautogui
from ui.base.element import UIElement

from logger_helper import *

import win32gui
import win32con

logger = get_logger()

class UI_Felling(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")


class UI_Mining(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")

class UI_Wait(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")

class UI_Working(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")
