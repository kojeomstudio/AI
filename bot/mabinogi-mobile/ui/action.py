import pyautogui
from ui.base.element import UIElement

from logger_helper import *

logger = get_logger()

class UI_Felling(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")

class UI_Attack(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")

class UI_Inventory(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")

class UI_Riding(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")

class UI_Riding_Out(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")

class UI_Mining(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")

class UI_Compass(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")

class UI_Working(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")

class UI_Craft(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")

class UI_Wing(UIElement):
    def action(self, position):
        logging.debug(f"[ACTION] {self.type.name}")