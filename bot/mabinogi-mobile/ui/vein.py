import pyautogui
from ui.element import UIElement

class CoalNode(UIElement):
    def action(self, position):
        x, y = position
        ox, oy = self.offset
        pyautogui.moveTo(x + ox, y + oy)
        pyautogui.click()
        pyautogui.press('space')
        print(f"[ACTION] 채광 수행: {self.name} at ({x}, {y})")


class IronNode(UIElement):
    def action(self, position):
        x, y = position
        ox, oy = self.offset
        pyautogui.moveTo(x + ox, y + oy)
        pyautogui.click()
        pyautogui.press('space')
        print(f"[ACTION] 채광 수행: {self.name} at ({x}, {y})")
