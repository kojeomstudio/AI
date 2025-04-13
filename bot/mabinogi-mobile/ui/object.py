import pyautogui
from ui.base.element import UIElement

import win32gui
import win32con

class CoalVeinNode(UIElement):
    def action(self, position):
        #x1, y1, x2, y2 = position
        #center_x = int((x1 + x2) / 2)
        #center_y = int((y1 + y2) / 2)
        #pyautogui.click(center_x, center_y)
        pyautogui.press('space')
        print(f"[ACTION] 채광 수행: {self.type.name} by space key")


class IronVeinNode(UIElement):
    def action(self, position):
        #x1, y1, x2, y2 = position
        #center_x = int((x1 + x2) / 2)
        #center_y = int((y1 + y2) / 2)
        #pyautogui.click(center_x, center_y)
        pyautogui.press('space')
        print(f"[ACTION] 채광 수행: {self.type.name} by space key")

class NormalVeinNode(UIElement):
    def action(self, position):
        #x1, y1, x2, y2 = position
        #center_x = int((x1 + x2) / 2)
        #center_y = int((y1 + y2) / 2)
        #pyautogui.click(center_x, center_y)
        pyautogui.press('space')
        print(f"[ACTION] 채광 수행: {self.type.name} by space key")

class TreeNode(UIElement):
    def action(self, position):
        #x1, y1, x2, y2 = position
        #center_x = int((x1 + x2) / 2)
        #center_y = int((y1 + y2) / 2)
        #pyautogui.click(center_x, center_y)
        pyautogui.press('space')
        print(f"[ACTION] 채광 수행: {self.type.name} by space key")