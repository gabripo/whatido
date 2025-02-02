import os
import pyautogui
from datetime import datetime
import numpy as np
import cv2
import time

DEFAULT_SCALE_FACTOR = 1.0

class ScreenshotManager:
    _instance = None

    def __new__(cls, screenshots_dir: str = ""):
        if cls._instance is None:
            cls._instance = super(ScreenshotManager, cls).__new__(cls)
            cls._instance.screenshots = []
            cls._instance.screenshots_dir = screenshots_dir
            cls._instance.set_screenshots_dir()
            cls._instance.scale_factor = DEFAULT_SCALE_FACTOR
        return cls._instance
    
    def set_screenshots_dir(self):
        if not self.screenshots_dir:
            self.screenshots_dir = os.path.join(os.getcwd(), "screenshots")
        self.screenshots_dir = os.path.abspath(self.screenshots_dir)

        if not os.path.exists(self.screenshots_dir):
            print(f"Creating a folder for screenshots at path {self.screenshots_dir}")
            os.makedirs(self.screenshots_dir)

    def set_scale_factor(self, scale_factor: float):
        self.scale_factor = scale_factor

    def capture_screenshot(self):
        screenshot = np.array(pyautogui.screenshot())
        screenshot = self._scale_screenshot(screenshot)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(self.screenshots_dir, f"shot_{timestamp}.png")
        cv2.imwrite(filepath, screenshot)
        self.screenshots.append(filepath)
    
    def _scale_screenshot(self, screenshot: np.array):
        if self.scale_factor != DEFAULT_SCALE_FACTOR:
            width = int(screenshot.shape[1] * self.scale_factor)
            height = int(screenshot.shape[0] * self.scale_factor)
            screenshot = cv2.resize(screenshot, (width, height), interpolation=cv2.INTER_AREA)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        return screenshot
    
    def capture_screenshots_for_duration_s(self, duration_s: float = 1.0, interval_s: float = 0.5):
        # TODO find a more reliable way to perform screenshots given the time
        start_time = time.time()
        while (time.time() - start_time) < duration_s:
            self.capture_screenshot()
            time.sleep(interval_s)

    @classmethod
    def reset(cls):
        cls._instance = None
    
if __name__ == "__main__":
    manager = ScreenshotManager()

    manager.capture_screenshot()

    manager.set_scale_factor(0.5)
    manager.capture_screenshot()

    manager.set_scale_factor(DEFAULT_SCALE_FACTOR)
    manager.capture_screenshots_for_duration_s(0.5, 0.1)

    manager.set_scale_factor(0.5)
    manager.capture_screenshots_for_duration_s(1.5, 0.5)