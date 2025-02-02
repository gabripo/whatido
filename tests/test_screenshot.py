import os, unittest
import _setup_test_env
from src.screenshots_manager import Screenshot

class TestScreenshotManager(unittest.TestCase):
    def test_screenshotmanager_creation(self):
        current_folder = os.path.dirname(__file__)
        screenshot_folder = os.path.join(current_folder, "screenshots")
        screenshot_manager = Screenshot.ScreenshotManager(screenshots_dir=screenshot_folder)
        self.assertIsNot(screenshot_manager._instance, None)
        self.assertNotEqual(screenshot_manager.screenshots_dir, "")
        self.assertTrue(os.path.exists(screenshot_manager.screenshots_dir))
        self.assertEqual(screenshot_manager.scale_factor, Screenshot.DEFAULT_SCALE_FACTOR)
        Screenshot.ScreenshotManager.reset()

    def test_capture_screenshot(self):
        current_folder = os.path.dirname(__file__)
        screenshot_folder = os.path.join(current_folder, "screenshots")
        screenshot_manager = Screenshot.ScreenshotManager(screenshots_dir=screenshot_folder)
        
        screenshot_manager.capture_screenshot()
        self.assertTrue(os.path.exists(screenshot_manager.screenshots[0]))
        Screenshot.ScreenshotManager.reset()

    def test_scale_factor(self):
        current_folder = os.path.dirname(__file__)
        screenshot_folder = os.path.join(current_folder, "screenshots")
        screenshot_manager = Screenshot.ScreenshotManager(screenshots_dir=screenshot_folder)
        
        desired_scale_factor = 0.5
        screenshot_manager.set_scale_factor(desired_scale_factor)
        self.assertEqual(screenshot_manager.scale_factor, desired_scale_factor)

        screenshot_manager.capture_screenshot()
        self.assertTrue(os.path.exists(screenshot_manager.screenshots[0]))
        Screenshot.ScreenshotManager.reset()

    def test_capture_screenshot_for_duration(self):
        current_folder = os.path.dirname(__file__)
        screenshot_folder = os.path.join(current_folder, "screenshots")
        screenshot_manager = Screenshot.ScreenshotManager(screenshots_dir=screenshot_folder)

        duration_s = 2.0
        interval_s = 0.5
        num_screenshots = int(duration_s / interval_s) - 1
        screenshot_manager.capture_screenshots_for_duration_s(duration_s=duration_s, interval_s=interval_s)
        self.assertEqual(len(screenshot_manager.screenshots), num_screenshots)
        Screenshot.ScreenshotManager.reset()

if __name__ == '__main__':
    unittest.main(verbosity=2)