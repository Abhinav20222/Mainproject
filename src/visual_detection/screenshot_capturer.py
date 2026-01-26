"""
Screenshot Capturer Module
Captures screenshots of suspect websites using headless Chrome
for visual spoofing analysis.
"""
import os
import sys
import time
import glob
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent
TEMP_DIR = PROJECT_ROOT / "data" / "temp"


class ScreenshotError(Exception):
    """Raised when screenshot capture fails."""
    pass


class ScreenshotCapturer:
    """
    Captures screenshots of suspect websites using headless Chrome.
    
    Usage:
        capturer = ScreenshotCapturer()
        filepath = capturer.capture("http://suspicious-site.com")
    """

    def __init__(self, timeout=10, window_size=(1366, 768)):
        """
        Initialize the screenshot capturer.
        
        Args:
            timeout (int): Page load timeout in seconds
            window_size (tuple): Browser window dimensions (width, height)
        """
        self.timeout = timeout
        self.window_size = window_size
        os.makedirs(str(TEMP_DIR), exist_ok=True)

    def capture(self, url):
        """
        Capture a screenshot of the given URL.
        
        Args:
            url (str): URL to screenshot
            
        Returns:
            str: Absolute file path to the saved screenshot
            
        Raises:
            ScreenshotError: If capture fails
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
        except ImportError as e:
            raise ScreenshotError(f"Missing dependency: {e}. Install: pip install selenium webdriver-manager")

        # Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f"--window-size={self.window_size[0]},{self.window_size[1]}")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/120.0.0.0 Safari/537.36")

        driver = None
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.set_page_load_timeout(self.timeout)

            # Navigate to URL
            driver.get(url)
            time.sleep(3)  # Wait for rendering

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"suspect_{timestamp}.png"
            filepath = str(TEMP_DIR / filename)

            # Capture
            driver.save_screenshot(filepath)

            if not os.path.exists(filepath):
                raise ScreenshotError("Screenshot file was not created")

            return filepath

        except ScreenshotError:
            raise
        except Exception as e:
            raise ScreenshotError(f"Failed to capture screenshot of {url}: {str(e)}")
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception:
                    pass

    def cleanup_old_screenshots(self, max_age_minutes=30):
        """
        Delete old temporary screenshots.
        
        Args:
            max_age_minutes (int): Maximum age in minutes before deletion
        """
        cutoff = datetime.now() - timedelta(minutes=max_age_minutes)
        deleted = 0

        for filepath in glob.glob(str(TEMP_DIR / "suspect_*.png")):
            try:
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                if file_mtime < cutoff:
                    os.remove(filepath)
                    deleted += 1
            except Exception:
                pass

        if deleted > 0:
            print(f"[Cleanup] Deleted {deleted} old screenshot(s)")

        return deleted


# Quick test
if __name__ == "__main__":
    capturer = ScreenshotCapturer()

    test_url = "https://www.google.com"
    print(f"Capturing screenshot of {test_url}...")

    try:
        path = capturer.capture(test_url)
        print(f"[OK] Screenshot saved: {path}")
        print(f"     Size: {os.path.getsize(path) / 1024:.1f} KB")
    except ScreenshotError as e:
        print(f"[ERROR] {e}")

    # Test cleanup
    capturer.cleanup_old_screenshots(max_age_minutes=0)
