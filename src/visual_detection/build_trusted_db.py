"""
Trusted Screenshot Database Builder
Captures reference screenshots of legitimate websites using Selenium
and saves them as the trusted baseline for visual spoofing detection.
"""
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRUSTED_DIR = PROJECT_ROOT / "data" / "trusted_screenshots"

# Trusted site list — 50+ legitimate sites for visual cloning detection
TRUSTED_SITES = {
    # ── Indian Banks (10) ──
    "sbi":        "https://www.sbi.co.in",
    "hdfc":       "https://www.hdfcbank.com",
    "icici":      "https://www.icicibank.com",
    "axis":       "https://www.axisbank.com",
    "kotak":      "https://www.kotak.com",
    "pnb":        "https://www.pnbindia.in",
    "bob":        "https://www.bankofbaroda.in",
    "canara":     "https://www.canarabank.com",
    "unionbank":  "https://www.unionbankofindia.co.in",
    "indusind":   "https://www.indusind.com",

    # ── Global Banks / Finance (6) ──
    "paypal":     "https://www.paypal.com",
    "chase":      "https://www.chase.com",
    "wellsfargo": "https://www.wellsfargo.com",
    "bofa":       "https://www.bankofamerica.com",
    "citibank":   "https://www.citibank.com",
    "hsbc":       "https://www.hsbc.com",

    # ── Social Media (6) ──
    "google":     "https://accounts.google.com",
    "facebook":   "https://www.facebook.com",
    "instagram":  "https://www.instagram.com",
    "twitter":    "https://www.x.com",
    "linkedin":   "https://www.linkedin.com",
    "whatsapp":   "https://web.whatsapp.com",

    # ── E-Commerce (6) ──
    "amazon":     "https://www.amazon.in",
    "flipkart":   "https://www.flipkart.com",
    "myntra":     "https://www.myntra.com",
    "snapdeal":   "https://www.snapdeal.com",
    "ebay":       "https://www.ebay.com",
    "alibaba":    "https://www.alibaba.com",

    # ── Payment / UPI (5) ──
    "phonepe":    "https://www.phonepe.com",
    "paytm":      "https://www.paytm.com",
    "razorpay":   "https://www.razorpay.com",
    "bharatpe":   "https://www.bharatpe.com",
    "gpay":       "https://pay.google.com",

    # ── Email Providers (4) ──
    "gmail":      "https://mail.google.com",
    "outlook":    "https://outlook.live.com",
    "yahoo":      "https://mail.yahoo.com",
    "protonmail": "https://mail.proton.me",

    # ── Government / Services (5) ──
    "irctc":      "https://www.irctc.co.in",
    "incometax":  "https://www.incometax.gov.in",
    "digilocker": "https://www.digilocker.gov.in",
    "aadhaar":    "https://uidai.gov.in",
    "passport":   "https://www.passportindia.gov.in",

    # ── Tech / Cloud (5) ──
    "microsoft":  "https://www.microsoft.com",
    "apple":      "https://www.apple.com",
    "netflix":    "https://www.netflix.com",
    "dropbox":    "https://www.dropbox.com",
    "github":     "https://github.com",

    # ── Telecom (3) ──
    "jio":        "https://www.jio.com",
    "airtel":     "https://www.airtel.in",
    "vi":         "https://www.myvi.in",
}


def build_trusted_database():
    """
    Capture reference screenshots of all trusted sites.
    Saves full (1366x768) and thumbnail (256x256) versions.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        from PIL import Image
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("  Install: pip install selenium webdriver-manager Pillow")
        return

    os.makedirs(str(TRUSTED_DIR), exist_ok=True)

    # Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1366,768")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/120.0.0.0 Safari/537.36")

    succeeded = 0
    failed = 0
    failed_sites = []

    print("\n" + "=" * 60)
    print("BUILDING TRUSTED SCREENSHOT DATABASE")
    print("=" * 60)
    print(f"  Output directory: {TRUSTED_DIR}")
    print(f"  Sites to capture: {len(TRUSTED_SITES)}")
    print()

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(15)
    except Exception as e:
        print(f"[ERROR] Could not start Chrome: {e}")
        print("  Make sure Google Chrome is installed on your system.")
        return

    try:
        for site_key, site_url in TRUSTED_SITES.items():
            print(f"  Capturing {site_key} ({site_url})... ", end="", flush=True)
            try:
                driver.get(site_url)
                time.sleep(5)  # Wait for page load

                # Full screenshot
                full_path = TRUSTED_DIR / f"{site_key}.png"
                driver.save_screenshot(str(full_path))

                # Create 256x256 thumbnail
                thumb_path = TRUSTED_DIR / f"{site_key}_thumb.png"
                img = Image.open(full_path)
                img_thumb = img.resize((256, 256), Image.LANCZOS)
                img_thumb.save(str(thumb_path))

                succeeded += 1
                print(f"OK ({os.path.getsize(full_path) / 1024:.1f} KB)")

            except Exception as e:
                failed += 1
                failed_sites.append(site_key)
                print(f"FAILED ({str(e)[:50]})")

    finally:
        driver.quit()

    # Summary
    print("\n" + "=" * 60)
    print("SCREENSHOT CAPTURE SUMMARY")
    print("=" * 60)
    print(f"  Succeeded: {succeeded}/{len(TRUSTED_SITES)}")
    print(f"  Failed:    {failed}/{len(TRUSTED_SITES)}")
    if failed_sites:
        print(f"  Failed sites: {', '.join(failed_sites)}")
    print(f"  Screenshots saved to: {TRUSTED_DIR}")
    print("=" * 60)


def get_trusted_sites():
    """Return the trusted sites dictionary."""
    return TRUSTED_SITES.copy()


def get_trusted_dir():
    """Return the trusted screenshots directory path."""
    return TRUSTED_DIR


if __name__ == "__main__":
    build_trusted_database()
