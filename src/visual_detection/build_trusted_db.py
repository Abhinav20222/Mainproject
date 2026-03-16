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

# Trusted site list — 50 legitimate sites for visual cloning detection
TRUSTED_SITES = {
    # ── Indian Banks (10) ──
    "sbi":        "https://www.onlinesbi.sbi",
    "hdfc":       "https://www.hdfcbank.com",
    "icici":      "https://www.icicibank.com",
    "axis":       "https://www.axisbank.com",
    "kotak":      "https://www.kotak.com",
    "pnb":        "https://www.pnbindia.in",
    "bob":        "https://www.bankofbaroda.in",
    "canara":     "https://www.canarabank.com",
    "unionbank":  "https://unionbankofindia.co.in/english/home.aspx",
    "indusind":   "https://www.indusind.com",

    # ── Global Banks / Finance (6) ──
    "paypal":     "https://www.paypal.com/in/home",
    "chase":      "https://www.chase.com",
    "wellsfargo": "https://www.wellsfargo.com",
    "bofa":       "https://www.bankofamerica.com",
    "citibank":   "https://www.citibank.com",
    "hsbc":       "https://www.hsbc.co.in",

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
    "gpay":       "https://pay.google.com/about",

    # ── Email Providers (4) ──
    "gmail":      "https://mail.google.com",
    "outlook":    "https://outlook.live.com",
    "yahoo":      "https://mail.yahoo.com",
    "protonmail": "https://mail.proton.me",

    # ── Government / Services (5) ──
    "irctc":      "https://www.irctc.co.in",
    "incometax":  "https://www.incometax.gov.in",
    "digilocker": "https://www.digilocker.gov.in",
    "aadhaar":    "https://myaadhaar.uidai.gov.in",
    "passport":   "https://www.passportindia.gov.in",

    # ── Tech / Cloud (5) ──
    "microsoft":  "https://www.microsoft.com/en-in",
    "apple":      "https://www.apple.com/in",
    "netflix":    "https://www.netflix.com",
    "dropbox":    "https://www.dropbox.com",
    "github":     "https://github.com",

    # ── Telecom (3) ──
    "jio":        "https://www.jio.com",
    "airtel":     "https://www.airtel.in",
    "vi":         "https://www.myvi.in",

    # ── Food Delivery / Fintech (3) ── NEW to reach 50
    "swiggy":     "https://www.swiggy.com",
    "zomato":     "https://www.zomato.com",
    "cred":       "https://www.cred.club",

    # ── Government Services (additional) ──
    "postoffice": "https://www.indiapost.gov.in",
}

# JavaScript to dismiss common popups, cookie banners, and overlays
DISMISS_POPUPS_JS = """
(function() {
    // 1. Remove common cookie/consent banners by class/id keywords
    var keywords = ['cookie', 'consent', 'gdpr', 'privacy', 'popup', 'modal',
                    'overlay', 'banner', 'notice', 'alert', 'notification',
                    'dialog', 'snackbar', 'bottom-bar', 'onetrust', 'cc-banner',
                    'accessibility', 'ada-embed', 'chatbot', 'chat-widget',
                    'intercom', 'drift', 'crisp', 'tawk', 'livechat',
                    'language-selector', 'lang-select', 'locale-popup'];
    
    var allElements = document.querySelectorAll('*');
    for (var i = 0; i < allElements.length; i++) {
        var el = allElements[i];
        var cls = (el.className || '').toString().toLowerCase();
        var id = (el.id || '').toLowerCase();
        var role = (el.getAttribute('role') || '').toLowerCase();
        
        for (var j = 0; j < keywords.length; j++) {
            if (cls.indexOf(keywords[j]) !== -1 || id.indexOf(keywords[j]) !== -1) {
                // Check if it's a floating/fixed/absolute element (popup-like)
                var style = window.getComputedStyle(el);
                if (style.position === 'fixed' || style.position === 'absolute' || 
                    style.position === 'sticky' || style.zIndex > 100 ||
                    el.tagName === 'DIALOG') {
                    el.remove();
                    break;
                }
            }
        }
        // Also remove elements with role=dialog
        if (role === 'dialog' || role === 'alertdialog') {
            el.remove();
        }
    }
    
    // 2. Click common "Accept" / "OK" / "Close" / "Got it" buttons
    var buttons = document.querySelectorAll('button, a, [role="button"]');
    var acceptWords = ['accept', 'agree', 'ok', 'got it', 'continue', 'close',
                       'dismiss', 'decline', 'reject', 'no thanks', 'i understand',
                       'allow all', 'accept all', 'allow cookies'];
    for (var k = 0; k < buttons.length; k++) {
        var btn = buttons[k];
        var txt = (btn.textContent || '').trim().toLowerCase();
        for (var m = 0; m < acceptWords.length; m++) {
            if (txt === acceptWords[m] || txt === acceptWords[m] + ' all' ||
                txt === acceptWords[m] + ' cookies') {
                try { btn.click(); } catch(e) {}
                break;
            }
        }
    }
    
    // 3. Remove any fixed/sticky overlays that cover the page
    var fixedEls = document.querySelectorAll('*');
    for (var n = 0; n < fixedEls.length; n++) {
        var fel = fixedEls[n];
        var fstyle = window.getComputedStyle(fel);
        if ((fstyle.position === 'fixed' || fstyle.position === 'sticky') &&
            parseInt(fstyle.zIndex) > 999) {
            fel.remove();
        }
    }
    
    // 4. Remove backdrop/overlay divs
    var backdrops = document.querySelectorAll('.modal-backdrop, .overlay, [class*="backdrop"]');
    backdrops.forEach(function(b) { b.remove(); });
    
    // 5. Re-enable scrolling on body
    document.body.style.overflow = 'auto';
    document.documentElement.style.overflow = 'auto';
})();
"""

# Site-specific JavaScript to handle unique overlays/issues per site
SITE_SPECIFIC_JS = {
    "icici": """
    (function() {
        // Remove ICICI's accessibility panel (ada-embed widget)
        var adaEls = document.querySelectorAll('#ada-entry, .ada-embed, [id*="ada"], [class*="ada-"], [id*="accessibility"], .accessibe, #acsb-trigger');
        adaEls.forEach(function(el) { el.remove(); });
        // Remove any accessibility floating button
        var allBtns = document.querySelectorAll('button, div');
        allBtns.forEach(function(el) {
            var ariaLabel = (el.getAttribute('aria-label') || '').toLowerCase();
            if (ariaLabel.indexOf('accessibility') !== -1 || ariaLabel.indexOf('ada') !== -1) {
                el.remove();
            }
        });
        // Remove FD calculator popup/overlay and any modal dialogs
        var modals = document.querySelectorAll('[class*="modal"], [class*="Modal"], [class*="popup"], [class*="Popup"], [class*="dialog"], [class*="Dialog"], [class*="overlay"], [class*="Overlay"], [class*="calculator"], [class*="Calculator"]');
        modals.forEach(function(el) {
            var s = window.getComputedStyle(el);
            if (s.position === 'fixed' || s.position === 'absolute' || parseInt(s.zIndex) > 10) {
                el.remove();
            }
        });
        // Remove any backdrop overlays
        var backdrops = document.querySelectorAll('[class*="backdrop"], [class*="Backdrop"]');
        backdrops.forEach(function(b) { b.remove(); });
        // Close any open popups by clicking close/X buttons within modals
        var closeBtns = document.querySelectorAll('[class*="close"], [class*="Close"], [aria-label="Close"], [aria-label="close"]');
        closeBtns.forEach(function(btn) {
            try { btn.click(); } catch(e) {}
        });
        document.body.style.overflow = 'auto';
        document.documentElement.style.overflow = 'auto';
    })();
    """,
    "pnb": """
    (function() {
        // Remove PNB's cookie banner and any floating notifications
        var cookieEls = document.querySelectorAll('[class*="cookie"], [id*="cookie"], [class*="consent"], [id*="consent"], .cc-window, .cc-banner, #cookieNotice, .cookie-bar, .cookie-notice');
        cookieEls.forEach(function(el) { el.remove(); });
        // Click accept/close on any remaining cookie buttons
        var btns = document.querySelectorAll('button, a');
        btns.forEach(function(b) {
            var txt = (b.textContent || '').trim().toLowerCase();
            if (txt === 'accept' || txt === 'ok' || txt === 'i agree' || txt === 'close' || txt === 'accept all cookies' || txt === 'accept cookies') {
                try { b.click(); } catch(e) {}
            }
        });
        // Remove overlay backdrop
        var overlays = document.querySelectorAll('[class*="overlay"], [class*="backdrop"]');
        overlays.forEach(function(o) {
            var s = window.getComputedStyle(o);
            if (s.position === 'fixed' || s.position === 'absolute') o.remove();
        });
        document.body.style.overflow = 'auto';
    })();
    """,
    "aadhaar": """
    (function() {
        // Remove language selection modal if present
        var modals = document.querySelectorAll('.modal, [class*="language"], [class*="lang-"], [id*="language"], [id*="lang"], .popup, [class*="popup"]');
        modals.forEach(function(el) {
            var s = window.getComputedStyle(el);
            if (s.position === 'fixed' || s.position === 'absolute' || s.display === 'block') {
                el.remove();
            }
        });
        // Click English button if visible
        var links = document.querySelectorAll('a, button');
        links.forEach(function(l) {
            var txt = (l.textContent || '').trim().toLowerCase();
            if (txt === 'english' || txt === 'en') {
                try { l.click(); } catch(e) {}
            }
        });
        // Remove backdrop
        var backdrops = document.querySelectorAll('.modal-backdrop, [class*="backdrop"]');
        backdrops.forEach(function(b) { b.remove(); });
        document.body.style.overflow = 'auto';
        document.documentElement.style.overflow = 'auto';
    })();
    """,
    "unionbank": """
    (function() {
        // Remove any Cloudflare challenge elements
        var cfEls = document.querySelectorAll('#challenge-running, #challenge-form, .cf-browser-verification, #cf-wrapper, [class*="cloudflare"], [id*="challenge"]');
        cfEls.forEach(function(el) { el.remove(); });
    })();
    """,
    "flipkart": """
    (function() {
        // Remove login popup/tooltip that appears on hover/load
        var loginPopups = document.querySelectorAll('[class*="login"], [class*="Login"], [class*="signup"], [class*="SignUp"]');
        loginPopups.forEach(function(el) {
            var s = window.getComputedStyle(el);
            if (s.position === 'fixed' || s.position === 'absolute' || parseInt(s.zIndex) > 10) {
                el.remove();
            }
        });
        // Remove floating promo/notification bars
        var promos = document.querySelectorAll('[class*="promo"], [class*="notification"], [class*="toast"], [class*="_3Njd"]');
        promos.forEach(function(el) {
            var s = window.getComputedStyle(el);
            if (s.position === 'fixed' || s.position === 'absolute') el.remove();
        });
        // Remove any modal overlays
        var modals = document.querySelectorAll('[class*="modal"], [class*="Modal"], [class*="overlay"], [class*="Overlay"]');
        modals.forEach(function(el) {
            var s = window.getComputedStyle(el);
            if (s.position === 'fixed') el.remove();
        });
        document.body.style.overflow = 'auto';
    })();
    """,
    "hdfc": """
    (function() {
        // Remove HDFC's EVA chatbot bar
        var evaEls = document.querySelectorAll('[class*="eva"], [id*="eva"], [class*="Eva"], [id*="Eva"], [class*="chatbot"], [id*="chatbot"], [class*="chat-bot"], [class*="chat_bot"]');
        evaEls.forEach(function(el) { el.remove(); });
        // Remove accessibility button
        var accessEls = document.querySelectorAll('[class*="accessibility"], [id*="accessibility"], [aria-label*="accessibility"]');
        accessEls.forEach(function(el) { el.remove(); });
        // Remove any fixed bottom bars
        var allEls = document.querySelectorAll('*');
        for (var i = 0; i < allEls.length; i++) {
            var el = allEls[i];
            var s = window.getComputedStyle(el);
            if (s.position === 'fixed' && el.getBoundingClientRect().bottom > window.innerHeight - 100) {
                el.remove();
            }
        }
    })();
    """,
    "digilocker": """
    (function() {
        // Remove chatbot widget (bottom-right floating)
        var chatEls = document.querySelectorAll('[class*="chat"], [id*="chat"], [class*="bot"], [id*="bot"], [class*="support"], [class*="widget"]');
        chatEls.forEach(function(el) {
            var s = window.getComputedStyle(el);
            if (s.position === 'fixed' || s.position === 'absolute') el.remove();
        });
        // Remove accessibility icon
        var accessEls = document.querySelectorAll('[class*="accessibility"], [id*="accessibility"], [aria-label*="accessibility"]');
        accessEls.forEach(function(el) { el.remove(); });
    })();
    """,
    "myntra": """
    (function() {
        // Remove side coupon/discount popup (₹500 OFF strip)
        var sidePopups = document.querySelectorAll('[class*="coupon"], [class*="discount"], [class*="offer-strip"], [class*="side-bar"], [class*="sidebar"]');
        sidePopups.forEach(function(el) {
            var s = window.getComputedStyle(el);
            if (s.position === 'fixed' || s.position === 'absolute' || s.position === 'sticky') {
                el.remove();
            }
        });
        // Remove any right-edge floating elements
        var allEls = document.querySelectorAll('*');
        for (var i = 0; i < allEls.length; i++) {
            var el = allEls[i];
            var s = window.getComputedStyle(el);
            var rect = el.getBoundingClientRect();
            if ((s.position === 'fixed' || s.position === 'sticky') && rect.right >= window.innerWidth - 50) {
                el.remove();
            }
        }
        // Remove notification/promotion banners
        var promos = document.querySelectorAll('[class*="promo"], [class*="toast"], [class*="snackbar"]');
        promos.forEach(function(el) { el.remove(); });
    })();
    """,
    "outlook": """
    (function() {
        // Remove promotional banner bar (e.g., "Discover Microsoft 365 Copilot")
        var promoEls = document.querySelectorAll('[class*="promo"], [class*="banner"], [class*="announcement"], [id*="promo"], [id*="banner"]');
        promoEls.forEach(function(el) {
            var s = window.getComputedStyle(el);
            if (s.position === 'fixed' || s.position === 'sticky' || s.position === 'relative') {
                // Only remove relatively small banner-like elements at the top
                var rect = el.getBoundingClientRect();
                if (rect.height < 100) el.remove();
            }
        });
        // Remove feedback button
        var feedbackEls = document.querySelectorAll('[class*="feedback"], [id*="feedback"], [aria-label*="feedback"], [aria-label*="Feedback"]');
        feedbackEls.forEach(function(el) { el.remove(); });
    })();
    """,
}


def build_trusted_database(only_sites=None):
    """
    Capture reference screenshots of trusted sites.
    Saves full (1366x768) and thumbnail (256x256) versions.
    
    Args:
        only_sites: Optional list of site keys to re-capture. 
                    If None, captures all sites.
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

    # Chrome options — improved for cleaner screenshots
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1366,768")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-translate")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--lang=en-US")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disable-features=IsolateOrigins,site-per-process")
    # Block cookie consent banners at browser level
    chrome_options.add_experimental_option("prefs", {
        "profile.default_content_setting_values.notifications": 2,
        "profile.default_content_setting_values.geolocation": 2,
        "profile.cookie_controls_mode": 0,
    })
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )

    # Determine which sites to capture
    if only_sites:
        sites_to_capture = {k: v for k, v in TRUSTED_SITES.items() if k in only_sites}
        if not sites_to_capture:
            print("[ERROR] No matching sites found for the given keys.")
            return
    else:
        sites_to_capture = TRUSTED_SITES

    succeeded = 0
    failed = 0
    failed_sites = []

    print("\n" + "=" * 60)
    print("BUILDING TRUSTED SCREENSHOT DATABASE")
    print("=" * 60)
    print(f"  Output directory: {TRUSTED_DIR}")
    print(f"  Sites to capture: {len(sites_to_capture)}")
    if only_sites:
        print(f"  Mode: RE-CAPTURE selected sites")
    print()

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.set_page_load_timeout(20)
    except Exception as e:
        print(f"[ERROR] Could not start Chrome: {e}")
        print("  Make sure Google Chrome is installed on your system.")
        return

    try:
        # Hide webdriver property to bypass bot detection
        try:
            driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
            })
        except Exception:
            pass

        for site_key, site_url in sites_to_capture.items():
            print(f"  [{succeeded + failed + 1}/{len(sites_to_capture)}] "
                  f"Capturing {site_key} ({site_url})... ", end="", flush=True)
            try:
                driver.get(site_url)
                
                # Initial wait for page load
                time.sleep(4)
                
                # Dismiss popups/cookie banners via JavaScript
                try:
                    driver.execute_script(DISMISS_POPUPS_JS)
                except Exception:
                    pass
                
                # Apply site-specific JavaScript if available
                if site_key in SITE_SPECIFIC_JS:
                    try:
                        driver.execute_script(SITE_SPECIFIC_JS[site_key])
                    except Exception:
                        pass
                
                # Wait a moment after dismissing popups
                time.sleep(2)
                
                # Second pass — some popups appear after delay
                try:
                    driver.execute_script(DISMISS_POPUPS_JS)
                except Exception:
                    pass
                
                # Second pass of site-specific JS
                if site_key in SITE_SPECIFIC_JS:
                    try:
                        driver.execute_script(SITE_SPECIFIC_JS[site_key])
                    except Exception:
                        pass
                
                time.sleep(1)

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
    print(f"  Total sites in database: {len(TRUSTED_SITES)} (target: 50)")
    print(f"  Captured this run: {succeeded + failed}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed:    {failed}")
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Build trusted screenshot database")
    parser.add_argument("--sites", nargs="*", 
                        help="Specific site keys to re-capture (e.g., sbi icici axis)")
    parser.add_argument("--all", action="store_true",
                        help="Capture all sites")
    args = parser.parse_args()
    
    if args.sites:
        print(f"Re-capturing specific sites: {', '.join(args.sites)}")
        build_trusted_database(only_sites=args.sites)
    else:
        build_trusted_database()
