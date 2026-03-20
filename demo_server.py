"""
demo_server.py
==============
One-command local server to host the SBI phishing clone page for demo.

Usage:
    python demo_server.py

Then open http://localhost:8080  in your browser to preview it.
During the Full Scan demo, use message:
    "URGENT: Your SBI account has been BLOCKED. Verify KYC at http://localhost:8080"

The Selenium headless Chrome in screenshot_capturer.py will load http://localhost:8080,
take a screenshot, and the visual comparison pipeline (pHash + SSIM) will run
against the trusted SBI screenshot in data/trusted_screenshots/sbi.png
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8080
SERVE_DIR = Path(__file__).parent / "demo_phishing_page"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVE_DIR), **kwargs)

    def log_message(self, format, *args):
        print(f"[Demo Server] {self.address_string()} — {format % args}")


def main():
    print("=" * 55)
    print("  PHISHGUARD DEMO SERVER")
    print("=" * 55)
    print(f"  Serving:  {SERVE_DIR}")
    print(f"  URL:      http://localhost:{PORT}")
    print(f"  Purpose:  SBI phishing clone for visual comparison demo")
    print()
    print("  DEMO MESSAGE to paste in Full Scan:")
    print()
    print('  "URGENT: Your SBI account has been BLOCKED due to')
    print('  suspicious activity. Verify your KYC NOW or lose')
    print('  access permanently. Visit http://localhost:8080')
    print('  to prevent account closure!"')
    print()
    print("  [OK] Check 'Include Visual Spoofing Analysis' in the app")
    print("  [OK] Click Launch Full Scan")
    print()
    print("  Press Ctrl+C to stop the server.")
    print("=" * 55)

    if not SERVE_DIR.exists():
        print(f"[ERROR] demo_phishing_page/ not found at {SERVE_DIR}")
        print("  Make sure demo_phishing_page/index.html exists.")
        return

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[Demo Server] Stopped.")


if __name__ == "__main__":
    main()
