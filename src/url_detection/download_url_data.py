"""
URL Dataset Downloader
Downloads and prepares the phishing URL dataset for training.
Primary source: UCI ML Repository (ARFF format).
Fallback: generates a comprehensive synthetic dataset with realistic URL patterns.
"""
import pandas as pd
import urllib.request
import os
import sys
import re
import random
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import RAW_DATA_DIR


OUTPUT_PATH = RAW_DATA_DIR / "phishing_urls.csv"


def parse_arff(filepath):
    """
    Parse a .arff file and return a DataFrame.
    The UCI phishing dataset has numeric features + class label.
    """
    data_started = False
    rows = []
    attributes = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.upper().startswith('@ATTRIBUTE'):
                parts = line.split()
                if len(parts) >= 2:
                    attributes.append(parts[1])
            elif line.upper().startswith('@DATA'):
                data_started = True
            elif data_started:
                values = line.split(',')
                rows.append(values)

    df = pd.DataFrame(rows, columns=attributes[:len(rows[0])] if rows else attributes)
    return df


def download_uci_dataset(force_download=False):
    """
    Download the UCI Phishing Websites dataset (ARFF).
    This dataset contains numeric features (not raw URLs).
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
    arff_path = RAW_DATA_DIR / "phishing_training.arff"

    if OUTPUT_PATH.exists() and not force_download:
        print(f"[OK] Dataset already exists: {OUTPUT_PATH}")
        df = pd.read_csv(OUTPUT_PATH)
        print(f"     Shape: {df.shape}")
        return df

    print("Attempting to download UCI Phishing dataset...")
    try:
        urllib.request.urlretrieve(url, str(arff_path))
        print(f"[OK] Downloaded to: {arff_path}")
        df = parse_arff(str(arff_path))
        # The UCI dataset has numeric features, not raw URLs.
        # We'll use the synthetic dataset instead for raw URL training
        print("[INFO] UCI dataset has numeric features. Using synthetic URL dataset instead.")
        return None
    except Exception as e:
        print(f"[WARN] UCI download failed: {e}")
        return None


def create_synthetic_url_dataset():
    """
    Create a comprehensive synthetic dataset of phishing and legitimate URLs.
    This provides realistic URL patterns for feature-based model training.
    """
    print("\nGenerating synthetic URL dataset...")

    random.seed(42)

    # --- Legitimate URL patterns ---
    legit_domains = [
        "google.com", "facebook.com", "youtube.com", "amazon.com", "wikipedia.org",
        "twitter.com", "instagram.com", "linkedin.com", "reddit.com", "netflix.com",
        "microsoft.com", "apple.com", "github.com", "stackoverflow.com", "medium.com",
        "sbi.co.in", "hdfcbank.com", "icicibank.com", "axisbank.com", "kotak.com",
        "flipkart.com", "myntra.com", "zomato.com", "swiggy.com", "ola.in",
        "gmail.com", "outlook.com", "yahoo.com", "bbc.com", "cnn.com",
        "nytimes.com", "theguardian.com", "reuters.com", "bloomberg.com", "forbes.com",
        "paypal.com", "stripe.com", "shopify.com", "etsy.com", "ebay.com",
        "spotify.com", "twitch.tv", "slack.com", "zoom.us", "dropbox.com",
        "adobe.com", "salesforce.com", "oracle.com", "ibm.com", "intel.com",
    ]

    legit_paths = [
        "", "/", "/about", "/contact", "/help", "/support", "/login",
        "/products", "/services", "/blog", "/news", "/careers",
        "/search?q=python", "/docs/getting-started", "/en/home",
        "/user/profile", "/shop/electronics", "/pricing",
    ]

    legitimate_urls = []
    for domain in legit_domains:
        scheme = "https"
        for path in random.sample(legit_paths, min(6, len(legit_paths))):
            legitimate_urls.append(f"{scheme}://www.{domain}{path}")
        # Also some without www
        legitimate_urls.append(f"https://{domain}")
        legitimate_urls.append(f"https://{domain}/")

    # --- Phishing URL patterns ---
    phishing_urls = []

    # Pattern 1: IP address-based URLs
    for _ in range(60):
        ip = f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        paths = ["/login", "/verify", "/secure", "/account", "/update",
                 "/banking/confirm", "/signin", "/password-reset",
                 f"/index.php?id={random.randint(1000,9999)}"]
        phishing_urls.append(f"http://{ip}{random.choice(paths)}")

    # Pattern 2: Brand-in-subdomain spoofing
    brands = ["paypal", "google", "facebook", "amazon", "netflix", "apple", "microsoft", "sbi", "hdfc", "icici"]
    fake_domains = [
        "secure-login.xyz", "verify-now.tk", "account-check.ml", "update-info.cf",
        "login-secure.com", "verify-account.net", "secure-portal.org", "auth-check.info",
        "banking-update.xyz", "credential-verify.tk", "urgent-alert.ml",
        "securesite.xyz", "loginportal.tk", "verifyaccount.ml",
    ]
    for brand in brands:
        for fake_domain in random.sample(fake_domains, 6):
            phishing_urls.append(f"http://{brand}.{fake_domain}/login")
            phishing_urls.append(f"http://{brand}-secure.{fake_domain}/verify")

    # Pattern 3: Long, obfuscated URLs with many subdomains
    for _ in range(50):
        subs = '.'.join([f"{''.join(random.choices('abcdefghijklmnop0123456789', k=random.randint(4,10)))}"
                         for _ in range(random.randint(3, 6))])
        tld = random.choice(['.xyz', '.tk', '.ml', '.cf', '.ga', '.gq'])
        phishing_urls.append(f"http://{subs}{tld}/login/verify/account")

    # Pattern 4: @ sign abuse (redirect to attacker)
    for _ in range(40):
        legit = random.choice(["www.paypal.com", "www.sbi.co.in", "www.google.com"])
        attacker_ip = f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        phishing_urls.append(f"http://{legit}@{attacker_ip}/login")

    # Pattern 5: Suspicious keywords in path
    suspicious_paths = [
        "/secure/login/verify/account/update",
        "/banking/confirm/password",
        "/credential/verify/suspended",
        "/urgent/alert/account/signin",
        "/verify-account?token=abc123&redirect=true",
        "/login.php?user=admin&password=reset",
        "/confirm/banking/update?id=12345",
    ]
    for _ in range(50):
        fake = f"{''.join(random.choices('abcdefghijklmnop', k=8))}.{random.choice(['xyz', 'tk', 'ml', 'cf'])}"
        phishing_urls.append(f"http://{fake}{random.choice(suspicious_paths)}")

    # Pattern 6: HTTP-in-domain (deception)
    for _ in range(30):
        base = random.choice(["http-secure", "https-verify", "http-login", "http-banking"])
        tld = random.choice([".com", ".xyz", ".tk", ".net"])
        phishing_urls.append(f"http://{base}{tld}/verify")

    # Pattern 7: URL shorteners (used to mask real destination)
    shorteners = ["bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd"]
    for _ in range(40):
        code = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=random.randint(5, 8)))
        phishing_urls.append(f"http://{random.choice(shorteners)}/{code}")

    # Pattern 8: Port-based attacks
    for _ in range(30):
        ip = f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
        port = random.choice([8080, 8443, 3000, 4443, 9090])
        phishing_urls.append(f"http://{ip}:{port}/login")

    # Pattern 9: Heavy URL encoding / percent abuse
    for _ in range(30):
        encoded_parts = '%'.join([f"{random.randint(0x20,0x7E):02x}" for _ in range(random.randint(5, 15))])
        fake = f"{''.join(random.choices('abcdefghijklmnop', k=6))}.xyz"
        phishing_urls.append(f"http://{fake}/login/%{encoded_parts}")

    # Pattern 10: Double-slash redirect
    for _ in range(30):
        legit = random.choice(["www.google.com", "www.paypal.com", "www.facebook.com"])
        attacker = f"{''.join(random.choices('abcdefghijklmnop', k=8))}.xyz"
        phishing_urls.append(f"http://{legit}//{attacker}/login")

    # Combine and label
    all_urls = (
        [(url, 0) for url in legitimate_urls] +
        [(url, 1) for url in phishing_urls]
    )

    # Shuffle
    random.shuffle(all_urls)

    df = pd.DataFrame(all_urls, columns=['url', 'label'])

    # Save
    os.makedirs(str(RAW_DATA_DIR), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[OK] Synthetic URL dataset created: {OUTPUT_PATH}")
    print(f"     Total samples: {len(df)}")
    print(f"     Legitimate (0): {(df['label'] == 0).sum()}")
    print(f"     Phishing (1):   {(df['label'] == 1).sum()}")
    print(f"     Imbalance ratio: {(df['label'] == 1).sum() / max((df['label'] == 0).sum(), 1):.2f}")

    return df


def main():
    """Main function to download/generate URL dataset."""
    print("\n" + "=" * 60)
    print("URL DATASET DOWNLOADER")
    print("=" * 60)

    os.makedirs(str(RAW_DATA_DIR), exist_ok=True)

    # Check if dataset already exists
    if OUTPUT_PATH.exists():
        print(f"\n[OK] Dataset already exists: {OUTPUT_PATH}")
        df = pd.read_csv(OUTPUT_PATH)
        print(f"     Shape: {df.shape}")
        print(f"     Columns: {list(df.columns)}")
        print(f"     Label distribution:\n{df['label'].value_counts().to_string()}")
        return df

    # Try UCI download first
    df = download_uci_dataset()

    # Fallback to synthetic dataset
    if df is None:
        df = create_synthetic_url_dataset()

    print("\n" + "=" * 60)
    print("[OK] URL DATASET READY!")
    print("=" * 60)

    return df


if __name__ == "__main__":
    main()
