"""
Dataset Downloader
Downloads and prepares SMS and URL datasets
"""
import pandas as pd
import urllib.request
import zipfile
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DATA_DIR, SMS_RAW_DATA, URL_RAW_DATA

def download_sms_dataset(force_download=False, use_offline=True):
    """Download SMS Spam Collection dataset from UCI
    
    Args:
        force_download: If True, download even if data exists
        use_offline: If True, use sample data instead of downloading
    """
    
    print("="*60)
    print("SMS SPAM DATASET")
    print("="*60)
    
    # Check if dataset already exists (caching)
    if SMS_RAW_DATA.exists() and not force_download:
        df = pd.read_csv(SMS_RAW_DATA)
        print(f"\n✓ Dataset already exists! Using cached data.")
        print(f"  Location: {SMS_RAW_DATA}")
        print(f"  Total messages: {len(df)}")
        return True
    
    # Use offline mode (sample data) - much faster
    if use_offline:
        print(f"\n→ Using OFFLINE mode (instant, no download required)")
        create_sample_sms_dataset()
        return True
    
    # Online mode - download from internet
    print(f"\n→ Using ONLINE mode (downloading from internet)...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = RAW_DATA_DIR / "smsspamcollection.zip"
    
    try:
        print(f"\n1. Downloading from UCI repository...")
        print(f"   URL: {url}")
        
        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\r   Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end="")
        
        urllib.request.urlretrieve(url, zip_path, reporthook=progress_hook)
        print("\n   ✓ Download complete!")
        
        print(f"\n2. Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print("   ✓ Extraction complete!")
        
        print(f"\n3. Loading dataset...")
        # Read the tab-separated file
        df = pd.read_csv(
            RAW_DATA_DIR / 'SMSSpamCollection', 
            sep='\t', 
            names=['label', 'message'],
            encoding='utf-8'
        )
        
        print(f"   ✓ Loaded {len(df)} messages")
        print(f"\n4. Dataset statistics:")
        print(f"   - Total messages: {len(df)}")
        print(f"   - Ham (legitimate): {len(df[df['label']=='ham'])} ({len(df[df['label']=='ham'])/len(df)*100:.1f}%)")
        print(f"   - Spam (phishing): {len(df[df['label']=='spam'])} ({len(df[df['label']=='spam'])/len(df)*100:.1f}%)")
        
        print(f"\n5. Saving as CSV...")
        df.to_csv(SMS_RAW_DATA, index=False)
        print(f"   ✓ Saved to: {SMS_RAW_DATA}")
        
        print(f"\n6. Sample messages:")
        print("-" * 60)
        print("\nLegitimate messages:")
        for msg in df[df['label']=='ham']['message'].head(3):
            print(f"  • {msg[:70]}...")
        
        print("\nSpam messages:")
        for msg in df[df['label']=='spam']['message'].head(3):
            print(f"  • {msg[:70]}...")
        
        # Clean up
        os.remove(zip_path)
        print(f"\n✓ SMS dataset ready!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading SMS dataset: {e}")
        print("\nCreating sample dataset instead...")
        create_sample_sms_dataset()
        return False

def create_sample_sms_dataset():
    """Create a sample SMS dataset if download fails"""
    
    sample_data = {
        'label': ['ham', 'ham', 'ham', 'spam', 'spam', 'spam', 'ham', 'spam',
                  'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
        'message': [
            "Hi, how are you? Want to grab lunch tomorrow?",
            "Meeting at 3pm in conference room B. Please confirm.",
            "Thanks for your help today! Really appreciate it.",
            "URGENT! Your account has been suspended. Click here to verify immediately: bit.ly/xyz123",
            "Congratulations! You've won a $1000 Walmart gift card. Claim now before it expires!",
            "ALERT: Unusual activity detected on your bank account. Verify identity now: secure-bank.tk",
            "Can you pick up some milk on your way home?",
            "Your package delivery failed. Update address here: fedex-delivery.com/update",
            "Don't forget about dinner tomorrow at 7pm!",
            "ACT NOW! Limited offer. Click to get your FREE iPhone 15 Pro: apple-promo.net",
            "The project deadline is next Monday. Let me know if you need help.",
            "Your PayPal account has been limited. Click to restore access: paypal-secure.org/verify",
            "Happy birthday! Hope you have an amazing day!",
            "FINAL NOTICE: Your payment is overdue. Pay now to avoid legal action: pay-now.site",
            "See you at the gym tomorrow morning!"
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(SMS_RAW_DATA, index=False)
    print(f"✓ Sample SMS dataset created: {len(df)} messages")

def create_url_dataset():
    """Create URL phishing dataset"""
    
    print("\n" + "="*60)
    print("CREATING URL DATASET")
    print("="*60)
    
    # Phishing URLs (realistic patterns)
    phishing_urls = [
        # IP-based URLs
        "http://125.56.78.90/bank/login.php",
        "http://192.168.100.50/paypal/signin",
        "https://172.16.254.1/amazon/account",
        
        # Suspicious subdomains
        "https://paypal-verify.com/secure/login",
        "https://secure-amazon.net/update-payment",
        "https://apple-security.org/verify-account",
        "https://bankofamerica-secure.tk/signin",
        "https://chase-bank-security.ml/login",
        
        # URL shorteners
        "http://bit.ly/3xK9mP2",
        "https://tinyurl.com/y5h8km9p",
        
        # Long suspicious URLs
        "https://secure-login-verify-account-paypal.com/webscr?cmd=login_verify&dispatch=5885d80a13c0db1f8e263663d3faee8d",
        "http://amazon-security-check.com/ap/signin?openid.return_to=https://www.amazon.com/",
        
        # Typosquatting
        "https://paypa1.com/signin",
        "https://arnaz0n.com/login",
        "https://g00gle.com/accounts",
        
        # Country TLD abuse
        "https://secure-paypal.tk/login",
        "https://microsoft-support.ml/verify",
        "https://apple-id.cf/signin",
        
        # Mixed patterns
        "http://secure-banking-login.com/verify?id=84729",
        "https://account-recovery-service.net/reset-password",
    ]
    
    # Legitimate URLs
    legitimate_urls = [
        "https://www.google.com",
        "https://www.facebook.com",
        "https://www.amazon.com",
        "https://www.wikipedia.org",
        "https://www.github.com",
        "https://www.youtube.com",
        "https://www.linkedin.com",
        "https://www.twitter.com",
        "https://www.instagram.com",
        "https://www.reddit.com",
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.netflix.com",
        "https://www.stackoverflow.com",
        "https://www.medium.com",
        "https://www.paypal.com",
        "https://www.chase.com",
        "https://www.bankofamerica.com",
        "https://www.wellsfargo.com",
        "https://www.citi.com",
    ]
    
    # Create DataFrame
    url_data = []
    
    for url in phishing_urls:
        url_data.append({'url': url, 'label': 'phishing'})
    
    for url in legitimate_urls:
        url_data.append({'url': url, 'label': 'legitimate'})
    
    df = pd.DataFrame(url_data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    df.to_csv(URL_RAW_DATA, index=False)
    
    print(f"\n✓ URL dataset created: {len(df)} URLs")
    print(f"  - Phishing: {len(df[df['label']=='phishing'])} ({len(df[df['label']=='phishing'])/len(df)*100:.1f}%)")
    print(f"  - Legitimate: {len(df[df['label']=='legitimate'])} ({len(df[df['label']=='legitimate'])/len(df)*100:.1f}%)")
    print(f"  - Saved to: {URL_RAW_DATA}")
    
    print(f"\nSample URLs:")
    print("-" * 60)
    print("\nPhishing URLs:")
    for url in df[df['label']=='phishing']['url'].head(3):
        print(f"  • {url}")
    
    print("\nLegitimate URLs:")
    for url in df[df['label']=='legitimate']['url'].head(3):
        print(f"  • {url}")

def main():
    """Main function to download all datasets"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download/create phishing detection datasets')
    parser.add_argument('--online', action='store_true', 
                        help='Download from internet instead of using sample data')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if data exists')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PHISHING DETECTION SYSTEM - DATA COLLECTION")
    print("="*60)
    
    if args.online:
        print("\n📡 Mode: ONLINE (downloading from internet)")
    else:
        print("\n⚡ Mode: OFFLINE (instant, using sample data)")
    
    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    download_sms_dataset(force_download=args.force, use_offline=not args.online)
    create_url_dataset()

    # Also generate the comprehensive URL detection dataset
    try:
        from src.url_detection.download_url_data import main as download_url_main
        download_url_main()
    except Exception as e:
        print(f"\n[WARN] URL detection dataset generation skipped: {e}")
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60)
    print(f"\nDatasets saved in: {RAW_DATA_DIR}")
    print(f"  1. SMS Dataset: {SMS_RAW_DATA}")
    print(f"  2. URL Dataset: {URL_RAW_DATA}")
    print(f"  3. URL Detection Dataset: {RAW_DATA_DIR / 'phishing_urls.csv'}")
    print("\n✓ Ready for preprocessing!")

if __name__ == "__main__":
    main()