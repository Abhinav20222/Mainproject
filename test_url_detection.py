"""
URL Detection Test Script
Tests the URL predictor with known phishing and legitimate URLs.
Run from project root: python test_url_detection.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_url_predictor():
    """Test URL predictor with sample URLs."""
    from src.url_detection.url_predictor import URLPredictor

    predictor = URLPredictor()

    phishing_urls = [
        "http://192.168.1.1/sbi/login?user=admin",
        "http://paypal.secure-login.xyz/verify/account",
        "http://www.google.com@192.168.1.1/login",
        "http://bit.ly/3xK9mP2",
        "http://sbi.login-secure.com:8080/banking/update@@confirm",
    ]

    legitimate_urls = [
        "https://www.google.com",
        "https://www.sbi.co.in",
        "https://www.github.com/microsoft/vscode",
        "https://www.wikipedia.org/wiki/Python",
        "https://www.amazon.com/dp/B09V3KXJPB",
    ]

    print("\n" + "=" * 70)
    print("URL DETECTION — TEST SUITE")
    print("=" * 70)

    # Test phishing URLs
    print("\n──── PHISHING URLs ────")
    phishing_correct = 0
    for url in phishing_urls:
        start = time.time()
        result = predictor.predict(url)
        elapsed = (time.time() - start) * 1000

        label = "✓ PHISHING" if result['is_phishing'] else "✗ MISSED"
        if result['is_phishing']:
            phishing_correct += 1

        print(f"\n  URL:   {url}")
        print(f"  {label} | Score: {result['threat_score']:.4f} | Risk: {result['risk_level']} | {elapsed:.1f}ms")
        if result['top_risk_features']:
            print(f"  Risks: {', '.join(result['top_risk_features'][:3])}")

    # Test legitimate URLs
    print("\n\n──── LEGITIMATE URLs ────")
    legit_correct = 0
    for url in legitimate_urls:
        start = time.time()
        result = predictor.predict(url)
        elapsed = (time.time() - start) * 1000

        label = "✓ SAFE" if not result['is_phishing'] else "✗ FALSE ALARM"
        if not result['is_phishing']:
            legit_correct += 1

        print(f"\n  URL:   {url}")
        print(f"  {label} | Score: {result['threat_score']:.4f} | Risk: {result['risk_level']} | {elapsed:.1f}ms")

    # Summary
    total = len(phishing_urls) + len(legitimate_urls)
    correct = phishing_correct + legit_correct
    print("\n\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"  Phishing detection: {phishing_correct}/{len(phishing_urls)}")
    print(f"  Legitimate correct: {legit_correct}/{len(legitimate_urls)}")
    print(f"  Overall accuracy:   {correct}/{total} ({correct/total*100:.1f}%)")
    print("=" * 70)

    # Assertions
    assert phishing_correct >= 3, f"Expected at least 3/5 phishing detected, got {phishing_correct}"
    assert legit_correct >= 3, f"Expected at least 3/5 legitimate correct, got {legit_correct}"
    print("\n✓ All assertions passed!")


def test_feature_extractor():
    """Test the URL feature extractor directly."""
    from src.url_detection.url_feature_extractor import URLFeatureExtractor

    extractor = URLFeatureExtractor()

    print("\n\n" + "=" * 70)
    print("URL FEATURE EXTRACTOR — TEST")
    print("=" * 70)

    test_url = "http://paypal.secure-login.xyz:8080/verify/account?user=admin"
    features = extractor.extract(test_url)

    # Verify feature count
    assert len(features) == len(extractor.FEATURE_NAMES), \
        f"Expected {len(extractor.FEATURE_NAMES)} features, got {len(features)}"

    # Verify specific features
    assert features['has_ip_address'] == 0, "Should not detect IP"
    assert features['has_suspicious_words'] == 1, "Should detect 'verify' + 'account'"
    assert features['has_brand_in_subdomain'] == 1, "Should detect 'paypal' in subdomain"
    assert features['has_port'] == 1, "Should detect port 8080"
    assert features['has_https'] == 0, "Should detect HTTP (not HTTPS)"

    print(f"  Test URL: {test_url}")
    print(f"  Features extracted: {len(features)}")
    print(f"  IP: {features['has_ip_address']}, HTTPS: {features['has_https']}")
    print(f"  Brand spoof: {features['has_brand_in_subdomain']}, Port: {features['has_port']}")
    print(f"  Suspicious: {features['has_suspicious_words']}")

    # Batch test
    urls = ["https://www.google.com", "http://evil.xyz/login"]
    df = extractor.extract_batch(urls)
    assert df.shape == (2, len(extractor.FEATURE_NAMES)), f"Batch shape mismatch: {df.shape}"

    print(f"  Batch test ({len(urls)} URLs): OK")
    print("\n✓ Feature extractor tests passed!")


if __name__ == "__main__":
    test_feature_extractor()
    test_url_predictor()
    print("\n\n✅ ALL URL DETECTION TESTS PASSED!")
