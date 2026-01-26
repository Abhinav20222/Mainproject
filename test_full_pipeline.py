"""
Full Pipeline End-to-End Test Script
Tests all API endpoints: health, analyze, analyze-url, full-scan
Run from project root: python test_full_pipeline.py
Requires: API server running on localhost:5000
"""
import requests
import json
import time
import sys

API_URL = "http://localhost:5000"

TESTS_PASSED = 0
TESTS_FAILED = 0


def test(name, func):
    """Run a test function and report results."""
    global TESTS_PASSED, TESTS_FAILED
    print(f"\n── {name} ──")
    try:
        func()
        TESTS_PASSED += 1
        print(f"   ✓ PASSED")
    except Exception as e:
        TESTS_FAILED += 1
        print(f"   ✗ FAILED: {e}")


def pretty(data):
    """Pretty-print JSON."""
    print(f"   {json.dumps(data, indent=2, default=str)[:500]}")


def test_health():
    """Test health endpoint."""
    r = requests.get(f"{API_URL}/api/health", timeout=5)
    assert r.status_code == 200, f"Status: {r.status_code}"
    data = r.json()
    assert data['status'] == 'online', f"Status: {data['status']}"
    print(f"   API status: {data['status']}")
    print(f"   SMS model: {data['models_loaded']}")
    print(f"   URL model: {data.get('url_model_loaded')}")


def test_sms_analyze():
    """Test SMS analysis endpoint."""
    r = requests.post(f"{API_URL}/api/analyze", json={
        "message": "URGENT! Your bank account has been suspended. Click here: bit.ly/scam123"
    }, timeout=10)
    assert r.status_code == 200, f"Status: {r.status_code}"
    data = r.json()
    assert 'is_phishing' in data, "Missing is_phishing field"
    assert 'threat_score' in data, "Missing threat_score field"
    print(f"   Phishing: {data['is_phishing']} | Score: {data['threat_score']}")
    print(f"   Time: {data.get('processing_time_ms', 'N/A')}ms")


def test_sms_safe():
    """Test SMS analysis with safe message."""
    r = requests.post(f"{API_URL}/api/analyze", json={
        "message": "Hi! Want to grab lunch tomorrow?"
    }, timeout=10)
    assert r.status_code == 200
    data = r.json()
    print(f"   Phishing: {data['is_phishing']} | Score: {data['threat_score']}")


def test_url_analyze():
    """Test URL analysis endpoint."""
    r = requests.post(f"{API_URL}/api/analyze-url", json={
        "url": "http://paypal.secure-login.xyz/verify/account"
    }, timeout=10)
    assert r.status_code == 200, f"Status: {r.status_code}"
    data = r.json()
    assert data.get('success') == True, f"success=False: {data.get('error')}"
    assert 'threat_score' in data, "Missing threat_score"
    assert 'risk_level' in data, "Missing risk_level"
    print(f"   Phishing: {data['is_phishing']} | Score: {data['threat_score']} | Risk: {data['risk_level']}")
    if data.get('top_risk_features'):
        print(f"   Top risks: {data['top_risk_features'][:3]}")


def test_url_safe():
    """Test URL analysis with safe URL."""
    r = requests.post(f"{API_URL}/api/analyze-url", json={
        "url": "https://www.google.com"
    }, timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert data.get('success') == True
    print(f"   Phishing: {data['is_phishing']} | Score: {data['threat_score']} | Risk: {data['risk_level']}")


def test_url_missing_field():
    """Test URL endpoint with missing field."""
    r = requests.post(f"{API_URL}/api/analyze-url", json={}, timeout=5)
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print(f"   Correctly returned 400 for missing url field")


def test_full_scan():
    """Test full scan endpoint."""
    r = requests.post(f"{API_URL}/api/full-scan", json={
        "message": "Your PayPal account has been limited. Click to restore: paypal-verify.tk",
        "url": "http://paypal-verify.tk/login",
        "include_visual": False,
    }, timeout=15)
    assert r.status_code == 200, f"Status: {r.status_code}"
    data = r.json()
    assert data.get('success') == True, f"success=False: {data.get('error')}"
    assert 'combined_threat_score' in data, "Missing combined_threat_score"
    assert 'risk_level' in data, "Missing risk_level"
    print(f"   Combined score: {data['combined_threat_score']} | Risk: {data['risk_level']}")
    print(f"   Analyses: {data.get('analyses_performed')}")
    print(f"   Weights: {data.get('score_weights')}")
    print(f"   Time: {data.get('total_analysis_time_ms', 'N/A')}ms")


def test_full_scan_url_only():
    """Test full scan with URL only."""
    r = requests.post(f"{API_URL}/api/full-scan", json={
        "url": "https://www.github.com",
    }, timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert data.get('success') == True
    print(f"   Combined score: {data['combined_threat_score']} | Risk: {data['risk_level']}")
    print(f"   Analyses: {data.get('analyses_performed')}")


def test_heatmap_404():
    """Test heatmap endpoint returns 404 when no heatmap exists."""
    r = requests.get(f"{API_URL}/api/heatmap", timeout=5)
    # Could be 200 (if heatmap exists) or 404
    assert r.status_code in [200, 404], f"Unexpected status: {r.status_code}"
    print(f"   Heatmap status: {r.status_code}")


def main():
    print("\n" + "=" * 60)
    print("PhishGuard AI — FULL PIPELINE E2E TESTS")
    print("=" * 60)
    print(f"API: {API_URL}")

    # Check if API is reachable
    try:
        requests.get(f"{API_URL}/api/health", timeout=3)
    except requests.ConnectionError:
        print("\n✗ API server is not running!")
        print("  Start it with: python -m src.api")
        sys.exit(1)

    test("Health Check", test_health)
    test("SMS Analysis (Phishing)", test_sms_analyze)
    test("SMS Analysis (Safe)", test_sms_safe)
    test("URL Analysis (Phishing)", test_url_analyze)
    test("URL Analysis (Safe)", test_url_safe)
    test("URL Missing Field", test_url_missing_field)
    test("Full Scan (SMS + URL)", test_full_scan)
    test("Full Scan (URL Only)", test_full_scan_url_only)
    test("Heatmap Endpoint", test_heatmap_404)

    print("\n\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"  Passed: {TESTS_PASSED}")
    print(f"  Failed: {TESTS_FAILED}")
    print(f"  Total:  {TESTS_PASSED + TESTS_FAILED}")
    print("=" * 60)

    if TESTS_FAILED == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {TESTS_FAILED} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
