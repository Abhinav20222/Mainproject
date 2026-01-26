"""
URL Feature Extractor Module
Extracts 30 lexical and structural features from raw URL strings
for phishing detection. All features are computed locally — no external APIs.
"""
import re
import math
import string
from urllib.parse import urlparse, parse_qs
import pandas as pd
import numpy as np


# Known URL shortener domains
SHORTENER_DOMAINS = {
    'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly',
    'is.gd', 'buff.ly', 'shorte.st', 'tiny.cc', 'rb.gy',
    'cutt.ly', 'shorturl.at'
}

# Suspicious keywords commonly found in phishing URLs
SUSPICIOUS_WORDS = [
    'login', 'signin', 'verify', 'secure', 'account', 'update',
    'banking', 'confirm', 'password', 'credential', 'suspended',
    'urgent', 'alert'
]

# Known brand names used in subdomain spoofing
BRAND_NAMES = [
    'paypal', 'google', 'facebook', 'amazon', 'netflix',
    'apple', 'microsoft', 'sbi', 'hdfc', 'icici'
]

# IPv4 pattern
IPV4_PATTERN = re.compile(
    r'^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$'
)


class URLFeatureExtractor:
    """
    Extracts 30 lexical and structural features from a raw URL string.
    
    Features cover:
    - Length metrics (URL, hostname, path, query)
    - Character counts (dots, hyphens, digits, special chars)
    - Structural patterns (IP address, HTTPS, port, redirects)
    - Keyword signals (suspicious words, brand spoofing)
    - Entropy and ratio features
    """

    FEATURE_NAMES = [
        'url_length', 'hostname_length', 'path_length', 'query_length',
        'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
        'num_at_signs', 'num_question_marks', 'num_equals', 'num_percent',
        'num_digits',
        'has_ip_address', 'has_https', 'has_http_in_domain',
        'num_subdomains', 'has_port', 'has_double_slash_redirect',
        'domain_has_digits', 'tld_length', 'is_shortened',
        'has_suspicious_words', 'has_brand_in_subdomain',
        'hostname_entropy',
        'digit_to_letter_ratio', 'special_char_ratio',
        'path_depth'
    ]

    def __init__(self):
        """Initialize the URL feature extractor."""
        pass

    @staticmethod
    def _shannon_entropy(text):
        """Calculate Shannon entropy of a string."""
        if not text:
            return 0.0
        prob = {c: text.count(c) / len(text) for c in set(text)}
        return -sum(p * math.log2(p) for p in prob.values())

    @staticmethod
    def _get_registered_domain(hostname):
        """
        Extract the registered domain (domain + TLD) from hostname.
        e.g. 'login.paypal.secure-site.com' -> 'secure-site.com'
        """
        parts = hostname.split('.')
        if len(parts) >= 2:
            return '.'.join(parts[-2:])
        return hostname

    @staticmethod
    def _get_tld(hostname):
        """Extract TLD from hostname."""
        parts = hostname.split('.')
        if len(parts) >= 2:
            return parts[-1]
        return ''

    @staticmethod
    def _get_subdomains(hostname):
        """Extract subdomain portion of hostname."""
        parts = hostname.split('.')
        if len(parts) > 2:
            return '.'.join(parts[:-2])
        return ''

    def extract(self, url):
        """
        Extract all 30 features from a single URL string.
        
        Args:
            url (str): Raw URL string
            
        Returns:
            dict: Dictionary of feature_name -> value
        """
        # Ensure URL has a scheme for proper parsing
        url_str = url.strip()
        if not url_str.startswith(('http://', 'https://', 'ftp://')):
            url_str = 'http://' + url_str

        try:
            parsed = urlparse(url_str)
        except Exception:
            parsed = urlparse('http://invalid.url')

        hostname = (parsed.hostname or '').lower()
        path = parsed.path or ''
        query = parsed.query or ''
        scheme = (parsed.scheme or '').lower()
        url_lower = url.lower()

        # --- Length Features ---
        url_length = len(url)
        hostname_length = len(hostname)
        path_length = len(path)
        query_length = len(query)

        # --- Character Count Features ---
        num_dots = url.count('.')
        num_hyphens = url.count('-')
        num_underscores = url.count('_')
        num_slashes = url.count('/')
        num_at_signs = url.count('@')
        num_question_marks = url.count('?')
        num_equals = url.count('=')
        num_percent = url.count('%')
        num_digits = sum(c.isdigit() for c in url)

        # --- Structural / Pattern Features ---
        has_ip_address = 1 if IPV4_PATTERN.match(hostname) else 0
        has_https = 1 if scheme == 'https' else 0

        # Check if "http" appears in the domain name itself (not the scheme)
        domain_without_scheme = url_lower.split('://', 1)[-1] if '://' in url_lower else url_lower
        has_http_in_domain = 1 if 'http' in hostname else 0

        # Number of subdomains (dots in hostname minus 1, min 0)
        hostname_dots = hostname.count('.')
        num_subdomains = max(0, hostname_dots - 1)

        # Port detection
        has_port = 1 if parsed.port is not None else 0

        # Double slash redirect: '//' after the scheme
        after_scheme = url_str.split('://', 1)[-1] if '://' in url_str else url_str
        has_double_slash_redirect = 1 if '//' in after_scheme else 0

        # Digits in domain
        domain_has_digits = 1 if any(c.isdigit() for c in hostname) else 0

        # TLD length
        tld = self._get_tld(hostname)
        tld_length = len(tld)

        # URL shortener detection
        registered_domain = self._get_registered_domain(hostname)
        is_shortened = 1 if registered_domain in SHORTENER_DOMAINS or hostname in SHORTENER_DOMAINS else 0

        # --- Keyword Features ---
        has_suspicious_words = 1 if any(word in url_lower for word in SUSPICIOUS_WORDS) else 0

        # Brand in subdomain but NOT in registered domain
        subdomains = self._get_subdomains(hostname)
        has_brand_in_subdomain = 0
        if subdomains:
            for brand in BRAND_NAMES:
                if brand in subdomains.lower() and brand not in registered_domain.lower():
                    has_brand_in_subdomain = 1
                    break

        # --- Entropy Feature ---
        hostname_entropy = self._shannon_entropy(hostname)

        # --- Ratio Features ---
        num_letters = sum(c.isalpha() for c in url)
        digit_to_letter_ratio = num_digits / max(num_letters, 1)

        num_special = sum(1 for c in url if not c.isalnum())
        special_char_ratio = num_special / max(len(url), 1)

        # --- Path Depth ---
        path_segments = [seg for seg in path.split('/') if seg]
        path_depth = len(path_segments)

        return {
            'url_length': url_length,
            'hostname_length': hostname_length,
            'path_length': path_length,
            'query_length': query_length,
            'num_dots': num_dots,
            'num_hyphens': num_hyphens,
            'num_underscores': num_underscores,
            'num_slashes': num_slashes,
            'num_at_signs': num_at_signs,
            'num_question_marks': num_question_marks,
            'num_equals': num_equals,
            'num_percent': num_percent,
            'num_digits': num_digits,
            'has_ip_address': has_ip_address,
            'has_https': has_https,
            'has_http_in_domain': has_http_in_domain,
            'num_subdomains': num_subdomains,
            'has_port': has_port,
            'has_double_slash_redirect': has_double_slash_redirect,
            'domain_has_digits': domain_has_digits,
            'tld_length': tld_length,
            'is_shortened': is_shortened,
            'has_suspicious_words': has_suspicious_words,
            'has_brand_in_subdomain': has_brand_in_subdomain,
            'hostname_entropy': round(hostname_entropy, 4),
            'digit_to_letter_ratio': round(digit_to_letter_ratio, 4),
            'special_char_ratio': round(special_char_ratio, 4),
            'path_depth': path_depth,
        }

    def extract_batch(self, urls):
        """
        Extract features for a list of URLs.
        
        Args:
            urls (list): List of URL strings
            
        Returns:
            pd.DataFrame: DataFrame with one row per URL and 30 feature columns
        """
        records = []
        for url in urls:
            try:
                features = self.extract(url)
            except Exception:
                # Return zeros for unparseable URLs
                features = {name: 0 for name in self.FEATURE_NAMES}
            records.append(features)

        return pd.DataFrame(records, columns=self.FEATURE_NAMES)

    def get_feature_names(self):
        """Return ordered list of feature names."""
        return list(self.FEATURE_NAMES)


# Quick test
if __name__ == "__main__":
    extractor = URLFeatureExtractor()

    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/sbi/login?user=admin",
        "http://paypal.secure-login.xyz/verify/account",
        "https://bit.ly/3xYz123",
        "http://sbi.login-secure.com:8080/banking/update@@confirm",
    ]

    print("=" * 70)
    print("URL FEATURE EXTRACTOR — TEST")
    print("=" * 70)

    for url in test_urls:
        features = extractor.extract(url)
        print(f"\nURL: {url}")
        print(f"  Length: {features['url_length']}, Hostname: {features['hostname_length']}")
        print(f"  IP: {features['has_ip_address']}, HTTPS: {features['has_https']}, Shortened: {features['is_shortened']}")
        print(f"  Suspicious words: {features['has_suspicious_words']}, Brand spoof: {features['has_brand_in_subdomain']}")
        print(f"  Entropy: {features['hostname_entropy']}, Path depth: {features['path_depth']}")

    # Batch test
    df = extractor.extract_batch(test_urls)
    print(f"\nBatch result shape: {df.shape}")
    print(f"Feature names ({len(extractor.get_feature_names())}): {extractor.get_feature_names()}")
    print("\n✓ URL Feature Extractor working!")
