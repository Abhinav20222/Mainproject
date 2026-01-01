"""
Configuration file for Phishing Detection System
Contains all paths, constants, and settings
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
SMS_RAW_DATA = RAW_DATA_DIR / "sms_data.csv"
SMS_PROCESSED_DATA = PROCESSED_DATA_DIR / "sms_processed.csv"
SMS_FEATURES_DATA = PROCESSED_DATA_DIR / "sms_features.csv"

URL_RAW_DATA = RAW_DATA_DIR / "url_data.csv"
URL_PROCESSED_DATA = PROCESSED_DATA_DIR / "url_processed.csv"

# Model paths
FEATURE_EXTRACTOR_PATH = MODELS_DIR / "feature_extractor.pkl"
SMS_MODEL_PATH = MODELS_DIR / "sms_classifier.pkl"
URL_MODEL_PATH = MODELS_DIR / "url_classifier.pkl"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_TFIDF_FEATURES = 500

# Feature extraction settings
NGRAM_RANGE = (1, 2)  # unigrams and bigrams
MIN_DF = 2
MAX_DF = 0.8

# Keywords for feature extraction
URGENCY_KEYWORDS = [
    'urgent', 'immediately', 'now', 'asap', 'hurry', 'limited', 
    'expire', 'today', 'fast', 'quick', 'act now', 'limited time'
]

FINANCIAL_KEYWORDS = [
    'bank', 'account', 'credit', 'debit', 'card', 'money', 'cash', 
    'payment', 'transaction', 'dollar', 'prize', 'won', 'reward',
    'refund', 'tax', 'irs', 'paypal'
]

ACTION_KEYWORDS = [
    'click', 'call', 'reply', 'confirm', 'verify', 'update', 
    'claim', 'redeem', 'activate', 'download', 'install'
]

THREAT_KEYWORDS = [
    'suspend', 'block', 'locked', 'unauthorized', 'unusual activity',
    'security alert', 'compromised', 'fraud'
]

# Threat score thresholds
THREAT_LEVELS = {
    'low': (0, 30),
    'medium': (30, 60),
    'high': (60, 85),
    'critical': (85, 100)
}

# Color codes for dashboard
THREAT_COLORS = {
    'low': '#28a745',      # Green
    'medium': '#ffc107',   # Yellow
    'high': '#fd7e14',     # Orange
    'critical': '#dc3545'  # Red
}

print("✓ Configuration loaded successfully")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Directory: {DATA_DIR}")