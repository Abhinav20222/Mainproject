"""
Fix the pickled model to have proper module references
"""
import sys
from pathlib import Path

# Set up path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from src.sms_detection.feature_extraction import FeatureExtractor
from src.sms_detection.preprocessing import SMSPreprocessor
from src.config import FEATURE_EXTRACTOR_PATH, SMS_MODEL_PATH

print("=" * 50)
print("Fixing model pickle references...")
print("=" * 50)

# Load and re-save feature extractor
print("\n1. Loading feature extractor...")
fe = joblib.load(FEATURE_EXTRACTOR_PATH)
print(f"   Type: {type(fe)}")

# Re-save with proper module reference
print("2. Re-saving with correct module path...")
backup_path = FEATURE_EXTRACTOR_PATH.with_suffix('.pkl.backup')
FEATURE_EXTRACTOR_PATH.rename(backup_path)
print(f"   Backup saved to: {backup_path}")

joblib.dump(fe, FEATURE_EXTRACTOR_PATH)
print(f"   Saved to: {FEATURE_EXTRACTOR_PATH}")

print("\n✅ Done! The model should now load correctly.")
print("\nYou can now run:")
print("  python -m src.api_fast")
