"""
Setup Verification Script
Checks if all components are properly installed and configured
"""
import sys
from pathlib import Path
def verify_imports():
    """Verify all required packages can be imported"""
    print("\n" + "="*60)
    print("VERIFYING PACKAGE IMPORTS")
    print("="*60)
    
    packages = [
        ('pandas', 'import pandas as pd'),
        ('numpy', 'import numpy as np'),
        ('sklearn', 'import sklearn'),
        ('nltk', 'import nltk'),
        ('matplotlib', 'import matplotlib.pyplot as plt'),
        ('seaborn', 'import seaborn as sns'),
        ('streamlit', 'import streamlit as st'),
        ('validators', 'import validators'),
        ('tldextract', 'import tldextract'),
        ('joblib', 'import joblib'),
    ]
    
    failed = []
    
    for package_name, import_statement in packages:
        try:
            exec(import_statement)
            print(f"✓ {package_name:20s} imported successfully")
        except ImportError as e:
            print(f"✗ {package_name:20s} FAILED: {e}")
            failed.append(package_name)
    
    if failed:
        print(f"\n✗ Failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True

def verify_nltk_data():
    """Verify NLTK data is downloaded"""
    print("\n" + "="*60)
    print("VERIFYING NLTK DATA")
    print("="*60)
    
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    try:
        # Test stopwords
        stop_words = stopwords.words('english')
        print(f"✓ Stopwords: {len(stop_words)} words loaded")
        
        # Test tokenization
        test_text = "This is a test"
        tokens = word_tokenize(test_text)
        print(f"✓ Tokenization: '{test_text}' -> {tokens}")
        
        print("\n✓ NLTK data verified!")
        return True
        
    except Exception as e:
        print(f"✗ NLTK data verification failed: {e}")
        print("Run: python src/setup_nltk.py")
        return False

def verify_datasets():
    """Verify datasets exist"""
    print("\n" + "="*60)
    print("VERIFYING DATASETS")
    print("="*60)
    
    sys.path.append(str(Path(__file__).parent.parent))
    from src.config import SMS_RAW_DATA, URL_RAW_DATA
    import pandas as pd
    
    all_good = True
    
    # Check SMS dataset
    if SMS_RAW_DATA.exists():
        df = pd.read_csv(SMS_RAW_DATA)
        print(f"✓ SMS Dataset: {len(df)} messages")
        print(f"  - Columns: {list(df.columns)}")
        print(f"  - Labels: {df['label'].value_counts().to_dict()}")
    else:
        print(f"✗ SMS Dataset not found: {SMS_RAW_DATA}")
        print("  Run: python src/download_data.py")
        all_good = False
    
    # Check URL dataset
    if URL_RAW_DATA.exists():
        df = pd.read_csv(URL_RAW_DATA)
        print(f"\n✓ URL Dataset: {len(df)} URLs")
        print(f"  - Columns: {list(df.columns)}")
        print(f"  - Labels: {df['label'].value_counts().to_dict()}")
    else:
        print(f"\n✗ URL Dataset not found: {URL_RAW_DATA}")
        print("  Run: python src/download_data.py")
        all_good = False
    
    if all_good:
        print("\n✓ All datasets verified!")
    
    return all_good

def verify_project_structure():
    """Verify project structure"""
    print("\n" + "="*60)
    print("VERIFYING PROJECT STRUCTURE")
    print("="*60)
    
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/models",
        "src/sms_detection",
        "src/url_detection",
        "src/dashboard",
        "notebooks",
        "reports"
    ]
    
    project_root = Path(__file__).parent.parent
    all_good = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
            all_good = False
    
    if all_good:
        print("\n✓ Project structure verified!")
    else:
        print("\n✗ Some directories are missing")
    
    return all_good

def main():
    """Run all verification checks"""
    print("\n" + "="*60)
    print("PHISHING DETECTION SYSTEM - SETUP VERIFICATION")
    print("="*60)
    
    results = {
        'Imports': verify_imports(),
        'NLTK Data': verify_nltk_data(),
        'Project Structure': verify_project_structure(),
        'Datasets': verify_datasets(),
    }
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for check, status in results.items():
        status_symbol = "✓" if status else "✗"
        print(f"{status_symbol} {check}")
    
    if all(results.values()):
        print("\n" + "="*60)
        print("✓ ALL CHECKS PASSED! READY TO START DEVELOPMENT!")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("✗ SOME CHECKS FAILED. FIX ISSUES ABOVE.")
        print("="*60)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)