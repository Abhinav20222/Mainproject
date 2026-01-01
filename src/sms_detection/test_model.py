"""
Model Testing and Validation Script
Tests the trained model with various scenarios
"""
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import SMS_FEATURES_DATA, SMS_MODEL_PATH
from src.sms_detection.predict import SMSPredictor

def test_edge_cases():
    """Test model with edge cases"""
    print("\n" + "="*70)
    print("TESTING EDGE CASES")
    print("="*70)
    
    predictor = SMSPredictor()
    
    edge_cases = {
        "Empty message": "",
        "Very short": "OK",
        "Only numbers": "1234567890",
        "Only special chars": "!@#$%^&*()",
        "Mixed languages": "Hello नमस्ते مرحبا",
        "Very long message": "This is a very long message " * 20,
        "All caps": "THIS IS ALL CAPS MESSAGE",
        "URL only": "https://example.com",
    }
    
    for name, message in edge_cases.items():
        print(f"\n{name}: '{message[:50]}...'")
        try:
            result = predictor.predict(message)
            print(f"  Result: {result['prediction']} ({result['confidence']:.1%})")
        except Exception as e:
            print(f"  Error: {e}")

def test_accuracy_on_test_set():
    """Test model accuracy on held-out test set"""
    print("\n" + "="*70)
    print("TESTING ON TEST SET")
    print("="*70)
    
    # Load features and model
    df = pd.read_csv(SMS_FEATURES_DATA)
    model = joblib.load(SMS_MODEL_PATH)
    
    # Use last 20% as test
    test_size = int(len(df) * 0.2)
    test_df = df.tail(test_size)
    
    X_test = test_df.drop('label_encoded', axis=1).abs()
    y_test = test_df['label_encoded']
    
    # Predict
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = (predictions == y_test).mean()
    
    print(f"\n✓ Test set size: {len(y_test)}")
    print(f"✓ Accuracy: {accuracy:.2%}")
    print(f"✓ Correct: {sum(predictions == y_test)}/{len(y_test)}")
    
    # Show some misclassifications
    misclassified = test_df[predictions != y_test]
    if len(misclassified) > 0:
        print(f"\n⚠️  Misclassified examples ({len(misclassified)}):")
        # Note: We don't have original messages here, just features

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("MODEL TESTING SUITE")
    print("="*70)
    
    try:
        # Test edge cases
        test_edge_cases()
        
        # Test accuracy
        test_accuracy_on_test_set()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS COMPLETE")
        print("="*70)
        
    except FileNotFoundError:
        print("\n✗ Model not found. Please train first:")
        print("  python src/sms_detection/train_model.py")
        sys.exit(1)

if __name__ == "__main__":
    main()