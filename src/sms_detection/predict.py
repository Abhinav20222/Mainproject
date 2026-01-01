"""
SMS Prediction Module
Simple interface for making predictions on new messages
"""
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import SMS_MODEL_PATH, FEATURE_EXTRACTOR_PATH
from src.sms_detection.preprocessing import SMSPreprocessor

class SMSPredictor:
    """
    SMS Phishing Predictor
    Easy-to-use interface for predictions
    """
    
    def __init__(self):
        """Load trained model and preprocessor"""
        print("Loading SMS Phishing Detector...")
        
        try:
            # Load model
            self.model = joblib.load(SMS_MODEL_PATH)
            print(f"✓ Model loaded from: {SMS_MODEL_PATH}")
            
            # Load feature extractor
            self.feature_extractor = joblib.load(FEATURE_EXTRACTOR_PATH)
            print(f"✓ Feature extractor loaded")
            
            # Initialize preprocessor
            self.preprocessor = SMSPreprocessor(use_stemming=True)
            print(f"✓ Preprocessor initialized")
            
            print("✓ SMS Phishing Detector ready!\n")
            
        except FileNotFoundError as e:
            print(f"✗ Error: Could not load model files")
            print(f"  Make sure you've run: python src/sms_detection/train_model.py")
            raise e
    
    def predict(self, message):
        """
        Predict if a message is spam/phishing
        
        Args:
            message (str): SMS message text
            
        Returns:
            dict: Prediction results
        """
        # Create dataframe
        df = pd.DataFrame({'message': [message]})
        
        # Preprocess
        df_processed = self.preprocessor.preprocess_dataset(df, verbose=False)
        
        # Extract features
        features = self.feature_extractor.transform(df_processed)
        
        # Handle negative values
        features = features.abs()
        
        # Predict
        prediction = self.model.predict(features)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[prediction]
        else:
            confidence = 1.0
        
        # Extract key features that triggered detection
        feature_importance = self._get_feature_importance(df_processed.iloc[0])
        
        result = {
            'message': message,
            'prediction': 'spam' if prediction == 1 else 'ham',
            'label': prediction,
            'confidence': float(confidence),
            'threat_score': int(confidence * 100) if prediction == 1 else int((1 - confidence) * 100),
            'features': feature_importance
        }
        
        return result
    
    def _get_feature_importance(self, processed_row):
        """Extract important features from processed message"""
        features = {
            'urgency_keywords': int(processed_row.get('urgency_count', 0)),
            'financial_keywords': int(processed_row.get('financial_count', 0)),
            'action_keywords': int(processed_row.get('action_count', 0)),
            'threat_keywords': int(processed_row.get('threat_count', 0)),
            'has_url': bool(processed_row.get('has_url', 0)),
            'has_phone': bool(processed_row.get('has_phone', 0)),
            'message_length': int(processed_row.get('message_length', 0)),
            'uppercase_ratio': float(processed_row.get('uppercase_ratio', 0)),
            'excessive_caps': bool(processed_row.get('excessive_caps', 0))
        }
        
        return features
    
    def predict_batch(self, messages):
        """
        Predict multiple messages at once
        
        Args:
            messages (list): List of message strings
            
        Returns:
            list: List of prediction results
        """
        results = []
        for message in messages:
            try:
                result = self.predict(message)
                results.append(result)
            except Exception as e:
                results.append({
                    'message': message,
                    'error': str(e)
                })
        
        return results
    
    def print_result(self, result):
        """Pretty print prediction result"""
        print("\n" + "="*70)
        
        if result['prediction'] == 'spam':
            print("⚠️  PHISHING/SPAM DETECTED")
            color = '\033[91m'  # Red
        else:
            print("✓ LEGITIMATE MESSAGE")
            color = '\033[92m'  # Green
        
        print("="*70)
        
        print(f"\n📱 Message: {result['message'][:100]}...")
        print(f"\n{color}Prediction: {result['prediction'].upper()}\033[0m")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Threat Score: {result['threat_score']}/100")
        
        print(f"\n🔍 Key Indicators:")
        features = result['features']
        print(f"  • Urgency keywords: {features['urgency_keywords']}")
        print(f"  • Financial keywords: {features['financial_keywords']}")
        print(f"  • Action keywords: {features['action_keywords']}")
        print(f"  • Contains URL: {'Yes' if features['has_url'] else 'No'}")
        print(f"  • Contains phone: {'Yes' if features['has_phone'] else 'No'}")
        print(f"  • Excessive CAPS: {'Yes' if features['excessive_caps'] else 'No'}")
        
        print("="*70)


# Test the predictor
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SMS PHISHING DETECTOR - TESTING")
    print("="*70)
    
    # Initialize predictor
    try:
        predictor = SMSPredictor()
    except FileNotFoundError:
        print("\n✗ Please train the model first:")
        print("  python src/sms_detection/train_model.py")
        sys.exit(1)
    
    # Test messages
    test_messages = [
        "Hi! How are you? Want to grab lunch tomorrow?",
        "URGENT! Your account has been suspended. Click here to verify: bit.ly/xyz123",
        "Your Amazon order #12345 has been delivered. Thank you for shopping with us!",
        "ACT NOW! You've won $1000. Claim your prize by calling 555-1234 immediately!",
        "Meeting at 3pm in conference room B. Please confirm.",
        "ALERT: Unusual activity on your bank account. Verify now at secure-bank.tk",
        "Happy birthday! Hope you have a great day!",
        "FINAL NOTICE: Your payment is overdue. Click to avoid legal action."
    ]
    
    print("\n" + "="*70)
    print("TESTING WITH SAMPLE MESSAGES")
    print("="*70)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST {i}/{len(test_messages)}")
        result = predictor.predict(message)
        predictor.print_result(result)
    
    # Interactive mode
    print("\n\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter messages to check (or 'quit' to exit)")
    print("="*70)
    
    while True:
        message = input("\n📱 Enter message: ").strip()
        
        if message.lower() in ['quit', 'exit', 'q']:
            print("\n✓ Goodbye!")
            break
        
        if not message:
            continue
        
        result = predictor.predict(message)
        predictor.print_result(result)