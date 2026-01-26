"""
FAST SMS Prediction Module - Optimized for Real-time
NO DataFrames, NO progress bars, NO unnecessary overhead
Response time: ~100-200ms instead of 3-5 seconds
"""
import numpy as np
import joblib
import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (SMS_MODEL_PATH, FEATURE_EXTRACTOR_PATH,
                        URGENCY_KEYWORDS, FINANCIAL_KEYWORDS, 
                        ACTION_KEYWORDS, THREAT_KEYWORDS)

class FastSMSPredictor:
    """
    Ultra-fast SMS predictor optimized for single message predictions
    NO DataFrames, NO progress bars, NO unnecessary overhead
    """
    
    def __init__(self):
        """Load model and initialize components"""
        print("Loading Fast SMS Phishing Detector...")
        
        # Load model and feature extractor
        self.model = joblib.load(SMS_MODEL_PATH)
        self.feature_extractor = joblib.load(FEATURE_EXTRACTOR_PATH)
        
        # Initialize NLP components (one-time setup)
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            import nltk
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        
        # Compile regex patterns once
        self.url_pattern = re.compile(r'http\S+|www\.\S+|https\S+|\S+\.com|\S+\.org|\S+\.net')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.phone_pattern = re.compile(r'\d{10,}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}')
        self.number_pattern = re.compile(r'\d+')
        
        print("✓ Fast predictor ready!\n")
    
    def preprocess_single(self, text):
        """
        Fast preprocessing for single message
        Returns: cleaned text, processed text, features dict
        """
        if not text:
            text = ""
        
        text_lower = text.lower()
        
        # Extract features BEFORE cleaning (for pattern detection)
        features = self._extract_features_fast(text, text_lower)
        
        # Clean text
        cleaned = self.url_pattern.sub(' ', text_lower)
        cleaned = self.email_pattern.sub(' ', cleaned)
        cleaned = self.phone_pattern.sub(' ', cleaned)
        cleaned = self.number_pattern.sub(' ', cleaned)
        cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
        cleaned = ' '.join(cleaned.split())
        
        # Tokenize and process - use simple split instead of slow word_tokenize
        tokens = cleaned.split()
        tokens = [self.stemmer.stem(w) for w in tokens 
                 if w not in self.stop_words and len(w) > 2]
        
        processed_text = ' '.join(tokens)
        
        return cleaned, processed_text, features
    
    def _extract_features_fast(self, text, text_lower):
        """Extract features without DataFrame overhead"""
        features = {}
        
        # Basic stats
        features['message_length'] = len(text)
        words = text.split()
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        
        # Character counts
        features['special_char_count'] = sum(1 for c in text if c in string.punctuation)
        features['digit_count'] = sum(1 for c in text if c.isdigit())
        features['uppercase_count'] = sum(1 for c in text if c.isupper())
        
        # Ratios
        text_len = len(text) if len(text) > 0 else 1
        features['uppercase_ratio'] = features['uppercase_count'] / text_len
        features['digit_ratio'] = features['digit_count'] / text_len
        features['special_char_ratio'] = features['special_char_count'] / text_len
        
        # Pattern detection
        features['has_url'] = 1 if self.url_pattern.search(text_lower) else 0
        features['has_email'] = 1 if self.email_pattern.search(text_lower) else 0
        features['has_phone'] = 1 if self.phone_pattern.search(text) else 0
        features['has_currency'] = 1 if any(s in text for s in ['$', '£', '€', 'dollar', 'pound']) else 0
        
        # Keyword counts
        features['urgency_count'] = sum(1 for k in URGENCY_KEYWORDS if k in text_lower)
        features['financial_count'] = sum(1 for k in FINANCIAL_KEYWORDS if k in text_lower)
        features['action_count'] = sum(1 for k in ACTION_KEYWORDS if k in text_lower)
        features['threat_count'] = sum(1 for k in THREAT_KEYWORDS if k in text_lower)
        
        # Excessive patterns
        features['excessive_caps'] = 1 if features['uppercase_ratio'] > 0.3 else 0
        features['excessive_punctuation'] = 1 if features['special_char_ratio'] > 0.15 else 0
        
        return features
    
    def predict(self, message):
        """
        Ultra-fast prediction for single message
        Returns: prediction dict
        """
        if not message or not message.strip():
            return {
                'message': message,
                'prediction': 'unknown',
                'label': -1,
                'confidence': 0.0,
                'threat_score': 0,
                'threat_level': 'safe',
                'is_phishing': False,
                'features': {},
                'error': 'Empty message'
            }
        
        # Fast preprocessing
        cleaned, processed, stat_features = self.preprocess_single(message)
        
        # TF-IDF vectorization
        tfidf_features = self.feature_extractor.tfidf.transform([processed]).toarray()[0]
        
        # Get the numerical features in the same order as the feature extractor used during training
        numerical_feature_names = self.feature_extractor.numerical_features
        numerical_values = [stat_features.get(fname, 0) for fname in numerical_feature_names]
        
        # Scale numerical features
        numerical_scaled = self.feature_extractor.scaler.transform([numerical_values])[0]
        
        # Combine features
        all_features = np.concatenate([tfidf_features, numerical_scaled]).reshape(1, -1)
        all_features = np.abs(all_features)  # Handle negative values
        
        # Predict
        prediction = int(self.model.predict(all_features)[0])
        
        # Get probability
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(all_features)[0]
            confidence = float(probabilities[prediction])
        else:
            confidence = 1.0
        
        # Calculate threat score
        if prediction == 1:  # Spam/Phishing
            threat_score = int(confidence * 100)
        else:  # Ham/Safe
            threat_score = int((1 - confidence) * 100)
        
        # Determine threat level
        if threat_score < 30:
            threat_level = 'safe'
        elif threat_score < 60:
            threat_level = 'suspicious'
        elif threat_score < 85:
            threat_level = 'dangerous'
        else:
            threat_level = 'critical'
        
        return {
            'message': message,
            'prediction': 'spam' if prediction == 1 else 'ham',
            'label': prediction,
            'confidence': confidence,
            'threat_score': threat_score,
            'threat_level': threat_level,
            'is_phishing': bool(prediction == 1),
            'features': {
                'urgency_keywords': stat_features['urgency_count'],
                'financial_keywords': stat_features['financial_count'],
                'action_keywords': stat_features['action_count'],
                'threat_keywords': stat_features['threat_count'],
                'has_url': bool(stat_features['has_url']),
                'has_phone': bool(stat_features['has_phone']),
                'message_length': stat_features['message_length'],
                'uppercase_ratio': stat_features['uppercase_ratio'],
                'excessive_caps': bool(stat_features['excessive_caps'])
            }
        }


# Quick test
if __name__ == "__main__":
    import time
    
    predictor = FastSMSPredictor()
    
    test_messages = [
        "Hi how are you?",
        "URGENT! Your account has been suspended. Click here to verify: bit.ly/xyz123",
        "Meeting at 3pm tomorrow in conference room B",
        "Congratulations! You've won $10,000! Call NOW at 1-800-555-1234 to claim!"
    ]
    
    print("\n" + "="*70)
    print("SPEED TEST - FAST PREDICTOR")
    print("="*70)
    
    for msg in test_messages:
        start = time.time()
        result = predictor.predict(msg)
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        print(f"\nMessage: {msg[:50]}...")
        print(f"Result: {result['prediction'].upper()} (Score: {result['threat_score']})")
        print(f"⚡ Time: {elapsed:.2f}ms")
    
    print("\n" + "="*70)
    print("✓ Fast predictor test complete!")
    print("="*70)
