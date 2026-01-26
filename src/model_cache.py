"""
Model Cache Module - Singleton Pattern for Fast Model Access
Loads models ONCE when imported and keeps them in memory.
This ensures the server is ready immediately without background loading delays.
"""
import os
import sys
import time
import re
import string
import numpy as np
from pathlib import Path

# CRITICAL: Set up paths BEFORE any project imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import joblib
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from src.config import (SMS_MODEL_PATH, FEATURE_EXTRACTOR_PATH,
                        URGENCY_KEYWORDS, FINANCIAL_KEYWORDS,
                        ACTION_KEYWORDS, THREAT_KEYWORDS)


class ModelCache:
    """
    Singleton model cache that loads models once and keeps them in memory.
    Access via: from src.model_cache import model_cache
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if ModelCache._initialized:
            return
        
        print("\n" + "="*50)
        print("🔄 Loading PhishGuard AI Models...")
        print("="*50)
        
        start_time = time.time()
        
        try:
            # Load the trained model
            self.model = joblib.load(SMS_MODEL_PATH)
            print(f"✓ Model loaded from: {SMS_MODEL_PATH}")
            
            # Load the feature extractor
            self.feature_extractor = joblib.load(FEATURE_EXTRACTOR_PATH)
            print(f"✓ Feature extractor loaded")
            
            # Initialize NLP components
            self.stemmer = PorterStemmer()
            try:
                self.stop_words = set(stopwords.words('english'))
            except LookupError:
                import nltk
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(stopwords.words('english'))
            print(f"✓ NLP components initialized")
            
            # Pre-compile regex patterns
            self.url_pattern = re.compile(r'http\S+|www\.\S+|https\S+|\S+\.com|\S+\.org|\S+\.net')
            self.email_pattern = re.compile(r'\S+@\S+')
            self.phone_pattern = re.compile(r'\d{10,}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}')
            self.number_pattern = re.compile(r'\d+')
            
            # Store keywords for fast access
            self.urgency_keywords = URGENCY_KEYWORDS
            self.financial_keywords = FINANCIAL_KEYWORDS
            self.action_keywords = ACTION_KEYWORDS
            self.threat_keywords = THREAT_KEYWORDS
            
            load_time = time.time() - start_time
            print("="*50)
            print(f"✅ All models loaded in {load_time:.2f} seconds!")
            print("="*50 + "\n")
            
            ModelCache._initialized = True
            self.is_ready = True
            
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            import traceback
            traceback.print_exc()
            self.is_ready = False
            raise
    
    def extract_features_fast(self, text):
        """Fast feature extraction without DataFrame overhead"""
        text_lower = text.lower()
        words = text.split()
        text_len = len(text) if len(text) > 0 else 1
        
        features = {
            'message_length': len(text),
            'word_count': len(words),
            'char_count': len(text),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'special_char_count': sum(1 for c in text if c in string.punctuation),
            'digit_count': sum(1 for c in text if c.isdigit()),
            'uppercase_count': sum(1 for c in text if c.isupper()),
        }
        
        # Ratios
        features['uppercase_ratio'] = features['uppercase_count'] / text_len
        features['digit_ratio'] = features['digit_count'] / text_len
        features['special_char_ratio'] = features['special_char_count'] / text_len
        
        # Pattern detection
        features['has_url'] = 1 if self.url_pattern.search(text_lower) else 0
        features['has_email'] = 1 if self.email_pattern.search(text_lower) else 0
        features['has_phone'] = 1 if self.phone_pattern.search(text) else 0
        features['has_currency'] = 1 if any(s in text for s in ['$', '£', '€', 'dollar', 'pound']) else 0
        
        # Keyword counts
        features['urgency_count'] = sum(1 for k in self.urgency_keywords if k in text_lower)
        features['financial_count'] = sum(1 for k in self.financial_keywords if k in text_lower)
        features['action_count'] = sum(1 for k in self.action_keywords if k in text_lower)
        features['threat_count'] = sum(1 for k in self.threat_keywords if k in text_lower)
        
        # Excessive patterns
        features['excessive_caps'] = 1 if features['uppercase_ratio'] > 0.3 else 0
        features['excessive_punctuation'] = 1 if features['special_char_ratio'] > 0.15 else 0
        
        return features
    
    def preprocess_text_fast(self, text):
        """Fast text preprocessing"""
        text_lower = text.lower()
        
        # Clean text using pre-compiled patterns
        cleaned = self.url_pattern.sub(' ', text_lower)
        cleaned = self.email_pattern.sub(' ', cleaned)
        cleaned = self.phone_pattern.sub(' ', cleaned)
        cleaned = self.number_pattern.sub(' ', cleaned)
        cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
        cleaned = ' '.join(cleaned.split())
        
        # Fast tokenization
        tokens = cleaned.split()
        tokens = [self.stemmer.stem(w) for w in tokens if w not in self.stop_words and len(w) > 2]
        
        return ' '.join(tokens)
    
    def predict(self, message):
        """Ultra-fast prediction using cached models"""
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
        
        # Extract features
        stat_features = self.extract_features_fast(message)
        
        # Preprocess text
        processed_text = self.preprocess_text_fast(message)
        
        # TF-IDF vectorization
        tfidf_features = self.feature_extractor.tfidf.transform([processed_text]).toarray()[0]
        
        # Get numerical features in same order as training
        numerical_feature_names = self.feature_extractor.numerical_features
        numerical_values = [stat_features.get(fname, 0) for fname in numerical_feature_names]
        
        # Scale numerical features
        numerical_scaled = self.feature_extractor.scaler.transform([numerical_values])[0]
        
        # Combine features
        all_features = np.concatenate([tfidf_features, numerical_scaled]).reshape(1, -1)
        all_features = np.abs(all_features)
        
        # Predict
        prediction = int(self.model.predict(all_features)[0])
        
        # Get probability
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(all_features)[0]
            confidence = float(probabilities[prediction])
        else:
            confidence = 1.0
        
        # Calculate threat score
        threat_score = int(confidence * 100) if prediction == 1 else int((1 - confidence) * 100)
        
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


# Create singleton instance - models load when this module is first imported
# This is the key to the permanent fix: import triggers synchronous loading
model_cache = ModelCache()
