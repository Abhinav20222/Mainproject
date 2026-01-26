"""
Flask REST API for PhishGuard AI - ULTRA FAST VERSION
Lazy-loads NLTK and models for faster startup (~5 seconds to online)
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path
import threading
import time
import re
import string
import numpy as np

# CRITICAL: Add project root to path BEFORE any project imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

app = Flask(__name__)
CORS(app)

# Global state - lazy loaded
model = None
feature_extractor = None
stemmer = None
stop_words = None
models_ready = False
loading_start_time = None

# Pre-compiled regex patterns
url_pattern = re.compile(r'http\S+|www\.\S+|https\S+|\S+\.com|\S+\.org|\S+\.net')
email_pattern = re.compile(r'\S+@\S+')
phone_pattern = re.compile(r'\d{10,}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}')
number_pattern = re.compile(r'\d+')

# Keyword lists (inline to avoid importing config during startup)
URGENCY_KEYWORDS = ['urgent', 'immediately', 'now', 'asap', 'hurry', 'limited', 
    'expire', 'today', 'fast', 'quick', 'act now', 'limited time']
FINANCIAL_KEYWORDS = ['bank', 'account', 'credit', 'debit', 'card', 'money', 'cash', 
    'payment', 'transaction', 'dollar', 'prize', 'won', 'reward', 'refund', 'tax', 'irs', 'paypal']
ACTION_KEYWORDS = ['click', 'call', 'reply', 'confirm', 'verify', 'update', 
    'claim', 'redeem', 'activate', 'download', 'install']
THREAT_KEYWORDS = ['suspend', 'block', 'locked', 'unauthorized', 'unusual activity',
    'security alert', 'compromised', 'fraud']


def load_models_background():
    """Load ML models in background - optimized for speed"""
    global model, feature_extractor, stemmer, stop_words, models_ready, loading_start_time
    
    loading_start_time = time.time()
    print("\n[LOADING] Starting model initialization...")
    
    try:
        # Step 1: Load NLTK components (this is often the slowest part)
        print("[1/4] Loading NLTK stopwords...")
        t1 = time.time()
        from nltk.stem import PorterStemmer
        from nltk.corpus import stopwords
        import nltk
        
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        print(f"    Done in {time.time()-t1:.2f}s")
        
        # Step 2: Import classes needed for pickle deserialization
        print("[2/4] Loading dependencies for model deserialization...")
        t2 = time.time()
        import joblib
        # CRITICAL: These imports are required for pickle to work
        from src.sms_detection.feature_extraction import FeatureExtractor
        from src.sms_detection.preprocessing import SMSPreprocessor
        print(f"    Done in {time.time()-t2:.2f}s")
        
        # Step 3: Load the classifier model
        print("[3/4] Loading SMS classifier...")
        t3 = time.time()
        model_path = PROJECT_ROOT / "data" / "models" / "sms_classifier.pkl"
        model = joblib.load(model_path)
        print(f"    Done in {time.time()-t3:.2f}s")
        
        # Step 4: Load feature extractor
        print("[4/4] Loading feature extractor...")
        t4 = time.time()
        extractor_path = PROJECT_ROOT / "data" / "models" / "feature_extractor.pkl"
        feature_extractor = joblib.load(extractor_path)
        print(f"    Done in {time.time()-t4:.2f}s")
        
        total_time = time.time() - loading_start_time
        print(f"\n✅ ALL MODELS LOADED in {total_time:.2f} seconds!")
        models_ready = True
        return True
        
    except Exception as e:
        print(f"\n❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_features_fast(text):
    """Fast feature extraction without DataFrame"""
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
    
    features['uppercase_ratio'] = features['uppercase_count'] / text_len
    features['digit_ratio'] = features['digit_count'] / text_len
    features['special_char_ratio'] = features['special_char_count'] / text_len
    
    features['has_url'] = 1 if url_pattern.search(text_lower) else 0
    features['has_email'] = 1 if email_pattern.search(text_lower) else 0
    features['has_phone'] = 1 if phone_pattern.search(text) else 0
    features['has_currency'] = 1 if any(s in text for s in ['$', '£', '€', 'dollar', 'pound']) else 0
    
    features['urgency_count'] = sum(1 for k in URGENCY_KEYWORDS if k in text_lower)
    features['financial_count'] = sum(1 for k in FINANCIAL_KEYWORDS if k in text_lower)
    features['action_count'] = sum(1 for k in ACTION_KEYWORDS if k in text_lower)
    features['threat_count'] = sum(1 for k in THREAT_KEYWORDS if k in text_lower)
    
    features['excessive_caps'] = 1 if features['uppercase_ratio'] > 0.3 else 0
    features['excessive_punctuation'] = 1 if features['special_char_ratio'] > 0.15 else 0
    
    return features


def preprocess_text_fast(text):
    """Fast text preprocessing"""
    text_lower = text.lower()
    
    cleaned = url_pattern.sub(' ', text_lower)
    cleaned = email_pattern.sub(' ', cleaned)
    cleaned = phone_pattern.sub(' ', cleaned)
    cleaned = number_pattern.sub(' ', cleaned)
    cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
    cleaned = ' '.join(cleaned.split())
    
    tokens = cleaned.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    
    return ' '.join(tokens)


def predict_fast(message):
    """Ultra-fast prediction"""
    stat_features = extract_features_fast(message)
    processed_text = preprocess_text_fast(message)
    
    tfidf_features = feature_extractor.tfidf.transform([processed_text]).toarray()[0]
    
    numerical_feature_names = feature_extractor.numerical_features
    numerical_values = [stat_features.get(fname, 0) for fname in numerical_feature_names]
    numerical_scaled = feature_extractor.scaler.transform([numerical_values])[0]
    
    all_features = np.concatenate([tfidf_features, numerical_scaled]).reshape(1, -1)
    all_features = np.abs(all_features)
    
    prediction = int(model.predict(all_features)[0])
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(all_features)[0]
        confidence = float(probabilities[prediction])
    else:
        confidence = 1.0
    
    threat_score = int(confidence * 100) if prediction == 1 else int((1 - confidence) * 100)
    
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


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check - returns online when models are ready"""
    return jsonify({
        'status': 'online' if models_ready else 'loading',
        'service': 'PhishGuard AI',
        'model': 'SMS Phishing Detector v2.0 (FAST)',
        'models_loaded': models_ready
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_message():
    """Analyze a message for phishing"""
    start_time = time.time()
    
    try:
        if not models_ready:
            return jsonify({'error': 'Models still loading. Please wait.'}), 503
        
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Missing message field',
                'usage': 'POST {"message": "your text here"}'
            }), 400
        
        message = data['message'].strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        result = predict_fast(message)
        result['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("⚡ PhishGuard AI - ULTRA FAST API")
    print("="*50)
    
    # Start background loading
    loader = threading.Thread(target=load_models_background, daemon=True)
    loader.start()
    
    print("\n🚀 Server starting on http://localhost:5000")
    print("📦 Models loading in background...")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
