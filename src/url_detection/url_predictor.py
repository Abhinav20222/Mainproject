"""
URL Prediction Module
Singleton-pattern predictor for real-time URL phishing detection.
Loads the trained model once and reuses it for all predictions.
"""
import numpy as np
import joblib
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.url_detection.url_feature_extractor import URLFeatureExtractor

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
URL_MODEL_PATH = MODELS_DIR / "url_classifier.pkl"
URL_FEATURE_NAMES_PATH = MODELS_DIR / "url_feature_names.pkl"


class URLPredictor:
    """
    Singleton URL Phishing Predictor.
    Loads the trained model once on first instantiation and reuses it.
    
    Usage:
        predictor = URLPredictor()
        result = predictor.predict("http://suspicious-site.com/login")
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if URLPredictor._initialized:
            return

        print("[URL Predictor] Loading model...")
        self.is_ready = False

        try:
            self.model = joblib.load(URL_MODEL_PATH)
            self.feature_names = joblib.load(URL_FEATURE_NAMES_PATH)
            self.extractor = URLFeatureExtractor()
            self.is_ready = True
            print("[URL Predictor] Model loaded successfully!")

            # Get feature importances if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = dict(
                    zip(self.feature_names, self.model.feature_importances_)
                )
            elif hasattr(self.model, 'coef_'):
                self.feature_importances = dict(
                    zip(self.feature_names, np.abs(self.model.coef_[0]))
                )
            else:
                self.feature_importances = {}

        except FileNotFoundError as e:
            print(f"[URL Predictor] ERROR: Model files not found: {e}")
            print("  Run: python -m src.url_detection.train_url_model")
            self.model = None
            self.feature_names = []
            self.extractor = URLFeatureExtractor()
            self.feature_importances = {}

        URLPredictor._initialized = True

    def _get_risk_level(self, score):
        """Determine risk level from threat score."""
        if score < 0.3:
            return "LOW"
        elif score < 0.6:
            return "MEDIUM"
        elif score < 0.85:
            return "HIGH"
        else:
            return "CRITICAL"

    def _get_top_risk_features(self, features, top_n=5):
        """
        Get top N risk features sorted by importance × value.
        Only considers features that have nonzero values.
        """
        if not self.feature_importances:
            # Fall back to features with highest values
            sorted_features = sorted(
                features.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            return [name for name, val in sorted_features[:top_n] if val != 0]

        scored = {}
        for name, value in features.items():
            importance = self.feature_importances.get(name, 0)
            scored[name] = importance * abs(value)

        sorted_features = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return [name for name, score in sorted_features[:top_n] if score > 0]

    def predict(self, url):
        """
        Predict whether a URL is phishing.
        
        Args:
            url (str): Raw URL string
            
        Returns:
            dict: Prediction results including threat_score, risk_level, features
        """
        if not self.is_ready:
            return {
                "url": url,
                "error": "Model not loaded. Train the model first.",
                "is_phishing": False,
                "threat_score": 0.0,
                "risk_level": "UNKNOWN",
                "top_risk_features": [],
                "features": {},
            }

        # Extract features
        features = self.extractor.extract(url)

        # Build feature vector in correct order
        feature_vector = np.array(
            [features.get(name, 0) for name in self.feature_names]
        ).reshape(1, -1)

        # Handle NaN / Inf
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict
        prediction = self.model.predict(feature_vector)[0]

        # Get probability
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(feature_vector)[0]
            threat_score = float(probabilities[1])  # probability of phishing class
        elif hasattr(self.model, 'decision_function'):
            decision = self.model.decision_function(feature_vector)[0]
            # Sigmoid to convert to probability
            threat_score = float(1 / (1 + np.exp(-decision)))
        else:
            threat_score = float(prediction)

        is_phishing = bool(prediction == 1)
        risk_level = self._get_risk_level(threat_score)
        top_risk_features = self._get_top_risk_features(features)

        return {
            "url": url,
            "is_phishing": is_phishing,
            "threat_score": round(threat_score, 4),
            "risk_level": risk_level,
            "top_risk_features": top_risk_features,
            "features": features,
        }

    def predict_batch(self, urls):
        """
        Predict multiple URLs.
        
        Args:
            urls (list): List of URL strings
            
        Returns:
            list: List of prediction result dicts
        """
        results = []
        for url in urls:
            try:
                result = self.predict(url)
            except Exception as e:
                result = {
                    "url": url,
                    "error": str(e),
                    "is_phishing": False,
                    "threat_score": 0.0,
                    "risk_level": "UNKNOWN",
                    "top_risk_features": [],
                    "features": {},
                }
            results.append(result)
        return results


# Quick test
if __name__ == "__main__":
    import time

    predictor = URLPredictor()

    test_urls = [
        "https://www.google.com",
        "https://www.sbi.co.in",
        "http://192.168.1.1/sbi/login?user=admin",
        "http://paypal.secure-login.xyz/verify/account",
        "http://www.google.com@192.168.1.1/login",
    ]

    print("\n" + "=" * 70)
    print("URL PREDICTOR — QUICK TEST")
    print("=" * 70)

    for url in test_urls:
        start = time.time()
        result = predictor.predict(url)
        elapsed = (time.time() - start) * 1000

        label = "PHISHING" if result['is_phishing'] else "SAFE"
        print(f"\n  URL: {url}")
        print(f"  Result: {label} | Score: {result['threat_score']:.4f} | Risk: {result['risk_level']}")
        print(f"  Top risks: {result['top_risk_features'][:3]}")
        print(f"  Time: {elapsed:.1f}ms")

    print("\n" + "=" * 70)
    print("[OK] URL Predictor working!")
    print("=" * 70)
