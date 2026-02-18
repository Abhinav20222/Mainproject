"""
Flask REST API for PhishGuard AI - PERMANENTLY OPTIMIZED VERSION
Uses singleton model cache for instant startup (~100-200ms response time)
Includes SMS analysis, URL analysis, visual spoofing, and full-scan endpoints.
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sys
import os
import re
from pathlib import Path
from urllib.parse import urlparse
import time

# CRITICAL: Add project root to path BEFORE any project imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Import model cache - this triggers SYNCHRONOUS model loading
# The server will NOT start until models are fully loaded
from src.model_cache import model_cache

# Lazy-loaded singletons for URL and visual detection
_url_predictor = None
_image_comparator = None
_screenshot_capturer = None

TEMP_DIR = PROJECT_ROOT / "data" / "temp"
os.makedirs(str(TEMP_DIR), exist_ok=True)


def get_url_predictor():
    """Lazy-load URL predictor singleton."""
    global _url_predictor
    if _url_predictor is None:
        try:
            from src.url_detection.url_predictor import URLPredictor
            _url_predictor = URLPredictor()
        except Exception as e:
            print(f"[WARN] URL predictor not available: {e}")
    return _url_predictor


def get_image_comparator():
    """Lazy-load image comparator singleton."""
    global _image_comparator
    if _image_comparator is None:
        try:
            from src.visual_detection.image_comparator import ImageComparator
            _image_comparator = ImageComparator()
        except Exception as e:
            print(f"[WARN] Image comparator not available: {e}")
    return _image_comparator


def get_screenshot_capturer():
    """Lazy-load screenshot capturer singleton."""
    global _screenshot_capturer
    if _screenshot_capturer is None:
        try:
            from src.visual_detection.screenshot_capturer import ScreenshotCapturer
            _screenshot_capturer = ScreenshotCapturer()
        except Exception as e:
            print(f"[WARN] Screenshot capturer not available: {e}")
    return _screenshot_capturer


app = Flask(__name__)
CORS(app)  # Allow all origins for easier development


# ============================================================
# EXISTING ENDPOINTS (unchanged)
# ============================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    is_ready = model_cache.is_ready
    url_ready = get_url_predictor() is not None and get_url_predictor().is_ready
    return jsonify({
        'status': 'online' if is_ready else 'error',
        'service': 'PhishGuard AI',
        'model': 'SMS Phishing Detector v2.1 (CACHED)',
        'models_loaded': is_ready,
        'url_model_loaded': url_ready,
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_message():
    """Analyze a message for phishing threats - FAST VERSION"""
    start_time = time.time()
    
    try:
        # Check if models are ready (always true after startup now)
        if not model_cache.is_ready:
            return jsonify({
                'error': 'Models not loaded. Server startup failed.',
            }), 503
        
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Missing message field',
                'usage': 'POST {"message": "your text here"}'
            }), 400
        
        message = data['message'].strip()
        
        if not message:
            return jsonify({
                'error': 'Message cannot be empty'
            }), 400
        
        # Use cached model for prediction
        result = model_cache.predict(message)
        
        # Add processing time
        processing_time = (time.time() - start_time) * 1000
        result['processing_time_ms'] = round(processing_time, 2)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error analyzing message: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Analysis failed',
            'details': str(e)
        }), 500


# ============================================================
# NEW ENDPOINTS
# ============================================================

@app.route('/api/analyze-url', methods=['POST'])
def analyze_url():
    """Analyze a URL for phishing using ML-based URL classifier."""
    start_time = time.time()

    try:
        data = request.get_json()

        if not data or 'url' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing url field',
                'usage': 'POST {"url": "http://example.com"}'
            }), 400

        url = data['url'].strip()
        if not url:
            return jsonify({'success': False, 'error': 'URL cannot be empty'}), 400

        # Validate URL is parseable
        try:
            parsed = urlparse(url if '://' in url else 'http://' + url)
            if not parsed.hostname:
                raise ValueError("No hostname")
        except Exception:
            return jsonify({
                'success': False,
                'error': 'Malformed URL — cannot parse hostname'
            }), 422

        predictor = get_url_predictor()
        if predictor is None or not predictor.is_ready:
            return jsonify({
                'success': False,
                'error': 'URL model not loaded. Train the URL model first.'
            }), 503

        result = predictor.predict(url)

        analysis_time = (time.time() - start_time) * 1000
        return jsonify({
            'success': True,
            'url': result['url'],
            'is_phishing': result['is_phishing'],
            'threat_score': result['threat_score'],
            'risk_level': result['risk_level'],
            'top_risk_features': result['top_risk_features'],
            'features': result['features'],
            'analysis_time_ms': round(analysis_time, 2),
        })

    except Exception as e:
        print(f"Error analyzing URL: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/visual-check', methods=['POST'])
def visual_check():
    """Capture screenshot and compare against trusted site database."""
    start_time = time.time()

    try:
        data = request.get_json()

        if not data or 'url' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing url field'
            }), 400

        url = data['url'].strip()
        if not url:
            return jsonify({'success': False, 'error': 'URL cannot be empty'}), 400

        # Capture screenshot
        capturer = get_screenshot_capturer()
        if capturer is None:
            return jsonify({
                'success': False,
                'error': 'Screenshot capturer not available. Install selenium and webdriver-manager.'
            }), 503

        try:
            screenshot_path = capturer.capture(url)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Screenshot capture failed: {str(e)}',
                'url': url
            }), 503

        # Compare against trusted database
        comparator = get_image_comparator()
        if comparator is None:
            return jsonify({
                'success': False,
                'error': 'Image comparator not available.'
            }), 503

        result = comparator.compare(screenshot_path)

        analysis_time = (time.time() - start_time) * 1000
        return jsonify({
            'success': True,
            'url': url,
            'spoofing_detected': result['spoofing_detected'],
            'best_match_site': result['best_match_site'],
            'best_match_url': result['best_match_url'],
            'ssim_score': result.get('ssim_score', 0.0),
            'visual_threat_score': result.get('visual_threat_score', 0.0),
            'phash_distance': result.get('phash_distance', 999),
            'heatmap_available': result.get('heatmap_path') is not None,
            'analysis_method': result.get('analysis_method', 'unknown'),
            'analysis_time_ms': round(analysis_time, 2),
        })

    except Exception as e:
        print(f"Error in visual check: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/full-scan', methods=['POST'])
def full_scan():
    """
    Combined multi-channel analysis.
    Accepts a single message text, auto-extracts URLs from it,
    runs SMS analysis on the full text and URL analysis on extracted URLs.
    Weights: SMS=0.40, URL=0.45, Visual=0.15
    """
    start_time = time.time()

    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'Empty request body'}), 400

        message = data.get('message', '').strip()
        # Backward compat: allow explicit url field, but prefer auto-extraction
        explicit_url = data.get('url', '').strip()
        include_visual = data.get('include_visual', False)

        if not message and not explicit_url:
            return jsonify({
                'success': False,
                'error': 'Message field is required'
            }), 400

        # Auto-extract URLs from the message text
        url_pattern = re.compile(
            r'(?:https?://|www\.)[^\s<>\"\']+|'          # http(s):// or www.
            r'[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s<>\"\']*)?',  # domain.tld/path
            re.IGNORECASE
        )
        
        # This regex finds ALL URLs inside the message text
        # Example: "Click http://evil.com to verify" → extracts ["http://evil.com"]

        extracted_urls = url_pattern.findall(message) if message else []
        # Normalize: prepend http:// if missing scheme
        extracted_urls = [
            u if u.startswith(('http://', 'https://')) else 'http://' + u
            for u in extracted_urls
        ]
        # If explicit URL was provided, add it too
        if explicit_url and explicit_url not in extracted_urls:
            extracted_urls.append(explicit_url)

        analyses_performed = []
        sms_analysis = None
        url_analysis = None
        visual_analysis = None

        # Base weights
        weights = {'sms': 0.40, 'url': 0.45, 'visual': 0.15}
        scores = {}

        # --- SMS Analysis (on full message text) ---
        if message and model_cache.is_ready:
            try:
                sms_result = model_cache.predict(message)
                sms_analysis = sms_result
                # Normalize threat_score to 0-1 range
                sms_score = sms_result.get('threat_score', 0)
                if sms_score > 1:
                    sms_score = sms_score / 100.0
                scores['sms'] = sms_score
                analyses_performed.append('sms')
            except Exception as e:
                sms_analysis = {'error': str(e)}

        # --- URL Analysis (on each extracted URL, keep worst score) ---
        if extracted_urls:
            predictor = get_url_predictor()
            if predictor and predictor.is_ready:
                try:
                    best_url_result = None
                    best_url_score = -1
                    for u in extracted_urls:
                        try:
                            url_result = predictor.predict(u)
                            score = url_result.get('threat_score', 0)
                            if score > best_url_score:
                                best_url_score = score
                                best_url_result = url_result
                        except Exception:
                            continue
                    if best_url_result:
                        url_analysis = best_url_result
                        url_analysis['urls_checked'] = extracted_urls
                        scores['url'] = best_url_score
                        analyses_performed.append('url')
                except Exception as e:
                    url_analysis = {'error': str(e)}

        # --- Visual Analysis (on first extracted URL) ---
        if extracted_urls and include_visual:
            capturer = get_screenshot_capturer()
            comparator = get_image_comparator()
            if capturer and comparator:
                try:
                    screenshot_path = capturer.capture(extracted_urls[0])
                    vis_result = comparator.compare(screenshot_path)
                    visual_analysis = vis_result
                    scores['visual'] = vis_result.get('visual_threat_score', 0)
                    analyses_performed.append('visual')
                except Exception as e:
                    visual_analysis = {'error': str(e)}

        # --- Compute Combined Score ---
        if scores:
            # Normalize weights to only include performed analyses
            active_weights = {k: weights[k] for k in scores}
            total_weight = sum(active_weights.values())
            combined_score = sum(
                scores[k] * (active_weights[k] / total_weight)
                for k in scores
            )
            # If only SMS was performed (no URLs in message):
            #   combined = sms_score × (0.40 / 0.40) = sms_score × 1.0
            #
            # If SMS + URL performed:
            #   combined = sms_score × (0.40/0.85) + url_score × (0.45/0.85)
            #   combined = sms_score × 0.47 + url_score × 0.53
            #   (weights are RE-NORMALIZED to sum to 1.0)
            #
            # If SMS + URL + Visual performed:
            #   combined = sms × 0.40 + url × 0.45 + visual × 0.15
        else:
            combined_score = 0.0

        # Determine risk level
        if combined_score < 0.3:
            risk_level = "LOW"
        elif combined_score < 0.6:
            risk_level = "MEDIUM"
        elif combined_score < 0.85:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        total_time = (time.time() - start_time) * 1000

        return jsonify({
            'success': True,
            'combined_threat_score': round(combined_score, 4),
            'risk_level': risk_level,
            'sms_analysis': sms_analysis,
            'url_analysis': url_analysis,
            'visual_analysis': visual_analysis,
            'analyses_performed': analyses_performed,
            'score_weights': {k: round(weights[k] / sum(weights[k2] for k2 in scores), 2)
                              for k in scores} if scores else {},
            'total_analysis_time_ms': round(total_time, 2),
        })

    except Exception as e:
        print(f"Error in full scan: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/heatmap', methods=['GET'])
def serve_heatmap():
    """Serve the latest visual difference heatmap image."""
    heatmap_path = TEMP_DIR / "diff_heatmap.png"
    if heatmap_path.exists():
        return send_file(str(heatmap_path), mimetype='image/png')
    else:
        return jsonify({'error': 'No heatmap available'}), 404


if __name__ == '__main__':
    print("\n" + "="*50)
    print("PhishGuard AI API - FULL STACK VERSION")
    print("="*50)
    print("✅ SMS models loaded via cache!")
    print("🔗 URL model: loading on first request")
    print("👁️  Visual detection: loading on first request")
    print("🚀 SMS Predictions: ~100-200ms")
    print("="*50)
    print("Endpoints:")
    print("  GET  /api/health       - Health check")
    print("  POST /api/analyze      - SMS message analysis")
    print("  POST /api/analyze-url  - URL phishing analysis")
    print("  POST /api/visual-check - Visual spoofing check")
    print("  POST /api/full-scan    - Multi-channel full scan")
    print("  GET  /api/heatmap      - View diff heatmap")
    print("="*50 + "\n")
    
    # Start Flask immediately - models are already loaded!
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
