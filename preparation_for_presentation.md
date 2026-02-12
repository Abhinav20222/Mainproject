# PhishGuard AI — Comprehensive Presentation Preparation

---

## 1. PROJECT OVERVIEW

**PhishGuard AI** is a **multi-channel phishing detection system** that uses Machine Learning to detect phishing attempts through three independent analysis channels:

| Channel | What it does | Algorithm |
|---|---|---|
| **SMS/Text Analysis** | Analyzes message content for phishing patterns | TF-IDF + Random Forest / Naive Bayes / Logistic Regression / SVM |
| **URL Analysis** | Analyzes URL structure for phishing indicators | 28 lexical features + Random Forest / Gradient Boosting / XGBoost |
| **Visual Cloning** *(remaining 30%)* | Compares website screenshots against trusted sites | Selenium + pHash + SSIM (Structural Similarity Index) |

**Project Completion:** ~70% (Visual cloning via Selenium remaining)

**Tech Stack:**
- **Backend:** Python, Flask, scikit-learn, NLTK, joblib, NumPy, Pandas
- **Frontend:** React (Vite), Axios, Recharts, Lucide Icons, TailwindCSS
- **ML Libraries:** scikit-learn, xgboost, NLTK, TfidfVectorizer, StandardScaler
- **Visual Detection (planned):** Selenium, OpenCV, scikit-image, imagehash, Pillow

---

## 2. COMPREHENSIVE ALGORITHM EXPLANATION

### 2.1 SMS/Text Phishing Detection Algorithm

The SMS detection uses a **supervised classification** approach:

**Step 1: Text Preprocessing** (`src/sms_detection/preprocessing.py`)
```
Raw Text → Lowercase → Remove URLs/Emails/Phone/Numbers → Remove Punctuation → Tokenize → Remove Stopwords → Stemming (PorterStemmer) → Clean Text
```

- **PorterStemmer:** Reduces words to root form (e.g., "running" → "run", "verification" → "verifi")
- **Stopword Removal:** Removes common words like "the", "is", "at" that don't help classification
- **Tokenization:** Splits text into individual words

**Step 2: Feature Extraction** (`src/sms_detection/feature_extraction.py`)

Two types of features are extracted:

**A) TF-IDF Features (500 features):**
- TF-IDF = Term Frequency × Inverse Document Frequency
- Measures how important a word is to a document relative to the entire dataset
- **Config:** max_features=500, ngram_range=(1,2), min_df=2, max_df=0.8
  - `ngram_range=(1,2)` means it considers both single words AND pairs of words
  - `min_df=2` means a word must appear in at least 2 documents
  - `max_df=0.8` means ignore words appearing in more than 80% of documents

**B) Statistical/Numerical Features (19 features):**
- `message_length`, `word_count`, `avg_word_length`
- `special_char_count`, `digit_count`, `uppercase_count`
- `uppercase_ratio`, `digit_ratio`, `special_char_ratio`
- `has_url`, `has_email`, `has_phone`, `has_currency`
- `urgency_count` (words like "urgent", "immediately", "now")
- `financial_count` (words like "bank", "account", "credit")
- `action_count` (words like "click", "verify", "confirm")
- `threat_count` (words like "suspended", "blocked", "unauthorized")
- `excessive_caps`, `excessive_punctuation`

**Step 3: Feature Combination:**
- TF-IDF features (500) + Scaled Numerical features (19) = **519 total features**
- Numerical features are scaled using **StandardScaler** (zero mean, unit variance)

**Step 4: Model Training** (`src/sms_detection/train_model.py`)
- 4 classifiers are trained and compared
- **Best model is selected by F1-Score**
- Data split: **80% training, 20% testing** (`TEST_SIZE=0.2`)
- 5-fold Cross-Validation for robust evaluation

### 2.2 URL Phishing Detection Algorithm

**Step 1: URL Feature Extraction** (`src/url_detection/url_feature_extractor.py`)

28 features extracted purely from URL string (NO external API calls):

| Category | Features |
|---|---|
| **Length** | url_length, hostname_length, path_length, query_length |
| **Characters** | num_dots, num_hyphens, num_underscores, num_slashes, num_at_signs, num_question_marks, num_equals, num_percent, num_digits |
| **Structure** | has_ip_address, has_https, has_http_in_domain, num_subdomains, has_port, has_double_slash_redirect, domain_has_digits, tld_length, is_shortened |
| **Keywords** | has_suspicious_words, has_brand_in_subdomain |
| **Entropy** | hostname_entropy (Shannon entropy) |
| **Ratios** | digit_to_letter_ratio, special_char_ratio |
| **Path** | path_depth |

**Shannon Entropy:** Measures randomness in hostname. Phishing URLs often have random-looking hostnames with high entropy.

**Step 2: Model Training** (`src/url_detection/train_url_model.py`)
- Models: Random Forest, Gradient Boosting, Logistic Regression, XGBoost
- Best model selected by F1-Score with hyperparameter tuning via GridSearchCV
- Dataset: `data/raw/phishing_urls.csv`

### 2.3 Visual Cloning Detection Algorithm (Remaining 30%)

**Two-Stage Process** (`src/visual_detection/image_comparator.py`):

**Stage 1 — Perceptual Hash (pHash):** Fast pre-filter
- Captures screenshot of suspect URL using Selenium headless Chrome
- Resizes to 256×256 thumbnail
- Computes pHash (perceptual hash) — a fingerprint based on visual appearance
- Compares against pre-stored hashes of trusted sites (SBI, HDFC, Google, PayPal, etc.)
- If Hamming distance > 30 → skip to "no match" (fast rejection)

**Stage 2 — SSIM (Structural Similarity Index):** Deep comparison
- Only runs if Stage 1 finds a close match (distance ≤ 30)
- Converts both images to grayscale
- Computes SSIM score (0 to 1, where 1 = identical)
- If SSIM ≥ 0.5 → **spoofing detected**
- Generates a difference heatmap showing where images differ

---

## 3. HOW DATA FLOWS THROUGH EACH LAYER

### 3.1 Complete Data Flow: User Input → Final Threat Score

```
USER types input in React Frontend (App.jsx)
        │
        ▼
Frontend sends HTTP POST via Axios to Flask API
        │
        ├── /api/analyze     (SMS only)
        ├── /api/analyze-url (URL only)
        └── /api/full-scan   (Combined: SMS + URL + Visual)
        │
        ▼
Flask API (src/api.py) receives request
        │
        ▼
┌─── SMS Analysis Path ────────────────────────┐
│ model_cache.predict(message)                  │
│   ├── extract_features_fast(text)             │
│   │     → 19 statistical features             │
│   ├── preprocess_text_fast(text)              │
│   │     → cleaned, stemmed text               │
│   ├── TF-IDF transform → 500 features        │
│   ├── StandardScaler transform → scaled nums  │
│   ├── Combine: [500 TF-IDF + 19 numerical]   │
│   ├── model.predict(features) → 0 or 1       │
│   └── model.predict_proba() → confidence      │
│         → threat_score = confidence × 100     │
└──────────────────────────────────────────────┘

┌─── URL Analysis Path ────────────────────────┐
│ url_predictor.predict(url)                    │
│   ├── URLFeatureExtractor.extract(url)        │
│   │     → 28 structural features              │
│   ├── model.predict(features) → 0 or 1       │
│   └── model.predict_proba()[:,1]              │
│         → threat_score (probability of class 1)│
└──────────────────────────────────────────────┘

┌─── Full Scan Weighted Combination ───────────┐
│ Weights: SMS=0.40, URL=0.45, Visual=0.15     │
│                                               │
│ combined_score = Σ(score_i × weight_i)        │
│                  / Σ(weight_i for active)      │
│                                               │
│ Risk Levels:                                  │
│   < 0.3  → LOW                                │
│   < 0.6  → MEDIUM                             │
│   < 0.85 → HIGH                               │
│   ≥ 0.85 → CRITICAL                           │
└──────────────────────────────────────────────┘
        │
        ▼
JSON response sent back to Frontend
        │
        ▼
React displays result in POPUP MODAL with:
  - Threat Score (large number)
  - Color-coded gauge bar
  - Risk Level badge
  - Feature breakdown cards
  - Warning messages if phishing
```

### 3.2 How Input Type is Determined (Content vs URL)

The frontend has **three separate tabs** in `App.jsx` (line 41-45):
- **"SMS / Text" tab:** User types message → calls `/api/analyze` → SMS analysis only
- **"URL Check" tab:** User types URL → calls `/api/analyze-url` → URL analysis only  
- **"Full Scan" tab:** User types message (may contain URLs) → calls `/api/full-scan`

In **Full Scan mode**, the API auto-extracts URLs from the message text using regex (`api.py` lines 294-304):
```python
url_pattern = re.compile(
    r'(?:https?://|www\.)[^\s<>"\']+|'
    r'[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s<>"\']*)?',
    re.IGNORECASE
)
extracted_urls = url_pattern.findall(message)
```
Then it runs **both SMS analysis on the full text AND URL analysis on extracted URLs**.

### 3.3 Threat Score Calculation

**For SMS Analysis** (`model_cache.py` line 201):
```python
threat_score = int(confidence * 100) if prediction == 1 else int((1 - confidence) * 100)
```
- If prediction = spam (1): threat_score = confidence × 100  
- If prediction = ham (0): threat_score = (1 - confidence) × 100

**For URL Analysis** (`url_predictor.py` line 148):
```python
threat_score = float(probabilities[1])  # probability of phishing class
```
- Direct probability from model's predict_proba for class 1 (phishing)

**For Full Scan Combined Score** (`api.py` lines 371-378):
```python
active_weights = {k: weights[k] for k in scores}
total_weight = sum(active_weights.values())
combined_score = sum(scores[k] * (active_weights[k] / total_weight) for k in scores)
```
- Only uses weights for analyses that were actually performed
- Re-normalizes weights so they still sum to 1.0

### 3.4 How the Popup Shows on Frontend

In `App.jsx`, when analysis completes:
1. `setResult({ type: "sms", data: res.data })` stores the result (line 120)
2. When `result` is not null, the **Result Modal overlay** renders (line 487-500)
3. The modal shows `SmsResult`, `UrlResult`, or `FullScanResult` component based on `result.type`
4. Each component displays the threat score with color-coded gauge bar and risk badges
5. Clicking outside the modal or the X button sets `result` to null, closing it

---

## 4. ALL CLASSIFIERS USED

### 4.1 SMS Detection Classifiers (`train_model.py` lines 39-56)

| Classifier | Library | Why Used |
|---|---|---|
| **Multinomial Naive Bayes** | `sklearn.naive_bayes.MultinomialNB` | Fast, works well with text/TF-IDF data, good baseline for spam detection |
| **Logistic Regression** | `sklearn.linear_model.LogisticRegression` | Linear classifier, fast, interpretable, works well for binary text classification |
| **Random Forest** | `sklearn.ensemble.RandomForestClassifier` | Ensemble of decision trees, handles non-linear patterns, resistant to overfitting |
| **Support Vector Machine (SVM)** | `sklearn.svm.SVC` | Finds optimal hyperplane boundary, works well in high-dimensional space (TF-IDF) |

### 4.2 URL Detection Classifiers (`train_url_model.py` lines 78-104)

| Classifier | Library | Why Used |
|---|---|---|
| **Random Forest** | `sklearn.ensemble.RandomForestClassifier` | Handles mixed feature types well, provides feature importances |
| **Gradient Boosting** | `sklearn.ensemble.GradientBoostingClassifier` | Sequential tree boosting, often highest accuracy |
| **Logistic Regression** | `sklearn.linear_model.LogisticRegression` | Simple baseline, fast, interpretable |
| **XGBoost** | `xgboost.XGBClassifier` | Advanced gradient boosting, best for structured/tabular data, fast |

### 4.3 Best Model Selection
- All models are trained and evaluated on the same test set
- **F1-Score** is the primary selection metric (harmonic mean of precision and recall)
- The winning model is saved as a `.pkl` file using `joblib`
- **Hyperparameter tuning** is done using `GridSearchCV` with 5-fold cross-validation

---

## 5. ALL LIBRARIES AND WHERE/WHY THEY ARE USED

| Library | Where Used | Why Used |
|---|---|---|
| **Flask** | `src/api.py` | Lightweight Python web framework to create REST API endpoints |
| **flask-cors** | `src/api.py` line 70 | Enables Cross-Origin requests so React frontend can talk to Flask backend |
| **scikit-learn** | `train_model.py`, `feature_extraction.py`, `train_url_model.py` | Core ML library: classifiers, TF-IDF, scaling, metrics, train/test split |
| **NLTK** | `preprocessing.py`, `model_cache.py` | Natural Language Toolkit: stopwords, PorterStemmer, word_tokenize |
| **NumPy** | Throughout | Numerical computing: array operations, feature vectors |
| **Pandas** | `preprocessing.py`, `feature_extraction.py`, training scripts | Data manipulation: loading CSV, DataFrame operations |
| **joblib** | `model_cache.py`, `predict.py`, training scripts | Efficient serialization of ML models (save/load .pkl files) |
| **matplotlib** | Training scripts | Generating plots: confusion matrices, ROC curves, feature importance |
| **seaborn** | Training scripts | Statistical visualization: heatmaps for confusion matrices |
| **Axios** | `frontend/src/App.jsx` | HTTP client for making API calls from React to Flask |
| **Recharts** | `frontend/src/App.jsx` | React charting library for threat history area chart |
| **Lucide React** | `frontend/src/App.jsx` | Icon library for UI icons (Shield, AlertTriangle, etc.) |
| **Vite** | `frontend/vite.config.js` | Fast build tool and dev server for React frontend |
| **Selenium** | `screenshot_capturer.py` | Captures website screenshots using headless Chrome browser |
| **OpenCV (cv2)** | `image_comparator.py` | Image processing: resize, grayscale conversion, heatmap generation |
| **scikit-image** | `image_comparator.py` | SSIM computation (structural_similarity function) |
| **imagehash** | `image_comparator.py` | Perceptual hashing (pHash) for fast image similarity pre-filtering |
| **Pillow (PIL)** | `build_trusted_db.py`, `image_comparator.py` | Image loading, resizing, thumbnail creation |
| **XGBoost** | `train_url_model.py` | Advanced gradient boosting classifier for URL detection |
| **tqdm** | `preprocessing.py` | Progress bar for dataset preprocessing |
| **webdriver-manager** | `screenshot_capturer.py` | Auto-downloads correct ChromeDriver version |

---

## 6. DATABASE — IS IT CONNECTED?

### **NO DATABASE IS USED IN THIS PROJECT.**

The project does **NOT** use any database (no MySQL, PostgreSQL, MongoDB, SQLite, etc.).

**How data is stored and retrieved instead:**

| Data | Storage Method | File |
|---|---|---|
| SMS Training Data | CSV file | `data/raw/sms_data.csv` |
| URL Training Data | CSV file | `data/raw/phishing_urls.csv` |
| Processed SMS Data | CSV file | `data/processed/sms_processed.csv` |
| SMS Features | CSV file | `data/processed/sms_features.csv` |
| Trained SMS Model | Pickle file (.pkl) | `data/models/sms_classifier.pkl` |
| Trained URL Model | Pickle file (.pkl) | `data/models/url_classifier.pkl` |
| Feature Extractor | Pickle file (.pkl) | `data/models/feature_extractor.pkl` |
| URL Feature Names | Pickle file (.pkl) | `data/models/url_feature_names.pkl` |

### How the Frontend Shows Existing Phishing Reports

The frontend **does NOT load any pre-existing phishing reports from files/database**.

Instead, the "scan history" table and statistics on the dashboard are built **entirely from the user's current browser session** using **React state** (in-memory):

```javascript
// App.jsx line 58-59
const [scanHistory, setScanHistory] = useState([]);
const [threatHistory, setThreatHistory] = useState([]);
```

- Every time a scan is performed, the result is added to `scanHistory` array (line 89-108)
- The dashboard stats (SMS Scans, URL Scans, Full Scans, Phishing Found, Safe Messages) are **computed in real-time** from this array (lines 62-69)
- The threat history chart data is built from `threatHistory` array
- **When you refresh the page, ALL history is lost** because it only lives in browser memory (React useState)

The **sample messages/URLs** shown as quick-test buttons are **hardcoded** in the frontend (`SAMPLE_MESSAGES`, `SAMPLE_URLS`, `SAMPLE_FULLSCAN` at lines 16-30).

---

## 7. DATASET DETAILS

### 7.1 SMS Dataset (`data/raw/sms_data.csv`)
- Contains SMS messages labeled as "ham" (legitimate) or "spam" (phishing)
- Columns: `label`, `message`
- Labels are mapped: ham → 0, spam → 1

### 7.2 URL Dataset (`data/raw/phishing_urls.csv`)
- Contains URLs labeled as legitimate (0) or phishing (1)
- Columns: `url`, `label`
- Used to train URL anomaly detection model
- **No external blacklists** — the model learns structural patterns from URLs

### 7.3 Trusted Screenshots (`data/trusted_screenshots/`)
- Reference screenshots of legitimate websites (SBI, HDFC, Google, PayPal, etc.)
- Built by `build_trusted_db.py` using Selenium
- Stored as full screenshots (1366×768) and thumbnails (256×256)

---

## 8. LINE-BY-LINE CODE EXPLANATION OF KEY FILES

### 8.1 `src/api.py` — Flask REST API (Main Entry Point)

| Lines | What It Does |
|---|---|
| 1-5 | Module docstring — describes the API |
| 6-13 | Imports: Flask, CORS, sys, os, re, urlparse, time |
| 16-18 | Sets PROJECT_ROOT and adds to sys.path |
| 22 | Imports `model_cache` — this triggers SMS model loading on import |
| 25-27 | Declares lazy-load singletons for URL and visual detection |
| 33-42 | `get_url_predictor()` — lazy loads URLPredictor on first call |
| 45-66 | `get_image_comparator()` and `get_screenshot_capturer()` — lazy loads |
| 69-70 | Creates Flask app, enables CORS for cross-origin requests |
| 77-88 | `/api/health` — health check endpoint, returns model status |
| 91-134 | `/api/analyze` — SMS analysis: gets message → model_cache.predict() → returns JSON |
| 141-196 | `/api/analyze-url` — URL analysis: validates URL → url_predictor.predict() → returns JSON |
| 199-263 | `/api/visual-check` — captures screenshot → compares against trusted DB → returns JSON |
| 266-411 | `/api/full-scan` — combined multi-channel analysis with weighted scoring |
| 294-304 | Regex to auto-extract URLs from message text |
| 315 | Weight configuration: SMS=0.40, URL=0.45, Visual=0.15 |
| 370-380 | Weighted average calculation for combined threat score |
| 382-390 | Risk level determination from combined score |
| 424-443 | Main block: prints endpoints and starts Flask on port 5000 |

### 8.2 `src/model_cache.py` — Singleton Model Cache

| Lines | What It Does |
|---|---|
| 28-33 | Singleton pattern: `_instance` and `_initialized` class variables |
| 36-39 | `__new__()` ensures only one instance ever exists |
| 41-96 | `__init__()` loads SMS model, feature extractor, NLP components ONCE |
| 54 | `joblib.load(SMS_MODEL_PATH)` — loads the trained classifier from .pkl |
| 58 | Loads TF-IDF vectorizer and StandardScaler from feature_extractor.pkl |
| 62-68 | Initializes PorterStemmer and loads NLTK stopwords |
| 72-75 | Pre-compiles regex patterns for URL, email, phone, number detection |
| 98-135 | `extract_features_fast()` — extracts 19 statistical features from raw text |
| 137-153 | `preprocess_text_fast()` — cleans text, removes patterns, stems words |
| 155-232 | `predict()` — the core prediction pipeline for SMS |
| 177 | TF-IDF transformation of preprocessed text |
| 184 | StandardScaler transformation of numerical features |
| 187-188 | Concatenates TF-IDF + numerical features, takes absolute values |
| 191 | `model.predict()` — makes the binary classification |
| 194-198 | Gets prediction probability for confidence score |
| 201 | Calculates threat_score from confidence |
| 237 | Creates the singleton instance — models load on import |

### 8.3 `src/sms_detection/preprocessing.py` — Text Preprocessing

| Lines | What It Does |
|---|---|
| 21-26 | `SMSPreprocessor` class — handles all text cleaning |
| 27-59 | `__init__()` — loads stopwords, initializes stemmer, compiles regex |
| 61-95 | `clean_text()` — lowercase → remove URLs/emails/phones/numbers → remove punctuation |
| 97-113 | `tokenize_text()` — splits text into word tokens using NLTK's word_tokenize |
| 115-125 | `remove_stopwords()` — filters out common English words |
| 127-145 | `stem_or_lemmatize()` — reduces words to root form |
| 147-170 | `preprocess_text()` — complete pipeline: clean → tokenize → stopwords → stem |
| 172-222 | `extract_text_features()` — extracts all 19 statistical features |
| 224-280 | `preprocess_dataset()` — applies preprocessing to entire DataFrame |

### 8.4 `src/sms_detection/feature_extraction.py` — TF-IDF + Feature Combination

| Lines | What It Does |
|---|---|
| 19-54 | `FeatureExtractor.__init__()` — configures TfidfVectorizer and StandardScaler |
| 34-44 | TF-IDF config: max_features=500, ngram_range=(1,2), min_df=2, max_df=0.8 |
| 56-116 | `fit_transform()` — fits TF-IDF on training text + scales numerical features + combines |
| 70 | `tfidf.fit_transform()` — learns vocabulary AND transforms to TF-IDF matrix |
| 99 | `scaler.fit_transform()` — learns mean/std AND scales numerical features |
| 109 | `pd.concat([tfidf_df, numerical_df])` — combines into final feature matrix |

### 8.5 `src/sms_detection/train_model.py` — Model Training Pipeline

| Lines | What It Does |
|---|---|
| 39-56 | Defines 4 models: Naive Bayes, Logistic Regression, Random Forest, SVM |
| 66-100 | `load_data()` — loads features CSV, splits 80/20 with stratification |
| 102-148 | `train_model()` — trains one model, evaluates accuracy/precision/recall/F1/ROC-AUC |
| 128 | 5-fold cross-validation for robust evaluation |
| 209-226 | `select_best_model()` — picks model with highest F1-score |
| 228-258 | `plot_confusion_matrices()` — creates confusion matrix heatmaps |
| 285-306 | `plot_roc_curves()` — plots ROC curves for all models |
| 336-390 | `hyperparameter_tuning()` — GridSearchCV for best parameters |

### 8.6 `src/url_detection/url_feature_extractor.py` — URL Feature Extraction

| Lines | What It Does |
|---|---|
| 15-19 | `SHORTENER_DOMAINS` — known URL shorteners (bit.ly, tinyurl.com, etc.) |
| 22-26 | `SUSPICIOUS_WORDS` — words common in phishing URLs (login, verify, secure, etc.) |
| 29-32 | `BRAND_NAMES` — legitimate brands often spoofed (paypal, google, sbi, etc.) |
| 40-50 | Class docstring explaining the 28 features |
| 71-76 | `_shannon_entropy()` — measures randomness in hostname string |
| 105-233 | `extract()` — extracts all 28 features from a single URL |
| 149 | IP address detection using regex pattern |
| 176 | URL shortener detection by checking against known domains |
| 179 | Suspicious keyword detection in URL |
| 182-188 | Brand spoofing: checks if brand name appears in subdomain but NOT in registered domain |

### 8.7 `src/url_detection/url_predictor.py` — URL Prediction

| Lines | What It Does |
|---|---|
| 23-31 | Singleton pattern for URLPredictor |
| 41-75 | Loads URL model, feature names, creates URLFeatureExtractor |
| 56-65 | Extracts feature importances from model (Random Forest → feature_importances_, Logistic Regression → coef_) |
| 88-108 | `_get_top_risk_features()` — ranks features by importance × value |
| 110-167 | `predict()` — extracts features → builds feature vector → predicts → returns result dict |
| 148 | `probabilities[1]` — probability of being phishing class |

### 8.8 `frontend/src/App.jsx` — React Frontend

| Lines | What It Does |
|---|---|
| 1-11 | Imports: React hooks, Lucide icons, Recharts, Axios |
| 13 | `API_URL = "http://localhost:5000"` — Flask backend address |
| 16-30 | Sample messages and URLs for quick testing buttons |
| 47-69 | Main App state: activeTab, message, url, result, loading, scanHistory |
| 72-82 | Health check polling every 3 seconds to show API online/offline status |
| 115-124 | `analyzeMessage()` — sends POST to /api/analyze, stores result |
| 127-136 | `analyzeUrl()` — sends POST to /api/analyze-url |
| 139-152 | `fullScan()` — sends POST to /api/full-scan |
| 172-178 | `getThreatColor()` — returns gradient colors based on threat score |
| 250-254 | Stat cards showing SMS/URL/Full scan counts |
| 267-300 | Threat history chart using Recharts AreaChart |
| 302-358 | Scan input panel with tabs, text area, URL input, scan button |
| 487-500 | Result modal overlay — shows popup with threat analysis results |
| 536-583 | `SmsResult` component — displays SMS analysis with threat gauge and feature cards |
| 586-647 | `UrlResult` component — displays URL analysis with quick indicators and risk features |
| 650-761 | `FullScanResult` component — displays combined analysis with score breakdown bars |

---

## 9. MODELS SAVED AND LOADED

| File | Created By | Loaded By | Contains |
|---|---|---|---|
| `sms_classifier.pkl` | `train_model.py` → `joblib.dump()` | `model_cache.py` → `joblib.load()` | Best SMS classifier (e.g., Random Forest) |
| `feature_extractor.pkl` | `feature_extraction.py` → `extractor.save()` | `model_cache.py` → `joblib.load()` | TF-IDF vectorizer + StandardScaler + feature names |
| `url_classifier.pkl` | `train_url_model.py` → `joblib.dump()` | `url_predictor.py` → `joblib.load()` | Best URL classifier (e.g., XGBoost) |
| `url_feature_names.pkl` | `train_url_model.py` → `joblib.dump()` | `url_predictor.py` → `joblib.load()` | Ordered list of 28 URL feature names |

---

## 10. KEY VIVA QUESTIONS & ANSWERS

**Q: What type of ML is this? Supervised or Unsupervised?**
A: **Supervised Learning** — we have labeled datasets (ham/spam for SMS, legitimate/phishing for URLs) and train classifiers on them.

**Q: What is the classification type?**
A: **Binary Classification** — each input is classified into one of two classes (phishing or legitimate).

**Q: What is F1-Score and why is it the primary metric?**
A: F1 = 2 × (Precision × Recall) / (Precision + Recall). It balances both metrics. In phishing detection, both false positives (blocking legitimate) and false negatives (missing phishing) are costly.

**Q: What is a Confusion Matrix?**
A: A 2×2 table showing: True Positives (correctly detected phishing), True Negatives (correctly cleared legitimate), False Positives (legitimate flagged as phishing), False Negatives (phishing missed).

**Q: What is ROC Curve and AUC?**
A: ROC plots True Positive Rate vs False Positive Rate at various thresholds. AUC (Area Under Curve) measures overall model quality — closer to 1.0 is better.

**Q: How does the system detect zero-day attacks?**
A: It uses **structural analysis** of URLs (not blacklists). Features like hostname_entropy, has_ip_address, brand_in_subdomain detect patterns common in new phishing URLs that haven't been reported yet.

**Q: What is the Singleton Pattern used in the code?**
A: Both `ModelCache` and `URLPredictor` use Singleton pattern — only ONE instance is created. This ensures models are loaded once into memory and reused for all requests, giving ~100-200ms response time.

**Q: What is SSIM?**
A: Structural Similarity Index Measure — compares luminance, contrast, and structure between two images. Score ranges 0-1 where 1 means identical images. Used to detect if a phishing site visually copies a legitimate site.

**Q: How are the weights decided for Full Scan? (SMS=0.40, URL=0.45, Visual=0.15)**
A: URL analysis gets the highest weight (0.45) because URL structure is the strongest indicator of phishing. SMS content analysis gets 0.40 as it detects social engineering patterns. Visual detection gets 0.15 as supplementary evidence since it's computationally expensive and not always available.
