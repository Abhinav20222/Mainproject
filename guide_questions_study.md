# Guide Questions - Study File
## PhishGuard AI - Phishing Detection System

---

# ═══════════════════════════════════════════════════════════════
# QUESTION 1: Which Tool is Used to Train — Jupyter or Colab?
# ═══════════════════════════════════════════════════════════════

## Answer: We Used Python Scripts (.py Files), NOT Jupyter/Colab

We trained our models using **standalone Python scripts** (`.py` files), NOT Jupyter Notebook or Google Colab.

### The Training Files:
| Purpose | File Path |
|---------|-----------|
| SMS model training | `src/sms_detection/train_model.py` |
| URL model training | `src/url_detection/train_url_model.py` |

### How We Run Training:
```bash
# SMS model training
python src/sms_detection/train_model.py

# URL model training
python src/url_detection/train_url_model.py
```

### Why .py Scripts Instead of Jupyter/Colab?

| Reason | Explanation |
|--------|-------------|
| **Production Ready** | `.py` scripts can be directly integrated into the backend server (Flask API) |
| **No Manual Steps** | The entire training pipeline runs automatically — load data → train → evaluate → save model |
| **Reproducible** | Running the same script always produces the same results (we use `random_state=42`) |
| **Version Control** | `.py` files work cleanly with Git, unlike `.ipynb` files which have JSON metadata noise |
| **Server Deployment** | We can retrain models on any server without needing a browser/notebook interface |

### If Guide Asks "Could You Have Used Jupyter/Colab?":
Yes, we could have. The libraries are the same (scikit-learn, pandas, numpy). But for a **production system** with a Flask API backend, using `.py` scripts is the **professional and correct approach** because the training code needs to integrate with the deployment pipeline.

---

# ═══════════════════════════════════════════════════════════════
# QUESTION 2: Explain the Training Code (Line by Line)
# ═══════════════════════════════════════════════════════════════

## A) SMS Training Code — `src/sms_detection/train_model.py`

### Step 1: Imports (Lines 1–21)
```python
import pandas as pd                # For loading CSV data into DataFrames
import numpy as np                  # For numerical operations (arrays, math)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# train_test_split → splits data into training (80%) and testing (20%)
# cross_val_score → validates model by splitting training data into 5 folds
# GridSearchCV → tries multiple hyperparameter combinations to find the best

from sklearn.naive_bayes import MultinomialNB           # Naive Bayes model
from sklearn.ensemble import RandomForestClassifier     # Random Forest model
from sklearn.linear_model import LogisticRegression     # Logistic Regression model
from sklearn.svm import SVC                             # Support Vector Machine model

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve)
# These are evaluation metrics to measure how good each model is

import joblib              # To save the trained model to a .pkl file
import matplotlib.pyplot   # For creating charts and plots
import seaborn as sns      # For beautiful confusion matrix heatmaps
```

### Step 2: Class Initialization — `__init__` (Lines 28–64)
```python
class SMSModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state   # 42 = fixed seed for reproducibility

        # Initialize 4 different models to compare:
        self.models = {
            'Naive Bayes': MultinomialNB(),
            # ↑ Probabilistic classifier, good for text data
            # Uses Bayes theorem to calculate probability of spam

            'Logistic Regression': LogisticRegression(
                max_iter=1000,           # max iterations to converge
                random_state=random_state,
                solver='liblinear'       # optimization algorithm
            ),
            # ↑ Linear classifier that draws a decision boundary

            'Random Forest': RandomForestClassifier(
                n_estimators=100,        # 100 decision trees in the forest
                random_state=random_state,
                n_jobs=-1                # use ALL CPU cores for speed
            ),
            # ↑ Ensemble of 100 decision trees, each votes on the result

            'SVM': SVC(
                kernel='linear',         # linear decision boundary
                probability=True,        # enable probability estimation
                random_state=random_state
            ),
            # ↑ Finds the optimal hyperplane to separate spam from ham
        }

        self.results = {}              # stores results for each model
        self.best_model = None         # will hold the winner
        self.best_model_name = None    # name of the winner
```

### Step 3: Load Data — `load_data()` (Lines 66–100)
```python
def load_data(self):
    # Load the preprocessed features CSV file
    df = pd.read_csv(SMS_FEATURES_DATA)
    # This CSV contains: TF-IDF features (500 columns) + statistical features
    # + label_encoded column (0=ham, 1=spam)

    # Separate features (X) from labels (y)
    X = df.drop('label_encoded', axis=1)   # Everything EXCEPT the label
    y = df['label_encoded']                 # Just the label column (0 or 1)

    X = X.fillna(0)   # Replace any missing values with 0
    X = X.abs()        # Make all values positive (needed for Naive Bayes)

    # Split into training (80%) and testing (20%)
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y,
        test_size=0.2,              # 20% for testing
        random_state=self.random_state,  # reproducibility
        stratify=y                  # maintain same spam/ham ratio in both sets
    )
    # stratify=y means: if original data has 13% spam,
    # both train and test will have ~13% spam
```

### Step 4: Train a Single Model — `train_model()` (Lines 102–148)
```python
def train_model(self, name, model):
    # TRAIN: Feed training data to the model
    model.fit(self.X_train, self.y_train)
    # model.fit() = the model LEARNS patterns from the training data
    # X_train = features (numbers), y_train = labels (0 or 1)

    # PREDICT: Ask the model to classify test data
    y_pred = model.predict(self.X_test)
    # The model has NEVER seen this test data before
    # y_pred = array of 0s and 1s (model's guesses)

    # GET PROBABILITIES: How confident is the model?
    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
    # [:, 1] = get probability of being spam (class 1)
    # Example: 0.92 means 92% confident it's spam

    # CALCULATE METRICS:
    accuracy = accuracy_score(self.y_test, y_pred)
    # What % of ALL predictions were correct?

    precision = precision_score(self.y_test, y_pred)
    # Of all messages the model SAID were spam, what % actually WERE spam?

    recall = recall_score(self.y_test, y_pred)
    # Of all ACTUAL spam messages, what % did the model catch?

    f1 = f1_score(self.y_test, y_pred)
    # Harmonic mean of precision and recall (balanced single score)

    # CROSS-VALIDATION: Test model stability
    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
    # Splits training data into 5 folds, trains on 4, tests on 1
    # Repeats 5 times with different splits
    # Checks if model performs consistently, not just on one lucky split
```

### Step 5: Train All Models → Compare → Select Best (Lines 150–226)
```python
def train_all_models(self):
    for name, model in self.models.items():
        self.results[name] = self.train_model(name, model)
    # Trains all 4 models one by one, stores their results

def select_best_model(self):
    for name, result in self.results.items():
        if result['f1_score'] > best_f1:
            self.best_model_name = name
            self.best_model = result['model']
    # Compares F1-scores, picks the highest one as the winner
```

### Step 6: Hyperparameter Tuning (Lines 336–390)
```python
def hyperparameter_tuning(self):
    # Only for Random Forest (if it won):
    param_grid = {
        'n_estimators': [50, 100, 200],      # try 50, 100, or 200 trees
        'max_depth': [10, 20, None],          # how deep each tree can grow
        'min_samples_split': [2, 5, 10],      # min samples to split a node
        'min_samples_leaf': [1, 2, 4]         # min samples at a leaf
    }

    grid_search = GridSearchCV(
        model, param_grid,
        cv=5,             # 5-fold cross validation
        scoring='f1',     # optimize for F1-score
        n_jobs=-1         # use all CPU cores
    )
    grid_search.fit(self.X_train, self.y_train)
    # Tries ALL combinations (3×3×3×3 = 81 combinations × 5 folds = 405 trainings)
    # Picks the combination that gives the best F1-score
```

### Step 7: Save Model (Lines 308–334)
```python
def save_best_model(self):
    joblib.dump(self.best_model, filepath)
    # Saves the trained model to: data/models/sms_classifier.pkl
    # This .pkl file contains:
    #   - All 100 decision trees (with all their nodes and splits)
    #   - The learned patterns from training data
    #   - Ready to make predictions without retraining
```

### Step 8: Main Pipeline (Lines 393–448)
```python
def main():
    trainer = SMSModelTrainer()     # 1. Create trainer
    trainer.load_data()             # 2. Load & split data
    trainer.train_all_models()      # 3. Train 4 models
    trainer.compare_models()        # 4. Compare performance
    trainer.plot_confusion_matrices()  # 5. Visualize errors
    trainer.plot_roc_curves()       # 6. Plot ROC curves
    trainer.select_best_model()     # 7. Pick the winner
    trainer.generate_classification_report()  # 8. Detailed report
    trainer.hyperparameter_tuning() # 9. Fine-tune the winner
    trainer.save_best_model()       # 10. Save to .pkl file
```

---

## B) URL Training Code — `src/url_detection/train_url_model.py`

### Step 1: Imports (Lines 1–53)
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
# 3 main models + XGBoost if installed

try:
    from xgboost import XGBClassifier    # Try importing XGBoost
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False                  # Skip if not installed

from src.url_detection.url_feature_extractor import URLFeatureExtractor
# Custom class that extracts ~20 numerical features from a raw URL string
# Features like: url_length, has_ip_address, num_dots, has_https, etc.
```

### Step 2: Model Initialization (Lines 78–104)
```python
def _init_models(self):
    self.models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,        # 200 trees (more than SMS because URL features are fewer)
            class_weight='balanced', # automatically handle class imbalance
            random_state=42,
            n_jobs=-1                # use all CPU cores
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,        # 200 sequential trees
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
    }
    if HAS_XGBOOST:
        self.models['XGBoost'] = XGBClassifier(
            n_estimators=200,
            eval_metric='logloss',   # loss function
            random_state=42,
            verbosity=0              # silent mode
        )
```

### Step 3: Load Data & Extract Features (Lines 106–139)
```python
def load_data(self):
    # Load raw URLs from CSV
    df = pd.read_csv(URL_DATA_PATH)    # Contains 'url' and 'label' columns
    # label: 0 = legitimate, 1 = phishing

    # Extract numerical features from each URL
    X = self.feature_extractor.extract_batch(df['url'].tolist())
    # For each URL, extracts ~20 features:
    # url_length, num_dots, num_hyphens, num_subdomains,
    # has_ip_address, has_https, path_length, has_suspicious_words, etc.
    # Returns a DataFrame with ~20 columns

    y = df['label'].values    # Labels: 0 or 1

    # Split into train (80%) and test (20%)
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle any bad values
    self.X_train = self.X_train.fillna(0).replace([np.inf, -np.inf], 0)
    self.X_test = self.X_test.fillna(0).replace([np.inf, -np.inf], 0)
```

### Step 4: Train Single Model (Lines 141–178)
```python
def train_model(self, name, model):
    # Cross-validation first (5 folds)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='f1')
    # StratifiedKFold ensures each fold has same phishing/legit ratio

    # Train on full training set
    model.fit(self.X_train, self.y_train)

    # Predict on test set
    y_pred = model.predict(self.X_test)

    # Calculate metrics
    acc = accuracy_score(self.y_test, y_pred)
    prec = precision_score(self.y_test, y_pred)
    rec = recall_score(self.y_test, y_pred)
    f1 = f1_score(self.y_test, y_pred)
```

### Step 5: Hyperparameter Tuning (Lines 209–265)
```python
def hyperparameter_tuning(self):
    param_grids = {
        'Random Forest': {
            'n_estimators': [200, 300],
            'max_depth': [None, 20, 30],
            'min_samples_split': [2, 5],
        },
        'XGBoost': {
            'n_estimators': [200, 300],
            'max_depth': [5, 7, 10],
            'learning_rate': [0.05, 0.1],
        },
        # ... more grids for each model
    }
    # GridSearchCV tries all combinations to find the best hyperparameters
```

### Step 6: Save Model (Lines 379–405)
```python
def save_model(self):
    joblib.dump(self.best_model, URL_MODEL_PATH)         # Save model: url_classifier.pkl
    joblib.dump(feature_names, URL_FEATURE_NAMES_PATH)   # Save feature names: url_feature_names.pkl
    joblib.dump(info, URL_MODEL_INFO_PATH)               # Save metadata: url_model_info.pkl
```

---

# ═══════════════════════════════════════════════════════════════
# QUESTION 3: All Models Used in SMS and URL Training
# ═══════════════════════════════════════════════════════════════

## SMS Models (4 Models):

### 1. Naive Bayes (MultinomialNB)
- **What it does**: Uses probability (Bayes Theorem) to classify text
- **How it works**: Calculates `P(spam | words)` — "given these words, what's the probability it's spam?"
- **Input**: TF-IDF word frequencies (500 features)
- **Strengths**: Very fast, works well with text data, simple
- **Weaknesses**: Assumes all features are independent (which they're not)

### 2. Logistic Regression
- **What it does**: Finds a linear decision boundary between spam and ham
- **How it works**: Applies sigmoid function: `P(spam) = 1 / (1 + e^(-z))` where z = weighted sum of features
- **Configuration**: `max_iter=1000` (enough iterations to converge), `solver='liblinear'`
- **Strengths**: Fast, interpretable, good baseline
- **Weaknesses**: Can only learn linear boundaries, struggles with complex patterns

### 3. Random Forest 🏆 (Winner)
- **What it does**: Creates 100 decision trees, each votes on the classification
- **How it works**: Each tree sees a random subset of data and features, makes its own decision. Final answer = majority vote
- **Configuration**: `n_estimators=100`, `n_jobs=-1` (all CPU cores)
- **Strengths**: Handles non-linear patterns, resistant to overfitting, works with any data
- **Weaknesses**: Slower than simple models, harder to interpret

### 4. SVM (Support Vector Machine)
- **What it does**: Finds the best hyperplane that separates spam from ham with maximum margin
- **How it works**: Maximizes the gap (margin) between the two classes
- **Configuration**: `kernel='linear'`, `probability=True`
- **Strengths**: Very effective in high-dimensional spaces
- **Weaknesses**: Very slow on large datasets, memory intensive

---

## URL Models (3-4 Models):

### 1. Random Forest 🏆 (Winner)
- **Configuration**: `n_estimators=200`, `class_weight='balanced'` (handles imbalanced data)
- **Input**: ~20 structural URL features (not text)

### 2. Gradient Boosting
- **What it does**: Builds trees sequentially, each tree corrects mistakes of previous trees
- **How it works**: Uses "gradient descent" to minimize prediction errors step by step
- **Configuration**: `n_estimators=200`
- **Strengths**: Often highest accuracy, learns from mistakes
- **Weaknesses**: Slow training, risk of overfitting

### 3. Logistic Regression
- Same as SMS but with `class_weight='balanced'` to handle more phishing URLs than legitimate ones

### 4. XGBoost (eXtreme Gradient Boosting) — Optional
- **What it does**: An optimized, faster version of Gradient Boosting
- **Configuration**: `n_estimators=200`, `eval_metric='logloss'`
- **Strengths**: Fastest boosting method, handles missing values, built-in regularization
- **Note**: Only used if `xgboost` package is installed

---

# ═══════════════════════════════════════════════════════════════
# QUESTION 4: Why Choose Random Forest Over Others?
# (Reasons BEYOND F1-Score)
# ═══════════════════════════════════════════════════════════════

## Reason 1: Robustness Against Overfitting
Random Forest uses **bagging** (bootstrap aggregation) — each tree trains on a random sample of the data. This means:
- One tree might overfit, but 100 trees **average out** their mistakes
- SVM and Logistic Regression don't have this self-correcting mechanism
- Naive Bayes assumes feature independence, which is wrong for text data

**Example**: If one phishing message has unusual words, a single tree might memorize it. But 100 trees together will ignore that noise because most trees didn't see that particular message.

## Reason 2: Handles Non-Linear Relationships
Phishing patterns are NOT linear:
- A message with 1 urgency word might be ham
- A message with 3 urgency words + a URL is almost certainly spam
- This is a **non-linear interaction** between features

| Model | Can Handle Non-Linear? |
|-------|----------------------|
| Naive Bayes | ❌ No — assumes features are independent |
| Logistic Regression | ❌ No — draws a straight line boundary |
| SVM (linear) | ❌ No — only with non-linear kernels (which are very slow) |
| **Random Forest** | ✅ **Yes** — trees naturally capture complex if-then rules |

## Reason 3: Feature Importance (Interpretability for Security)
Random Forest gives `feature_importances_` — which tells us WHICH features matter most.

This is critical for a **security application** because:
- We can explain to users WHY a message was flagged
- We can identify if a new type of phishing attack targets different features
- We can validate that the model is learning real patterns, not noise

**Example output**: "URL presence contributed 25%, urgency words contributed 20%, financial keywords contributed 15%"

Naive Bayes and SVM don't provide this level of feature-level explainability.

## Reason 4: Works with Both Text (TF-IDF) and Numerical Features
Our SMS model uses a MIX of features:
- **TF-IDF features** (500 columns) — word frequencies
- **Statistical features** — message_length, uppercase_ratio, urgency_count, etc.

Random Forest handles this mixed feature space naturally. Each tree splits on whatever feature gives the best separation — whether it's a TF-IDF weight or a numeric count.

Logistic Regression and Naive Bayes treat all features the same (linear combination), which loses the interaction effects.

## Reason 5: No Feature Scaling Required
Random Forest doesn't need feature normalization:
- SVM requires all features to be on the same scale (0-1)
- Logistic Regression works better with scaled features
- **Random Forest doesn't care** — it only looks at "is value > threshold?"

This makes the pipeline simpler and less error-prone.

## Reason 6: Fast Prediction Time (Critical for Real-Time API)
Our system is a **live API** that needs to respond in milliseconds:
- Random Forest prediction: **~5-10ms** (just traverse 100 small trees)
- SVM prediction: **~50-100ms** (needs kernel computation)
- Gradient Boosting: ~10-15ms (sequential trees)

For a real-time phishing detection API, prediction speed matters.

## Reason 7: Parallel Training and Prediction
`n_jobs=-1` means Random Forest uses ALL CPU cores:
- 100 trees train independently → **parallel training**
- 100 trees predict independently → **parallel prediction**
- SVM and Gradient Boosting are inherently sequential

## Reason 8: Cross-Validation Stability
Random Forest showed the **lowest variance** across all 5 cross-validation folds:
- Low variance = consistent performance regardless of which data split we use
- High variance = model is unstable and may fail on new data
- This is especially important for security — we need RELIABLE detection

---

# ═══════════════════════════════════════════════════════════════
# QUESTION 5: Line-by-Line Explanation of 3 Analysis Modes
# ═══════════════════════════════════════════════════════════════

## Analysis Mode 1: SMS Only — `/api/analyze`
**File**: `src/api.py` → Lines 91–134

```python
@app.route('/api/analyze', methods=['POST'])
def analyze_message():
    # This endpoint ONLY analyzes the TEXT content of a message

    start_time = time.time()
    # Records when request started (for measuring response time)

    if not model_cache.is_ready:
        return jsonify({'error': 'Models not loaded'}), 503
    # Check if the SMS model is loaded in memory
    # model_cache is a singleton that loads the model ONCE when the server starts

    data = request.get_json()
    # Parse the JSON body sent from frontend
    # Expected format: {"message": "Your suspicious text here"}

    if not data or 'message' not in data:
        return jsonify({'error': 'Missing message field'}), 400
    # Validate that the request contains a 'message' field

    message = data['message'].strip()
    # Remove whitespace from start/end

    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    # Don't process empty strings

    result = model_cache.predict(message)
    # THIS IS THE CORE LINE — sends the message through:
    #   1. SMSPreprocessor → cleans text, removes stopwords, stems words
    #   2. Feature Extractor → converts text to TF-IDF + statistical features
    #   3. Random Forest Model → predicts 0 (ham) or 1 (spam)
    #   4. Returns: {prediction, confidence, threat_score, features}

    processing_time = (time.time() - start_time) * 1000
    result['processing_time_ms'] = round(processing_time, 2)
    # Calculate how long the analysis took in milliseconds

    return jsonify(result)
    # Send the result back to the frontend as JSON
    # Example response:
    # {
    #   "prediction": "spam",
    #   "confidence": 0.95,
    #   "threat_score": 95,
    #   "features": {"urgency_keywords": 2, "has_url": true, ...},
    #   "processing_time_ms": 12.5
    # }
```

**What SMS Analysis Does (Summary)**:
1. Takes raw text message
2. Preprocesses: lowercase, remove stopwords, stem words
3. Extracts features: TF-IDF (500 most important words) + statistical features (urgency count, URL presence, etc.)
4. Feeds features to trained Random Forest model
5. Returns prediction (ham/spam) + confidence + threat score + key indicators

---

## Analysis Mode 2: URL Only — `/api/analyze-url`
**File**: `src/api.py` → Lines 141–196

```python
@app.route('/api/analyze-url', methods=['POST'])
def analyze_url():
    # This endpoint ONLY analyzes the STRUCTURE of a URL

    start_time = time.time()

    data = request.get_json()
    # Expected format: {"url": "http://suspicious-site.com/login"}

    if not data or 'url' not in data:
        return jsonify({'success': False, 'error': 'Missing url field'}), 400

    url = data['url'].strip()
    if not url:
        return jsonify({'success': False, 'error': 'URL cannot be empty'}), 400

    # Validate URL format
    try:
        parsed = urlparse(url if '://' in url else 'http://' + url)
        if not parsed.hostname:
            raise ValueError("No hostname")
    except Exception:
        return jsonify({'success': False, 'error': 'Malformed URL'}), 422
    # urlparse breaks URL into parts: scheme, hostname, path, query
    # Example: "http://evil.com/login?id=1"
    #   scheme = "http"
    #   hostname = "evil.com"
    #   path = "/login"
    #   query = "id=1"

    predictor = get_url_predictor()
    # Lazy-loads the URL model (loads on first use, reuses after)
    # This is the Singleton pattern — only one instance exists

    if predictor is None or not predictor.is_ready:
        return jsonify({'success': False, 'error': 'URL model not loaded'}), 503

    result = predictor.predict(url)
    # THIS IS THE CORE LINE — sends the URL through:
    #   1. URLFeatureExtractor.extract(url) → extracts ~20 structural features:
    #      url_length, num_dots, num_hyphens, num_subdomains,
    #      has_ip_address, has_https, path_length, num_query_params,
    #      has_suspicious_words, has_brand_in_subdomain, etc.
    #   2. Builds feature vector in correct order
    #   3. Random Forest Model → predicts 0 (legit) or 1 (phishing)
    #   4. predict_proba → probability of being phishing
    #   5. Returns: {is_phishing, threat_score, risk_level, features, top_risk_features}

    analysis_time = (time.time() - start_time) * 1000
    return jsonify({
        'success': True,
        'url': result['url'],
        'is_phishing': result['is_phishing'],        # True/False
        'threat_score': result['threat_score'],        # 0.0 to 1.0
        'risk_level': result['risk_level'],            # LOW/MEDIUM/HIGH/CRITICAL
        'top_risk_features': result['top_risk_features'],  # Which features triggered detection
        'features': result['features'],                # All extracted features
        'analysis_time_ms': round(analysis_time, 2),
    })
```

**What URL Analysis Does (Summary)**:
1. Takes raw URL string
2. Parses and validates URL format
3. Extracts ~20 structural features (NO text analysis — only URL structure)
4. Feeds features to trained Random Forest model
5. Returns prediction + threat score + risk level + top risk features

**Key Difference from SMS**:
- SMS analyzes the **text content** (words, meaning)
- URL analyzes the **URL structure** (length, dots, IP address, path depth)
- Completely different models trained on different datasets

---

## Analysis Mode 3: Full Scan — `/api/full-scan`
**File**: `src/api.py` → Lines 266–411

```python
@app.route('/api/full-scan', methods=['POST'])
def full_scan():
    # COMBINED multi-channel analysis
    # Runs SMS + URL + Visual (optional) analysis together
    # and produces a WEIGHTED combined threat score

    start_time = time.time()

    data = request.get_json()
    message = data.get('message', '').strip()
    explicit_url = data.get('url', '').strip()
    include_visual = data.get('include_visual', False)
    # include_visual: if True, also captures a screenshot and compares visually

    # ═══ STEP 1: Auto-extract URLs from the message text ═══
    url_pattern = re.compile(
        r'(?:https?://|www\.)[^\s<>"\']+|'
        r'[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s<>"\']*)?',
        re.IGNORECASE
    )
    extracted_urls = url_pattern.findall(message) if message else []
    # This regex finds ALL URLs inside the message text
    # Example: "Click http://evil.com to verify" → extracts ["http://evil.com"]

    # Normalize: add http:// if missing
    extracted_urls = [
        u if u.startswith(('http://', 'https://')) else 'http://' + u
        for u in extracted_urls
    ]

    # Initialize weight system
    weights = {'sms': 0.40, 'url': 0.45, 'visual': 0.15}
    # SMS contributes 40% to final score
    # URL contributes 45% to final score (most important!)
    # Visual contributes 15% to final score
    scores = {}

    # ═══ STEP 2: SMS Analysis (on full message text) ═══
    if message and model_cache.is_ready:
        sms_result = model_cache.predict(message)
        sms_score = sms_result.get('threat_score', 0)
        if sms_score > 1:
            sms_score = sms_score / 100.0    # Normalize to 0-1 range
        scores['sms'] = sms_score
        analyses_performed.append('sms')
    # Runs the SAME SMS analysis as /api/analyze
    # Gets threat score for the text content

    # ═══ STEP 3: URL Analysis (on EACH extracted URL) ═══
    if extracted_urls:
        predictor = get_url_predictor()
        if predictor and predictor.is_ready:
            best_url_result = None
            best_url_score = -1
            for u in extracted_urls:
                url_result = predictor.predict(u)
                score = url_result.get('threat_score', 0)
                if score > best_url_score:
                    best_url_score = score
                    best_url_result = url_result
            # If message contains multiple URLs, we check ALL of them
            # But we keep the WORST (highest threat) score
            # "A message is as dangerous as its most dangerous URL"
            scores['url'] = best_url_score
            analyses_performed.append('url')

    # ═══ STEP 4: Visual Analysis (optional, on first URL) ═══
    if extracted_urls and include_visual:
        screenshot_path = capturer.capture(extracted_urls[0])
        # Takes a screenshot of the website using Selenium WebDriver

        vis_result = comparator.compare(screenshot_path)
        # Compares screenshot against trusted site database
        # Uses SSIM (Structural Similarity Index) and pHash (Perceptual Hash)
        # If a phishing site LOOKS like a bank website → spoofing detected!

        scores['visual'] = vis_result.get('visual_threat_score', 0)
        analyses_performed.append('visual')

    # ═══ STEP 5: Compute COMBINED weighted score ═══
    if scores:
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

    # ═══ STEP 6: Determine risk level ═══
    if combined_score < 0.3:
        risk_level = "LOW"
    elif combined_score < 0.6:
        risk_level = "MEDIUM"
    elif combined_score < 0.85:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    return jsonify({
        'success': True,
        'combined_threat_score': round(combined_score, 4),
        'risk_level': risk_level,
        'sms_analysis': sms_analysis,       # Full SMS results
        'url_analysis': url_analysis,       # Full URL results
        'visual_analysis': visual_analysis, # Full visual results (if performed)
        'analyses_performed': analyses_performed,  # ['sms', 'url', 'visual']
        'score_weights': {...},             # Actual weights used
        'total_analysis_time_ms': ...,
    })
```

**What Full Scan Does (Summary)**:
1. Takes a full message text
2. Auto-extracts ALL URLs from the text using regex
3. Runs SMS analysis on the text (40% weight)
4. Runs URL analysis on every extracted URL — keeps worst score (45% weight)
5. Optionally runs Visual analysis — screenshots + image comparison (15% weight)
6. Combines scores with normalized weights
7. Returns combined threat score + risk level + individual analysis results

---

# ═══════════════════════════════════════════════════════════════
# QUESTION 6: Frontend useState Diagram & Code Explanation
# ═══════════════════════════════════════════════════════════════

## What is useState?
`useState` is a React Hook that creates a **state variable**. When the state changes, the UI **automatically re-renders** to show the new value.

```
useState Flow:
┌──────────────┐     setState()      ┌──────────────┐
│  Old State   │ ──────────────────► │  New State   │
│  value = X   │                     │  value = Y   │
└──────────────┘                     └──────┬───────┘
                                            │
                                     React Re-renders
                                            │
                                     ┌──────▼───────┐
                                     │  Updated UI  │
                                     │  shows Y     │
                                     └──────────────┘
```

## All useState Variables in App.jsx (Line 48–59)

```
┌─────────────────────────── App Component ───────────────────────────┐
│                                                                      │
│  ┌── Navigation State ──┐   ┌── Input State ────────────┐           │
│  │ activeNav="dashboard"│   │ message=""                 │           │
│  │ activeTab="sms"      │   │ url=""                     │           │
│  └──────────────────────┘   │ includeVisual=false        │           │
│                              └──────────────────────────┘           │
│  ┌── Result State ──────┐   ┌── UI State ───────────────┐           │
│  │ result=null          │   │ loading=false              │           │
│  │ error=null           │   │ apiOnline=false            │           │
│  │ showHeatmap=false    │   └──────────────────────────┘           │
│  └──────────────────────┘                                           │
│  ┌── History State ──────────────────────────────────────┐          │
│  │ scanHistory=[]    ← stores last 20 scan results       │          │
│  │ threatHistory=[]  ← stores last 15 chart data points  │          │
│  └───────────────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────────┘
```

### Line-by-Line Explanation:

```javascript
// Line 48:
const [activeNav, setActiveNav] = useState("dashboard");
// Controls which sidebar item is highlighted
// Values: "dashboard", "sms", "url", "fullscan", "history", "settings"
// When user clicks sidebar → setActiveNav("sms") → sidebar highlights "SMS Scan"

// Line 49:
const [activeTab, setActiveTab] = useState("sms");
// Controls which scan tab is shown in the input panel
// Values: "sms", "url", "fullscan"
// When activeTab = "sms" → shows textarea for message input
// When activeTab = "url" → shows text input for URL
// When activeTab = "fullscan" → shows textarea + visual checkbox

// Line 50:
const [message, setMessage] = useState("");
// Stores the text entered by user in the SMS/Full Scan textarea
// Updated on every keystroke: onChange={e => setMessage(e.target.value)}
// Used in: analyzeMessage() and fullScan() API calls

// Line 51:
const [url, setUrl] = useState("");
// Stores the URL entered by user in the URL scan input
// Used in: analyzeUrl() API call

// Line 52:
const [includeVisual, setIncludeVisual] = useState(false);
// Checkbox state: whether to include visual spoofing analysis in full scan
// When true → full scan also captures screenshot and does image comparison

// Line 53:
const [result, setResult] = useState(null);
// Stores the API response after analysis is complete
// When null → no result modal shown
// When set → result modal appears with: {type: "sms"/"url"/"fullscan", data: {...}}
// Setting to null closes the modal: onClick={() => setResult(null)}

// Line 54:
const [loading, setLoading] = useState(false);
// true while API call is in progress
// When true → scan button shows spinner + "Analyzing…"
// When false → scan button shows "Initiate Threat Scan"
// Set to true before API call, false in finally block

// Line 55:
const [error, setError] = useState(null);
// Stores error message if API call fails
// When set → red error banner appears below the input panel
// Example: "Failed to connect to API."

// Line 56:
const [apiOnline, setApiOnline] = useState(false);
// Tracks whether the Flask backend is running
// Updated every 3 seconds by health check (useEffect on line 72)
// When false → sidebar shows "AI Offline" in red, scan button is disabled
// When true → sidebar shows "AI Engine Online" in green

// Line 57:
const [showHeatmap, setShowHeatmap] = useState(false);
// Controls the visual heatmap modal
// When true → shows the difference heatmap image from visual analysis
// Only relevant when full scan includes visual analysis

// Line 58:
const [scanHistory, setScanHistory] = useState([]);
// Array of past scan results (max 20 entries)
// Each entry: {id, scanType, input, score, isPhishing, riskLevel, time}
// Displayed in the "Recent Scans" table
// New entries are prepended: [newEntry, ...prev].slice(0, 20)

// Line 59:
const [threatHistory, setThreatHistory] = useState([]);
// Array of data points for the threat history chart (max 15)
// Each entry: {name: "#1", threat: 85, safe: 15}
// Used by the Recharts AreaChart to visualize threat trends
```

## How useState Controls the UI Flow:

```
User Types Message          User Clicks "Scan"          API Responds
      │                           │                          │
      ▼                           ▼                          ▼
setMessage("text")        setLoading(true)           setResult({data})
      │                   setError(null)             setLoading(false)
      ▼                   setResult(null)            addToHistory(...)
  Textarea shows               │                          │
  typed text                   ▼                          ▼
                        Button shows spinner        Result modal opens
                        "Analyzing…"                Threat chart updates
                               │                   History table updates
                               ▼
                     axios.post() → API call
```

## Complete Data Flow Diagram:

```
┌─────────────────────────────────────────────────────┐
│                    FRONTEND (React)                  │
│                                                      │
│  User Input                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ SMS Tab  │    │ URL Tab  │    │FullScan  │       │
│  │ textarea │    │  input   │    │Tab+check │       │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘       │
│       │               │               │              │
│       ▼               ▼               ▼              │
│  analyzeMessage() analyzeUrl()   fullScan()          │
│       │               │               │              │
│       ▼               ▼               ▼              │
│  POST /api/     POST /api/      POST /api/           │
│  analyze        analyze-url     full-scan            │
└───────┬───────────────┬───────────────┬──────────────┘
        │               │               │
        ▼               ▼               ▼
┌───────────────────────────────────────────────────────┐
│                    BACKEND (Flask)                     │
│                                                       │
│  /api/analyze     /api/analyze-url    /api/full-scan  │
│       │                │                    │         │
│       ▼                ▼                    ▼         │
│  model_cache      URLPredictor       SMS + URL +      │
│  .predict()       .predict()         Visual combined  │
│       │                │                    │         │
│   SMS Model        URL Model          All Models      │
│  (Random Forest)  (Random Forest)    + Weights        │
│       │                │                    │         │
│       ▼                ▼                    ▼         │
│  {prediction,     {is_phishing,      {combined_score, │
│   threat_score,    threat_score,       risk_level,    │
│   features}        risk_level}         sms_analysis,  │
│                                        url_analysis}  │
└───────┬───────────────┬───────────────┬──────────────┘
        │               │               │
        ▼               ▼               ▼
┌───────────────────────────────────────────────────────┐
│                   FRONTEND (React)                     │
│                                                       │
│  setResult(data)  →  Result Modal Opens               │
│  addToHistory()   →  Scan History Table Updates       │
│  setThreatHistory → Threat Chart Updates              │
│  setLoading(false)→  Button Returns to Normal         │
└───────────────────────────────────────────────────────┘
```

## Key Frontend Functions Explained:

### `handleAnalyze()` — Lines 154–158
```javascript
const handleAnalyze = () => {
    if (activeTab === "sms") analyzeMessage();      // Call SMS API
    else if (activeTab === "url") analyzeUrl();      // Call URL API
    else fullScan();                                  // Call Full Scan API
};
// This is the SINGLE scan button handler
// It checks which tab is active and calls the appropriate function
```

### `addToHistory()` — Lines 89–108
```javascript
const addToHistory = useCallback((scanType, input, score, isPhishing, riskLevel) => {
    const entry = {
        id: Date.now(),                    // unique ID (timestamp)
        scanType,                           // "sms", "url", or "fullscan"
        input: input.slice(0, 50) + "…",   // truncated input for display
        score: Math.round(score * 100),     // convert 0-1 to 0-100
        isPhishing,                         // true/false
        riskLevel,                          // "High", "Safe", etc.
        time: new Date().toLocaleTimeString(),  // "07:30 AM"
    };
    setScanHistory(prev => [entry, ...prev].slice(0, 20));
    // Prepend new entry, keep only last 20
    // [newScan, ...oldScans].slice(0, 20)

    setThreatHistory(prev => {
        const next = [...prev, {
            name: `#${prev.length + 1}`,    // "#1", "#2", "#3"
            threat: entry.score,             // threat percentage
            safe: 100 - entry.score,         // safe percentage
        }];
        return next.slice(-15);  // Keep only last 15 data points for chart
    });
}, []);
// useCallback = React optimization, prevents recreation on every render
```

### `useEffect` for Health Check — Lines 72–82
```javascript
useEffect(() => {
    const check = async () => {
        try {
            const res = await axios.get(`${API_URL}/api/health`, { timeout: 2000 });
            setApiOnline(res.data.status === "online");
        } catch { setApiOnline(false); }
    };
    check();                              // Check immediately on page load
    const iv = setInterval(check, 3000);  // Then check every 3 seconds
    return () => clearInterval(iv);       // Cleanup when component unmounts
}, []);
// Empty dependency array [] = runs only ONCE when the component first mounts
// This creates a polling loop that checks if the Flask backend is alive
```

---

# ═══════════════════════════════════════════════════════════════
# QUESTION 7: Feature Extraction Code — SMS
# ═══════════════════════════════════════════════════════════════

## SMS Feature Extraction has TWO stages:
1. **Preprocessing** (`src/sms_detection/preprocessing.py`) — cleans the raw text
2. **Feature Extraction** (`src/sms_detection/feature_extraction.py`) — converts cleaned text to numbers

```
SMS Feature Extraction Pipeline:
┌──────────────┐     ┌────────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Raw Message │ ──► │  Preprocessing │ ──► │ Feature Extraction│ ──► │  Numbers    │
│  "URGENT!    │     │  clean_text()  │     │  TF-IDF (500)    │     │  [0.12,     │
│   Click now" │     │  stem/tokenize │     │  + Statistical   │     │   0.45,     │
│              │     │  remove stops  │     │  (19 features)   │     │   0.03 ...] │
└──────────────┘     └────────────────┘     └──────────────────┘     └─────────────┘
                                                                    Total: ~519 numbers
                                                                    Fed to Random Forest
```

---

## Stage 1: SMS Preprocessing — `preprocessing.py`

### Class Initialization (Lines 21–59)
```python
class SMSPreprocessor:
    def __init__(self, use_stemming=True):
        # Load English stopwords (words like "the", "is", "and" that don't carry meaning)
        self.stop_words = set(stopwords.words('english'))

        # Porter Stemmer reduces words to their root form
        # Example: "running" → "run", "working" → "work"
        self.stemmer = PorterStemmer()

        # Pre-compile regex patterns for speed (compiled once, used many times)
        self.url_pattern = re.compile(r'http\S+|www\.\S+|https\S+|\S+\.com|\S+\.org|\S+\.net')
        # Matches: http://anything, www.anything, anything.com/.org/.net

        self.email_pattern = re.compile(r'\S+@\S+')
        # Matches: anything@anything (email addresses)

        self.phone_pattern = re.compile(r'\d{10,}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}')
        # Matches: 10+ digit numbers OR phone formats like 555-123-4567

        self.number_pattern = re.compile(r'\d+')
        # Matches: any sequence of digits
```

### `clean_text()` — Text Cleaning (Lines 61–95)
```python
def clean_text(self, text):
    if pd.isna(text):
        return ""                  # Handle missing values

    text = text.lower()            # "URGENT! Click" → "urgent! click"

    text = self.url_pattern.sub(' URL ', text)
    # Replace all URLs with the word "URL"
    # "click http://evil.com now" → "click URL now"
    # WHY: The actual URL text is noise; what matters is that a URL EXISTS

    text = self.email_pattern.sub(' EMAIL ', text)
    # "send to admin@bank.com" → "send to EMAIL"

    text = self.phone_pattern.sub(' PHONE ', text)
    # "call 555-123-4567" → "call PHONE"

    text = self.number_pattern.sub(' NUMBER ', text)
    # "win $1000" → "win $ NUMBER"

    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove ALL punctuation: "!@#$%^&*()" → ""

    text = ' '.join(text.split())
    # Remove extra whitespace: "hello    world" → "hello world"

    return text
```

**Example**:
```
Input:  "URGENT! Your account #12345 has been suspended. Verify at http://evil.com or call 555-1234"
Step 1: "urgent! your account #12345 has been suspended. verify at http://evil.com or call 555-1234"
Step 2: "urgent! your account #12345 has been suspended. verify at URL or call 555-1234"
Step 3: "urgent! your account #12345 has been suspended. verify at URL or call PHONE"
Step 4: "urgent! your account # NUMBER has been suspended. verify at URL or call PHONE"
Step 5: "urgent your account  NUMBER has been suspended verify at URL or call PHONE"
Step 6: "urgent your account NUMBER has been suspended verify at URL or call PHONE"
```

### `preprocess_text()` — Full Pipeline (Lines 147–170)
```python
def preprocess_text(self, text):
    cleaned = self.clean_text(text)        # Step 1: Clean text
    tokens = self.tokenize_text(cleaned)   # Step 2: Split into words
    # "urgent your account" → ["urgent", "your", "account"]

    tokens = self.remove_stopwords(tokens) # Step 3: Remove common words
    # ["urgent", "your", "account"] → ["urgent", "account"]
    # "your" is a stopword — removed
    # Only words with length > 2 are kept

    tokens = self.stem_or_lemmatize(tokens) # Step 4: Reduce to root form
    # ["urgent", "account"] → ["urgent", "account"]
    # "running" would become "run", "suspended" → "suspend"

    return ' '.join(tokens)
    # Join back: "urgent account"
    # This cleaned string goes into TF-IDF vectorizer
```

### `extract_text_features()` — Statistical Features (Lines 172–222)
```python
def extract_text_features(self, text):
    features = {}
    text_lower = text.lower()

    # ═══ BASIC STATISTICS ═══
    features['message_length'] = len(text)        # Total characters: 85
    features['word_count'] = len(text.split())    # Total words: 15
    features['char_count'] = len(text)            # Same as message_length
    features['avg_word_length'] = np.mean([len(word) for word in text.split()])
    # Average length of each word: (6+4+7+...)/15 = 5.2

    # ═══ CHARACTER COUNTS ═══
    features['special_char_count'] = sum(1 for char in text if char in string.punctuation)
    # Count of !@#$%^&*() etc. — phishing messages often have excessive punctuation

    features['digit_count'] = sum(1 for char in text if char.isdigit())
    # Count of 0-9 — phishing often has phone numbers, account numbers

    features['uppercase_count'] = sum(1 for char in text if char.isupper())
    # Count of A-Z — phishing uses CAPS for urgency: "URGENT!", "ACT NOW!"

    # ═══ RATIOS ═══
    features['uppercase_ratio'] = features['uppercase_count'] / len(text)
    # What percentage of characters are uppercase?
    # Normal message: ~5%, Phishing: often > 30%

    features['digit_ratio'] = features['digit_count'] / len(text)
    # What percentage of characters are digits?

    features['special_char_ratio'] = features['special_char_count'] / len(text)
    # What percentage are special characters?

    # ═══ PATTERN DETECTION (1 = present, 0 = absent) ═══
    features['has_url'] = 1 if bool(self.url_pattern.search(text_lower)) else 0
    # Does the message contain a URL? (Very strong phishing indicator)

    features['has_email'] = 1 if bool(self.email_pattern.search(text_lower)) else 0
    # Does it contain an email address?

    features['has_phone'] = 1 if bool(self.phone_pattern.search(text)) else 0
    # Does it contain a phone number? (scam call-to-action)

    features['has_currency'] = 1 if any(symbol in text for symbol in ['$', '£', '€', 'dollar']) else 0
    # Does it mention money? ($1000 prize, £500 refund)

    # ═══ KEYWORD COUNTS (count how many phishing keywords are present) ═══
    features['urgency_count'] = sum(1 for keyword in URGENCY_KEYWORDS if keyword in text_lower)
    # URGENCY_KEYWORDS = ["urgent", "immediately", "act now", "expires", "hurry", ...]
    # Example: "URGENT! Act now before it expires" → urgency_count = 3

    features['financial_count'] = sum(1 for keyword in FINANCIAL_KEYWORDS if keyword in text_lower)
    # FINANCIAL_KEYWORDS = ["bank", "account", "credit", "payment", "transfer", ...]
    # Example: "Your bank account payment is pending" → financial_count = 3

    features['action_count'] = sum(1 for keyword in ACTION_KEYWORDS if keyword in text_lower)
    # ACTION_KEYWORDS = ["click", "verify", "confirm", "update", "login", ...]

    features['threat_count'] = sum(1 for keyword in THREAT_KEYWORDS if keyword in text_lower)
    # THREAT_KEYWORDS = ["suspended", "blocked", "unauthorized", "locked", ...]

    # ═══ EXCESSIVE PATTERNS ═══
    features['excessive_caps'] = 1 if features['uppercase_ratio'] > 0.3 else 0
    # Flag if more than 30% of text is UPPERCASE — strong phishing indicator

    features['excessive_punctuation'] = 1 if features['special_char_ratio'] > 0.15 else 0
    # Flag if more than 15% are special chars (!!!, ???)

    return features
    # Returns 19 numerical features for this message
```

**All 19 SMS Statistical Features Summary**:

| # | Feature | Type | What It Measures | Phishing Signal |
|---|---------|------|------------------|-----------------|
| 1 | message_length | int | Total characters | Phishing tends to be longer |
| 2 | word_count | int | Total words | More words = more manipulation |
| 3 | avg_word_length | float | Average word size | Shorter simple words in scams |
| 4 | special_char_count | int | Punctuation count | !!! ??? excessive |
| 5 | digit_count | int | Numbers count | Phone numbers, amounts |
| 6 | uppercase_count | int | Capital letters | URGENT, ACT NOW |
| 7 | uppercase_ratio | float | % uppercase | >30% = suspicious |
| 8 | digit_ratio | float | % digits | Many digits = phone/account |
| 9 | special_char_ratio | float | % special chars | >15% = suspicious |
| 10 | has_url | 0/1 | URL present? | Very strong indicator |
| 11 | has_email | 0/1 | Email present? | Contact bait |
| 12 | has_phone | 0/1 | Phone present? | Call-to-action scam |
| 13 | has_currency | 0/1 | Money symbols? | Prize/payment scams |
| 14 | urgency_count | int | Urgency keywords | "urgent", "immediately" |
| 15 | financial_count | int | Money keywords | "bank", "account" |
| 16 | action_count | int | Action keywords | "click", "verify" |
| 17 | threat_count | int | Threat keywords | "suspended", "blocked" |
| 18 | excessive_caps | 0/1 | >30% uppercase? | Final flag |
| 19 | excessive_punctuation | 0/1 | >15% special? | Final flag |

---

## Stage 2: TF-IDF + Feature Combination — `feature_extraction.py`

### Class Initialization (Lines 19–54)
```python
class FeatureExtractor:
    def __init__(self, max_features=500, ngram_range=(1,2)):

        self.tfidf = TfidfVectorizer(
            max_features=500,       # Keep only top 500 most important words
            ngram_range=(1,2),      # Use single words AND word pairs
            # (1,2) means: "urgent" (unigram) AND "act now" (bigram)
            min_df=2,               # Word must appear in at least 2 messages
            max_df=0.95,            # Word must NOT appear in >95% of messages
            # If a word appears in 95%+ messages, it's too common to be useful
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{2,}',  # Words with at least 2 characters
            stop_words=None           # Already removed in preprocessing
        )

        self.scaler = StandardScaler()
        # StandardScaler normalizes numerical features to mean=0, std=1
        # This ensures message_length (range: 10-500) doesn't dominate
        # over uppercase_ratio (range: 0.0-1.0)
```

### `fit_transform()` — Creating Features (Lines 56–116)
```python
def fit_transform(self, df, text_column='processed_text'):
    # ═══ PART 1: TF-IDF Features (500 columns) ═══
    text_data = df[text_column].fillna('').astype(str)  # Handle missing text

    tfidf_features = self.tfidf.fit_transform(text_data)
    # fit_transform() does TWO things:
    #   fit → learns vocabulary (which 500 words are most important)
    #   transform → converts each message to a vector of 500 TF-IDF weights
    #
    # TF-IDF = Term Frequency × Inverse Document Frequency
    #   TF = how often word appears in THIS message
    #   IDF = how rare the word is ACROSS ALL messages
    #   Words that are frequent in this message BUT rare overall get HIGH scores
    #
    # Example for the word "urgent":
    #   TF = 2/15 (appears twice in a 15-word message) = 0.133
    #   IDF = log(5000/50) (appears in only 50 of 5000 messages) = 4.6
    #   TF-IDF = 0.133 × 4.6 = 0.613 (HIGH score — important word)
    #
    # Example for the word "the":
    #   TF = 3/15 = 0.2
    #   IDF = log(5000/4800) = 0.04 (appears in almost all messages)
    #   TF-IDF = 0.2 × 0.04 = 0.008 (LOW score — not useful)

    tfidf_df = pd.DataFrame(
        tfidf_features.toarray(),
        columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    )
    # Converts sparse matrix to DataFrame with column names: tfidf_0, tfidf_1, ..., tfidf_499
    # Each row = one message, each column = TF-IDF weight for one word

    # ═══ PART 2: Statistical Features (19 columns) ═══
    self.numerical_features = [
        'message_length', 'word_count', 'avg_word_length',
        'special_char_count', 'digit_count', 'uppercase_count',
        'uppercase_ratio', 'digit_ratio', 'special_char_ratio',
        'has_url', 'has_email', 'has_phone', 'has_currency',
        'urgency_count', 'financial_count', 'action_count', 'threat_count',
        'excessive_caps', 'excessive_punctuation'
    ]
    # These were extracted by SMSPreprocessor.extract_text_features()

    numerical_data = df[available_features].values
    numerical_scaled = self.scaler.fit_transform(numerical_data)
    # StandardScaler: transforms each feature to mean=0, std=1
    # message_length=150 → might become 0.72 (above average)
    # message_length=30  → might become -1.3 (below average)

    # ═══ PART 3: Combine Both ═══
    features_df = pd.concat([tfidf_df, numerical_df], axis=1)
    # Final shape: (5572, 519)
    # = 5572 messages × (500 TF-IDF words + 19 statistical features)
    # THESE 519 numbers are what the Random Forest model receives

    return features_df
```

### `transform()` — For New Messages at Prediction Time (Lines 118–151)
```python
def transform(self, df, text_column='processed_text'):
    # Used when predicting a NEW message (not training)
    # IMPORTANT: Uses self.tfidf.transform() NOT fit_transform()
    # This means it uses the SAME vocabulary learned during training
    # A new word not in the vocabulary is simply ignored

    tfidf_features = self.tfidf.transform(text_data)  # transform only, no fitting
    numerical_scaled = self.scaler.transform(numerical_data)  # scale using learned params

    features_df = pd.concat([tfidf_df, numerical_df], axis=1)
    return features_df
```

---

# ═══════════════════════════════════════════════════════════════
# QUESTION 8: Feature Extraction Code — URL
# ═══════════════════════════════════════════════════════════════

## URL Feature Extraction — `src/url_detection/url_feature_extractor.py`

**Key Difference from SMS**: URL features are purely **structural** — we analyze the URL structure itself, NOT any text content. No TF-IDF is used.

```
URL Feature Extraction Pipeline:
┌───────────────────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Raw URL                      │ ──► │  URLFeatureExtractor │ ──► │  28 Numbers     │
│  "http://paypal.secure-       │     │  .extract(url)       │     │  [52, 24, 15,   │
│   login.xyz/verify/account"   │     │  Parse + Calculate   │     │   0, 1, 0, 1..] │
└───────────────────────────────┘     └──────────────────────┘     └─────────────────┘
                                                                   Fed to Random Forest
```

### Constant Definitions (Lines 14–37)
```python
# Known URL shortener domains — services that hide the real URL
SHORTENER_DOMAINS = {
    'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly',
    'is.gd', 'buff.ly', 'shorte.st', 'tiny.cc', 'rb.gy',
    'cutt.ly', 'shorturl.at'
}
# WHY: Phishers use URL shorteners to hide their real malicious URL

# Suspicious keywords commonly found in phishing URLs
SUSPICIOUS_WORDS = [
    'login', 'signin', 'verify', 'secure', 'account', 'update',
    'banking', 'confirm', 'password', 'credential', 'suspended',
    'urgent', 'alert'
]
# WHY: Phishing URLs often contain words like "verify", "login", "secure"
# to fool users into thinking the URL is legitimate

# Known brand names used in subdomain spoofing
BRAND_NAMES = [
    'paypal', 'google', 'facebook', 'amazon', 'netflix',
    'apple', 'microsoft', 'sbi', 'hdfc', 'icici'
]
# WHY: Phishers put brand names in subdomains:
# "paypal.evil-site.com" — "paypal" is subdomain, NOT the real site

# IPv4 pattern to detect IP-address based URLs
IPV4_PATTERN = re.compile(
    r'^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$'
)
# Matches: 192.168.1.1, 10.0.0.1, etc.
# WHY: Legitimate sites use domain names (google.com), phishing uses raw IPs
```

### `extract()` Method — Extracting 28 Features (Lines 105–233)
```python
def extract(self, url):
    # ═══ STEP 1: Parse the URL ═══
    url_str = url.strip()
    if not url_str.startswith(('http://', 'https://', 'ftp://')):
        url_str = 'http://' + url_str   # Add scheme if missing

    parsed = urlparse(url_str)
    # urlparse("http://paypal.secure-login.xyz/verify/account?id=1")
    # Returns:
    #   scheme   = "http"
    #   hostname = "paypal.secure-login.xyz"
    #   path     = "/verify/account"
    #   query    = "id=1"

    hostname = (parsed.hostname or '').lower()
    path = parsed.path or ''
    query = parsed.query or ''
    scheme = (parsed.scheme or '').lower()
    url_lower = url.lower()

    # ═══ FEATURE GROUP 1: Length Features (4 features) ═══
    url_length = len(url)              # Total URL length — phishing URLs tend to be longer
    hostname_length = len(hostname)    # Length of just the domain
    path_length = len(path)            # Length of path after domain
    query_length = len(query)          # Length of query parameters

    # ═══ FEATURE GROUP 2: Character Count Features (9 features) ═══
    num_dots = url.count('.')          # "paypal.secure-login.xyz" → 2 dots
    # More dots = more subdomains = suspicious

    num_hyphens = url.count('-')       # "secure-login" has 1 hyphen
    # Phishing loves hyphens: "paypal-secure-login"

    num_underscores = url.count('_')   # Less common but used in phishing paths

    num_slashes = url.count('/')       # Deep paths like /verify/account/step1/confirm
    # More slashes = deeper path = more suspicious

    num_at_signs = url.count('@')      # "@" in URL is very suspicious
    # http://google.com@evil.com redirects to evil.com!

    num_question_marks = url.count('?')  # Query parameters
    num_equals = url.count('=')          # Key-value pairs in query
    num_percent = url.count('%')         # URL-encoded characters (hiding chars)

    num_digits = sum(c.isdigit() for c in url)  # Count all digits
    # Phishing URLs often have random numbers: verify123, id=8472

    # ═══ FEATURE GROUP 3: Structural/Pattern Features (9 features) ═══
    has_ip_address = 1 if IPV4_PATTERN.match(hostname) else 0
    # Is the hostname an IP address like 192.168.1.1?
    # Legitimate sites: google.com    Phishing: 192.168.1.1/login
    # VERY STRONG phishing indicator

    has_https = 1 if scheme == 'https' else 0
    # Does it use HTTPS (secure)?
    # Legitimate sites mostly use HTTPS, phishing often uses HTTP

    has_http_in_domain = 1 if 'http' in hostname else 0
    # Is "http" part of the hostname itself? (deceptive naming)
    # Example: "http-secure-bank.com" ← suspicious

    num_subdomains = max(0, hostname.count('.') - 1)
    # "paypal.secure-login.xyz" → 2 dots - 1 = 1 subdomain
    # "www.google.com" → 2 dots - 1 = 1 subdomain (normal)
    # "login.paypal.secure.evil.com" → 4 dots - 1 = 3 subdomains (suspicious!)

    has_port = 1 if parsed.port is not None else 0
    # Is there a custom port? :8080, :3000
    # Normal sites use default ports (80/443)
    # Phishing might use: http://evil.com:8080/login

    has_double_slash_redirect = 1 if '//' in after_scheme else 0
    # "//" after the scheme can indicate URL redirect attacks
    # http://legit.com//http://evil.com

    domain_has_digits = 1 if any(c.isdigit() for c in hostname) else 0
    # "secure123.com" has digits in domain — unusual for legit sites

    tld_length = len(tld)
    # TLD = Top Level Domain: ".com" (3), ".xyz" (3), ".tk" (2)
    # Suspicious TLDs like .tk, .ml, .ga are often free and used by phishers

    is_shortened = 1 if registered_domain in SHORTENER_DOMAINS else 0
    # Is this a URL shortener? bit.ly, tinyurl.com
    # Shorteners hide the real destination

    # ═══ FEATURE GROUP 4: Keyword Features (2 features) ═══
    has_suspicious_words = 1 if any(word in url_lower for word in SUSPICIOUS_WORDS) else 0
    # Does the URL contain words like "login", "verify", "secure", "banking"?
    # http://paypal.secure-login.xyz/verify ← "secure", "login", "verify" found!

    has_brand_in_subdomain = 0
    subdomains = self._get_subdomains(hostname)
    if subdomains:
        for brand in BRAND_NAMES:
            if brand in subdomains.lower() and brand not in registered_domain.lower():
                has_brand_in_subdomain = 1
                break
    # CRITICAL: Checks if a brand name appears in the SUBDOMAIN but NOT in the real domain
    # "paypal.evil-site.com" → "paypal" is subdomain, real domain is "evil-site.com"
    # This is a classic spoofing technique!
    # "paypal.com" → "paypal" IS the real domain → NOT flagged

    # ═══ FEATURE GROUP 5: Entropy Feature (1 feature) ═══
    hostname_entropy = self._shannon_entropy(hostname)
    # Shannon Entropy measures "randomness" of the hostname
    # Formula: H = -Σ p(x) × log2(p(x))
    #
    # "google.com"         → entropy ≈ 2.5 (low — predictable, few unique chars)
    # "x7k2m9p4q.com"      → entropy ≈ 3.8 (high — very random)
    #
    # Phishing domains are often randomly generated: "a3x8k2m.tk"
    # Legitimate domains are meaningful words: "amazon.com"

    # ═══ FEATURE GROUP 6: Ratio Features (2 features) ═══
    num_letters = sum(c.isalpha() for c in url)
    digit_to_letter_ratio = num_digits / max(num_letters, 1)
    # Ratio of digits to letters in the entire URL
    # High ratio = lots of numbers = suspicious

    num_special = sum(1 for c in url if not c.isalnum())
    special_char_ratio = num_special / max(len(url), 1)
    # Ratio of special characters to total URL length
    # High ratio = lots of unusual characters

    # ═══ FEATURE GROUP 7: Path Depth (1 feature) ═══
    path_segments = [seg for seg in path.split('/') if seg]
    path_depth = len(path_segments)
    # "/verify/account/step1" → path_depth = 3
    # Deeper paths are more suspicious
    # Legitimate: /products  Phishing: /verify/account/secure/update/confirm

    return {
        'url_length': url_length,
        'hostname_length': hostname_length,
        # ... all 28 features as a dictionary
    }
```

### `extract_batch()` — For Training on Many URLs (Lines 235–254)
```python
def extract_batch(self, urls):
    records = []
    for url in urls:
        try:
            features = self.extract(url)    # Extract 28 features for each URL
        except Exception:
            features = {name: 0 for name in self.FEATURE_NAMES}  # Zeros for bad URLs
        records.append(features)

    return pd.DataFrame(records, columns=self.FEATURE_NAMES)
    # Returns a DataFrame with shape (num_urls, 28)
    # Each row = one URL, each column = one feature
    # This DataFrame is fed directly to the Random Forest model
```

### Complete Example Walkthrough:

```
URL: "http://paypal.secure-login.xyz/verify/account?id=8472"

Parsing:
  scheme   = "http"
  hostname = "paypal.secure-login.xyz"
  path     = "/verify/account"
  query    = "id=8472"

Feature Extraction Results:
  ┌─────────────────────────────┬────────┬──────────────────────────────────┐
  │ Feature                     │ Value  │ Why It's Suspicious              │
  ├─────────────────────────────┼────────┼──────────────────────────────────┤
  │ url_length                  │ 52     │ Long URL                         │
  │ hostname_length             │ 24     │ Long hostname                    │
  │ path_length                 │ 16     │ Deep path                        │
  │ query_length                │ 7      │ Has parameters                   │
  │ num_dots                    │ 3      │ Multiple subdomains              │
  │ num_hyphens                 │ 1      │ Hyphen in domain                 │
  │ has_ip_address              │ 0      │ Not an IP (but still phishing)   │
  │ has_https                   │ 0      │ ⚠ No HTTPS — insecure!          │
  │ num_subdomains              │ 1      │ "paypal" is a subdomain          │
  │ has_suspicious_words        │ 1      │ ⚠ "verify" found in URL!        │
  │ has_brand_in_subdomain      │ 1      │ ⚠ "paypal" in subdomain only!   │
  │ hostname_entropy            │ 3.42   │ Somewhat random domain           │
  │ path_depth                  │ 2      │ /verify/account = 2 levels       │
  │ is_shortened                │ 0      │ Not a URL shortener              │
  └─────────────────────────────┴────────┴──────────────────────────────────┘
  
  → Random Forest sees these 28 numbers → Predicts: PHISHING (score: 0.94)
```

### All 28 URL Features Summary:

| # | Feature | Type | What It Measures |
|---|---------|------|------------------|
| 1 | url_length | int | Total URL length |
| 2 | hostname_length | int | Domain name length |
| 3 | path_length | int | Path length after domain |
| 4 | query_length | int | Query string length |
| 5 | num_dots | int | Dot count (subdomains) |
| 6 | num_hyphens | int | Hyphen count |
| 7 | num_underscores | int | Underscore count |
| 8 | num_slashes | int | Slash count (path depth) |
| 9 | num_at_signs | int | @ signs (redirect trick) |
| 10 | num_question_marks | int | Query markers |
| 11 | num_equals | int | Key-value pairs |
| 12 | num_percent | int | URL-encoded chars |
| 13 | num_digits | int | Digit count |
| 14 | has_ip_address | 0/1 | IP instead of domain? |
| 15 | has_https | 0/1 | Secure connection? |
| 16 | has_http_in_domain | 0/1 | "http" in domain name? |
| 17 | num_subdomains | int | Subdomain count |
| 18 | has_port | 0/1 | Custom port? |
| 19 | has_double_slash_redirect | 0/1 | // redirect trick? |
| 20 | domain_has_digits | 0/1 | Digits in domain? |
| 21 | tld_length | int | TLD length (.com=3) |
| 22 | is_shortened | 0/1 | URL shortener? |
| 23 | has_suspicious_words | 0/1 | Phishing keywords? |
| 24 | has_brand_in_subdomain | 0/1 | Brand spoofing? |
| 25 | hostname_entropy | float | Domain randomness |
| 26 | digit_to_letter_ratio | float | Digits vs letters |
| 27 | special_char_ratio | float | Special chars vs total |
| 28 | path_depth | int | URL path depth |

---

## SMS vs URL Feature Extraction — Key Differences

| Aspect | SMS Features | URL Features |
|--------|-------------|--------------|
| **File** | `preprocessing.py` + `feature_extraction.py` | `url_feature_extractor.py` |
| **Input** | Raw text message | Raw URL string |
| **Total Features** | ~519 (500 TF-IDF + 19 statistical) | 28 structural features |
| **Uses TF-IDF?** | ✅ Yes — converts words to numbers | ❌ No — URL has no "words" |
| **Uses NLP?** | ✅ Yes — stemming, stopword removal | ❌ No — purely structural |
| **What it analyzes** | Word meaning and patterns | URL structure and format |
| **Needs preprocessing?** | ✅ Heavy (clean → tokenize → stem) | ❌ Light (just parse URL) |
| **Vocabulary needed?** | ✅ Yes — learned during training | ❌ No — feature formulas are fixed |
| **Feature type** | Mixed (continuous + binary) | Mixed (counts + binary + ratios) |

---

# ═══════════════════════════════════════════════════════════════
# QUICK REFERENCE SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════

| Question | Answer |
|----------|--------|
| Training tool | Python scripts (`.py` files), NOT Jupyter/Colab |
| SMS training file | `src/sms_detection/train_model.py` |
| URL training file | `src/url_detection/train_url_model.py` |
| SMS models (4) | Naive Bayes, Logistic Regression, **Random Forest** ✅, SVM |
| URL models (3-4) | **Random Forest** ✅, Gradient Boosting, Logistic Regression, (XGBoost) |
| Why Random Forest? | Overfitting resistant, non-linear, feature importance, no scaling needed, fast prediction, parallel processing, stable CV |
| SMS feature extraction | `preprocessing.py` (clean/stem/tokenize) + `feature_extraction.py` (TF-IDF 500 + 19 statistical = ~519 features) |
| URL feature extraction | `url_feature_extractor.py` (28 structural features: length, chars, patterns, keywords, entropy, ratios) |
| SMS analysis | Analyzes text content → TF-IDF + features → RF prediction |
| URL analysis | Analyzes URL structure → 28 features → RF prediction |
| Full scan | SMS (40%) + URL (45%) + Visual (15%) weighted combination |
| Frontend state | 12 useState variables controlling UI, inputs, results, history |
