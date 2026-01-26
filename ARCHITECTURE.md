# PhishGuard AI - System Architecture

> **A Multi-Layered Phishing Detection System using Machine Learning**

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE (React)                               │
│                     frontend/src/App.jsx                                     │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ HTTP Requests (axios)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FLASK REST API                                      │
│                       src/api_fast.py                                        │
│                                                                              │
│   Endpoints:                                                                 │
│   • GET  /api/health  → Check if models are loaded                          │
│   • POST /api/analyze → Analyze message for phishing                        │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│  SMS Detection      │   │   URL Detection     │   │  Visual Forensics   │
│  Module             │   │   Module            │   │  Module             │
│                     │   │                     │   │                     │
│  src/sms_detection/ │   │  src/url_detection/ │   │  src/dashboard/     │
│  ✅ IMPLEMENTED     │   │  ⏳ PLANNED         │   │  ⏳ PLANNED         │
└──────────┬──────────┘   └─────────────────────┘   └─────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ML MODELS (Naive Bayes, Random Forest)                    │
│                                                                              │
│   Training: src/sms_detection/train_model.py                                 │
│   Prediction: src/model_cache.py (Singleton Pattern)                        │
│                                                                              │
│   Models Compared:                                                           │
│   • Naive Bayes           • Random Forest (100 trees)                        │
│   • Logistic Regression   • SVM (Linear Kernel)                              │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LOCAL STORAGE (Preprocessed Data, Models)                 │
│                                                                              │
│   data/models/                                                               │
│   ├── sms_classifier.pkl      → Trained ML model                            │
│   ├── feature_extractor.pkl   → TF-IDF vectorizer + scaler                  │
│   └── sms_model_info.pkl      → Model metadata                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
phishing_detection/
├── frontend/                          # React Frontend
│   ├── src/
│   │   ├── App.jsx                   # Main application component
│   │   ├── index.css                 # Tailwind CSS styles
│   │   └── main.jsx                  # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── src/                               # Python Backend
│   ├── api_fast.py                   # Flask REST API (main server)
│   ├── config.py                     # Configuration constants
│   ├── model_cache.py                # Singleton model loader
│   │
│   ├── sms_detection/                # SMS Detection Module
│   │   ├── preprocessing.py          # Text cleaning & tokenization
│   │   ├── feature_extraction.py     # TF-IDF & statistical features
│   │   ├── train_model.py            # Model training pipeline
│   │   ├── predict.py                # Standard prediction
│   │   └── predict_fast.py           # Optimized fast prediction
│   │
│   ├── url_detection/                # URL Detection Module (planned)
│   └── dashboard/                    # Visual Forensics (planned)
│
├── data/
│   ├── raw/                          # Raw SMS dataset
│   ├── processed/                    # Preprocessed data
│   └── models/                       # Trained models (.pkl files)
│
├── reports/                          # Generated visualizations
│   ├── model_comparison.png
│   ├── confusion_matrices_all.png
│   ├── roc_curves.png
│   └── classification_report.txt
│
└── requirements.txt                  # Python dependencies
```

---

## 🔄 Data Flow

### 1. User Input → Prediction

```
User types message
        │
        ▼
┌───────────────────┐
│  React Frontend   │  ← Captures user input
│  (App.jsx)        │
└────────┬──────────┘
         │ POST /api/analyze { message: "..." }
         ▼
┌───────────────────┐
│  Flask API        │  ← Receives HTTP request
│  (api_fast.py)    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Text Preprocessing│ ← Clean, tokenize, stem
│  - Remove URLs     │
│  - Remove numbers  │
│  - Stemming        │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Feature Extraction │ ← Extract 500+ features
│  - TF-IDF (500)    │
│  - Statistical (19)│
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  ML Model Predict  │ ← Random Forest / Naive Bayes
│  - predict()       │
│  - predict_proba() │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  JSON Response     │
│  {                 │
│    is_phishing,    │
│    threat_score,   │
│    confidence,     │
│    features        │
│  }                 │
└───────────────────┘
```

---

## 🧠 Component Details

### 1. User Interface (React)

**Location:** `frontend/src/App.jsx`

| Feature | Description |
|---------|-------------|
| Message Input | Textarea for pasting suspicious messages |
| Sample Messages | Pre-loaded test samples (safe + phishing) |
| Health Indicator | Shows AI online/offline status (polls every 1s) |
| Threat Gauge | Animated progress bar showing risk score |
| Feature Cards | Visual breakdown of threat indicators |

**Technologies:**
- React 18 with Hooks (useState, useEffect)
- Vite for build tooling
- TailwindCSS for styling
- Axios for HTTP requests
- Lucide React for icons

---

### 2. Flask REST API

**Location:** `src/api_fast.py`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Returns `{status: "online"}` when models loaded |
| `/api/analyze` | POST | Analyzes message, returns prediction |

**Key Features:**
- CORS enabled for cross-origin requests
- Background model loading for fast startup (~5 seconds)
- Thread-safe singleton model cache
- Response time: ~50ms per prediction

**Response Schema:**
```json
{
  "message": "original message",
  "prediction": "spam" | "ham",
  "is_phishing": true | false,
  "confidence": 0.95,
  "threat_score": 85,
  "threat_level": "safe" | "suspicious" | "dangerous" | "critical",
  "features": {
    "urgency_keywords": 2,
    "financial_keywords": 1,
    "has_url": true,
    "has_phone": false,
    ...
  },
  "processing_time_ms": 48.5
}
```

---

### 3. SMS Detection Module

**Location:** `src/sms_detection/`

#### Preprocessing Pipeline (`preprocessing.py`)

```
Raw Text
    │
    ▼ lowercase()
    │
    ▼ Remove URLs → Replace with "URL"
    │
    ▼ Remove Emails → Replace with "EMAIL"
    │
    ▼ Remove Phone Numbers → Replace with "PHONE"
    │
    ▼ Remove Other Numbers → Replace with "NUMBER"
    │
    ▼ Remove Punctuation
    │
    ▼ Tokenize (word_tokenize)
    │
    ▼ Remove Stopwords (NLTK English)
    │
    ▼ Apply Stemming (PorterStemmer)
    │
    ▼ Join tokens
    │
Processed Text
```

#### Feature Extraction (`feature_extraction.py`)

**TF-IDF Features (500 features):**
```python
TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),    # Unigrams + Bigrams
    min_df=2,              # Ignore rare terms
    max_df=0.95            # Ignore very common terms
)
```

**Statistical Features (19 features):**

| Feature | Description |
|---------|-------------|
| `message_length` | Total character count |
| `word_count` | Number of words |
| `avg_word_length` | Average word length |
| `special_char_count` | Punctuation count |
| `digit_count` | Number of digits |
| `uppercase_count` | Number of uppercase letters |
| `uppercase_ratio` | Uppercase / total chars |
| `digit_ratio` | Digits / total chars |
| `special_char_ratio` | Special chars / total chars |
| `has_url` | Contains URL (1/0) |
| `has_email` | Contains email (1/0) |
| `has_phone` | Contains phone number (1/0) |
| `has_currency` | Contains $, £, € (1/0) |
| `urgency_count` | Count of urgency keywords |
| `financial_count` | Count of financial keywords |
| `action_count` | Count of action keywords |
| `threat_count` | Count of threat keywords |
| `excessive_caps` | >30% uppercase (1/0) |
| `excessive_punctuation` | >15% special chars (1/0) |

---

### 4. ML Models

**Location:** `src/sms_detection/train_model.py`

#### Models Trained & Compared

| Model | Description | Parameters |
|-------|-------------|------------|
| **Naive Bayes** | Probabilistic classifier | MultinomialNB |
| **Logistic Regression** | Linear classifier | max_iter=1000, solver='liblinear' |
| **Random Forest** | Ensemble of decision trees | n_estimators=100 |
| **SVM** | Support Vector Machine | kernel='linear', probability=True |

#### Model Selection Criteria
- Primary: **F1-Score** (balance of precision and recall)
- Secondary: ROC-AUC, Cross-validation scores

#### Evaluation Metrics Generated
- Confusion matrices for all models
- ROC curves with AUC scores
- Classification report (precision, recall, F1)
- Model comparison bar charts

---

#### 🧠 Algorithms Explained in Simple Words

##### 1. Random Forest (Primary Algorithm - Best Performance)

**Simple Explanation:**
Imagine you're trying to decide if a message is phishing. Instead of asking just one person, you ask **100 different experts** (decision trees). Each expert looks at the message and votes "phishing" or "safe." The final answer is whatever **most experts voted for**.

**How It Works:**
```
Message: "URGENT! Your bank account suspended. Click here!"
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
      ┌─────────┐        ┌─────────┐        ┌─────────┐
      │ Tree 1  │        │ Tree 2  │        │ Tree 3  │   ... (100 trees)
      │ 🌲      │        │ 🌲      │        │ 🌲      │
      └────┬────┘        └────┬────┘        └────┬────┘
           │                  │                  │
           ▼                  ▼                  ▼
       PHISHING            PHISHING           PHISHING
           │                  │                  │
           └──────────────────┴──────────────────┘
                              │
                              ▼
                    FINAL VOTE: PHISHING ✓
                    (Majority wins)
```

**Why It's Good:**
- Multiple "opinions" reduce mistakes
- Works well with many features (we have 519!)
- Can identify which features are most important
- Handles both text and numerical data

**Real-World Analogy:**
> Like a **jury of 100 people** deciding if someone is guilty. One person might make a mistake, but if 95 out of 100 say "guilty," it's probably correct.

---

##### 2. Naive Bayes (Fast & Simple)

**Simple Explanation:**
This algorithm calculates the **probability** of a message being phishing based on the words it contains. It's "naive" because it assumes each word is independent (which isn't always true, but works surprisingly well!).

**How It Works:**
```
Step 1: Learn from training data
        - "urgent" appears in 80% of phishing messages
        - "click" appears in 70% of phishing messages
        - "meeting" appears in 90% of safe messages

Step 2: For new message "URGENT! Click here to verify"
        
        P(Phishing) = P(urgent|phishing) × P(click|phishing) × P(verify|phishing)
                    = 0.80 × 0.70 × 0.60
                    = 0.336 (33.6%)

        P(Safe)     = P(urgent|safe) × P(click|safe) × P(verify|safe)
                    = 0.05 × 0.10 × 0.15
                    = 0.00075 (0.075%)

Step 3: Compare probabilities
        Phishing probability >> Safe probability
        
        RESULT: PHISHING ✓
```

**Why It's Good:**
- Very fast (works in milliseconds)
- Works great with text data
- Needs less training data
- Easy to understand and explain

**Real-World Analogy:**
> Like a **spam filter in your email**. It learns that words like "lottery," "winner," and "urgent" usually appear in spam, and words like "meeting" and "project" appear in normal emails.

---

##### 3. Logistic Regression (Linear Classifier)

**Simple Explanation:**
This algorithm draws a **line (or boundary)** between phishing and safe messages. It assigns a "weight" to each feature. If the total weighted score is above a threshold, it's phishing.

**How It Works:**
```
Features of a message:
┌────────────────────────────────────────────────────┐
│ has_url = 1         × weight 0.8  =  0.8          │
│ urgency_keywords = 2 × weight 0.6  =  1.2          │
│ has_phone = 1       × weight 0.4  =  0.4          │
│ message_length = 50 × weight 0.01 =  0.5          │
└────────────────────────────────────────────────────┘
                              │
                              ▼
                        Total Score = 2.9
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Score > 0.5?       │
                   │  2.9 > 0.5 = YES    │
                   └──────────┬──────────┘
                              │
                              ▼
                         PHISHING ✓
```

**Why It's Good:**
- Fast and efficient
- Shows exactly which features matter most
- Outputs probability (not just yes/no)
- Works well when features have linear relationship

**Real-World Analogy:**
> Like a **credit score system**. Each factor (income, debt, history) has a weight. Add them up, and if the total is above a threshold, you get approved.

---

##### 4. SVM - Support Vector Machine (Maximum Margin)

**Simple Explanation:**
SVM finds the **best possible line** that separates phishing from safe messages. It doesn't just find any line—it finds the one that has the **maximum distance** from both types.

**How It Works:**
```
          SAFE MESSAGES                    PHISHING MESSAGES
               ○                                  ×
            ○     ○                            ×     ×
         ○    ○      ○                      ×    ×     ×
           ○     ○           ║              ×       ×
              ○              ║  ← Maximum      ×    ×
            ○    ○           ║    Margin        ×
                             ║                    ×
         ○                   ║                      ×
              ○              ║
                        
        The line (║) is positioned to maximize
        the distance from both ○ and ×
```

**Why It's Good:**
- Finds the optimal separation boundary
- Works well in high-dimensional spaces (519 features)
- Effective when classes are clearly separable
- Less prone to overfitting

**Real-World Analogy:**
> Like drawing a **fence between two groups of animals**. You want the fence to be as far as possible from both groups, so if an animal gets close to the fence, you can still correctly identify which group it belongs to.

---

#### 📊 Algorithm Comparison Summary

| Algorithm | Speed | Accuracy | Best For | How It Decides |
|-----------|-------|----------|----------|----------------|
| **Random Forest** | Medium | Highest ✓ | Complex patterns | Voting by 100 trees |
| **Naive Bayes** | Fastest | Good | Text classification | Word probabilities |
| **Logistic Regression** | Fast | Good | Linear relationships | Weighted score |
| **SVM** | Slow | High | Clear separation | Maximum margin line |

---

#### 🏆 Why Random Forest Was Selected

Our system compared all 4 algorithms and selected **Random Forest** because:

1. **Highest F1-Score** (98.39% accuracy)
2. **Handles 519 features** effectively
3. **Robust to noise** - doesn't get confused by unusual messages
4. **Feature importance** - tells us which features matter most
5. **No overfitting** - generalizes well to new messages

```
Model Comparison Results:
┌────────────────────┬──────────┬───────────┬────────┬──────────┐
│ Model              │ Accuracy │ Precision │ Recall │ F1-Score │
├────────────────────┼──────────┼───────────┼────────┼──────────┤
│ Random Forest      │  98.39%  │   98.23%  │ 96.68% │  97.45%  │ ← WINNER
│ Logistic Regression│  97.84%  │   97.12%  │ 95.34% │  96.22%  │
│ SVM                │  97.53%  │   96.89%  │ 94.87% │  95.87%  │
│ Naive Bayes        │  96.77%  │   95.45%  │ 93.21% │  94.31%  │
└────────────────────┴──────────┴───────────┴────────┴──────────┘
```

---

### 5. Keyword Detection

**Location:** `src/config.py`

```python
URGENCY_KEYWORDS = [
    'urgent', 'immediately', 'now', 'asap', 'hurry', 'limited',
    'expire', 'today', 'fast', 'quick', 'act now', 'limited time'
]

FINANCIAL_KEYWORDS = [
    'bank', 'account', 'credit', 'debit', 'card', 'money', 'cash',
    'payment', 'transaction', 'dollar', 'prize', 'won', 'reward',
    'refund', 'tax', 'irs', 'paypal'
]

ACTION_KEYWORDS = [
    'click', 'call', 'reply', 'confirm', 'verify', 'update',
    'claim', 'redeem', 'activate', 'download', 'install'
]

THREAT_KEYWORDS = [
    'suspend', 'block', 'locked', 'unauthorized', 'unusual activity',
    'security alert', 'compromised', 'fraud'
]
```

---

### 6. Threat Level Classification

| Score Range | Level | Color | Description |
|-------------|-------|-------|-------------|
| 0 - 29 | ✅ Safe | Green | Legitimate message |
| 30 - 59 | ⚠️ Suspicious | Yellow | Some warning signs |
| 60 - 84 | 🔶 Dangerous | Orange | High risk of phishing |
| 85 - 100 | 🔴 Critical | Red | Definite phishing attempt |

---

### 7. Local Storage (Preprocessed Data, Models)

**Location:** `data/` directory

The Local Storage layer is the **persistence layer** of the system. It stores all preprocessed datasets and trained machine learning models, ensuring that the system doesn't need to retrain models or reprocess data on every startup.

#### 📁 Directory Structure

```
data/
├── raw/                              # Original, unprocessed datasets
│   └── sms_spam.csv                  # Raw SMS dataset (5,574 messages)
│
├── processed/                        # Preprocessed, cleaned datasets
│   └── sms_processed.csv             # Cleaned text + extracted features
│
└── models/                           # Trained ML models (serialized)
    ├── sms_classifier.pkl            # Trained ML model (~2.4 MB)
    ├── feature_extractor.pkl         # TF-IDF + Scaler (~460 KB)
    └── sms_model_info.pkl            # Model metadata (~243 bytes)
```

---

#### 📄 Detailed File Descriptions

##### 1. Raw Data (`data/raw/sms_spam.csv`)

| Property | Value |
|----------|-------|
| **Purpose** | Store original SMS dataset before any processing |
| **Format** | CSV (Comma-Separated Values) |
| **Columns** | `label` (ham/spam), `message` (raw text) |
| **Records** | 5,574 SMS messages |
| **Source** | UCI Machine Learning Repository - SMS Spam Collection |

**Sample Data:**
```csv
label,message
ham,"How are you doing today?"
spam,"URGENT! You've won $1000. Call NOW to claim!"
ham,"See you at the meeting tomorrow."
spam,"Your account has been suspended. Verify at bit.ly/xyz"
```

---

##### 2. Processed Data (`data/processed/sms_processed.csv`)

| Property | Value |
|----------|-------|
| **Purpose** | Store cleaned, tokenized text with extracted features |
| **Format** | CSV |
| **Columns** | 25+ columns (original + processed text + all features) |
| **Created By** | `src/sms_detection/preprocessing.py` |

**Columns Added During Preprocessing:**

| Column | Description |
|--------|-------------|
| `cleaned_text` | Text after removing URLs, emails, numbers |
| `processed_text` | Final tokenized + stemmed text for TF-IDF |
| `message_length` | Character count of original message |
| `word_count` | Number of words |
| `has_url` | 1 if contains URL, 0 otherwise |
| `has_phone` | 1 if contains phone number, 0 otherwise |
| `urgency_count` | Count of urgency keywords found |
| `financial_count` | Count of financial keywords found |
| `label_encoded` | 0 = ham (safe), 1 = spam (phishing) |

**Why Store Processed Data?**
- Avoids re-processing 5,574 messages on each training run
- Speeds up model experimentation
- Allows analysis of feature distributions

---

##### 3. SMS Classifier Model (`data/models/sms_classifier.pkl`)

| Property | Value |
|----------|-------|
| **Purpose** | Store the trained machine learning model |
| **Format** | Pickle file (serialized Python object) |
| **Size** | ~2.4 MB |
| **Library** | `joblib` (optimized pickle for NumPy arrays) |
| **Contains** | Trained Random Forest / Naive Bayes classifier |
| **Created By** | `src/sms_detection/train_model.py` |

**What's Inside the Model?**
```python
# The model contains:
# - Decision tree structures (if Random Forest)
# - Feature importances
# - Class probabilities
# - Trained parameters (weights, splits, thresholds)
```

**How It's Saved:**
```python
import joblib
joblib.dump(trained_model, 'data/models/sms_classifier.pkl')
```

**How It's Loaded:**
```python
import joblib
model = joblib.load('data/models/sms_classifier.pkl')
prediction = model.predict(features)
```

---

##### 4. Feature Extractor (`data/models/feature_extractor.pkl`)

| Property | Value |
|----------|-------|
| **Purpose** | Store TF-IDF vectorizer and feature scaler |
| **Format** | Pickle file |
| **Size** | ~460 KB |
| **Created By** | `src/sms_detection/feature_extraction.py` |

**What's Inside?**

```python
class FeatureExtractor:
    tfidf              # TfidfVectorizer - converts text to TF-IDF vectors
    scaler             # StandardScaler - normalizes numerical features
    feature_names      # List of all feature names
    numerical_features # List of statistical feature names
```

**Why Is This Needed?**
- The TF-IDF vectorizer must use the **same vocabulary** as training
- The scaler must use the **same mean/std** values as training
- Without this, predictions would be inconsistent

**TF-IDF Vocabulary Example:**
```python
# The vectorizer learns these word → index mappings during training:
{
    'urgent': 0,
    'bank': 1,
    'account': 2,
    'verify': 3,
    'click': 4,
    ...  # 500 total features
}
```

---

##### 5. Model Info (`data/models/sms_model_info.pkl`)

| Property | Value |
|----------|-------|
| **Purpose** | Store metadata about the trained model |
| **Format** | Pickle file (dictionary) |
| **Size** | ~243 bytes |
| **Created By** | `src/sms_detection/train_model.py` |

**Contents:**
```python
{
    'name': 'Random Forest',           # Selected model name
    'accuracy': 0.9839,                # Test accuracy
    'f1_score': 0.9745,                # F1 score
    'precision': 0.9823,               # Precision
    'recall': 0.9668                   # Recall
}
```

---

#### 🔄 Data Flow in Local Storage

```
                    TRAINING PHASE
                    ═══════════════
┌─────────────────┐
│  data/raw/      │
│  sms_spam.csv   │ ────────────────────────────────────────┐
└─────────────────┘                                          │
        │                                                    │
        ▼ preprocessing.py                                   │
┌─────────────────┐                                          │
│  data/processed/│                                          │
│  sms_processed  │ ──────┐                                  │
└─────────────────┘       │                                  │
                          │                                  │
                          ▼ feature_extraction.py            │
                    ┌─────────────────┐                      │
                    │ Extract TF-IDF  │                      │
                    │ + Statistics    │                      │
                    └────────┬────────┘                      │
                             │                               │
                             ▼ train_model.py                │
                    ┌─────────────────┐                      │
                    │ Train Models    │                      │
                    │ Select Best     │                      │
                    └────────┬────────┘                      │
                             │                               │
        ┌────────────────────┼────────────────────┐          │
        ▼                    ▼                    ▼          │
┌───────────────┐  ┌────────────────┐  ┌───────────────┐     │
│sms_classifier │  │feature_extractor│  │sms_model_info │     │
│    .pkl       │  │     .pkl       │  │    .pkl       │     │
└───────────────┘  └────────────────┘  └───────────────┘     │
        │                    │                    │          │
        └────────────────────┴────────────────────┘          │
                             │                               │
                             ▼                               │
                    PREDICTION PHASE                         │
                    ════════════════                         │
                             │                               │
┌─────────────────┐          │                               │
│  model_cache.py │◄─────────┘                               │
│  (Loads once)   │                                          │
└────────┬────────┘                                          │
         │                                                   │
         ▼                                                   │
┌─────────────────┐                                          │
│  api_fast.py    │  ← Uses cached models for predictions    │
│  (Flask API)    │                                          │
└─────────────────┘                                          │
```

---

#### 💾 Serialization Technology: Joblib

**Why Joblib Instead of Pickle?**

| Feature | Pickle | Joblib |
|---------|--------|--------|
| NumPy array handling | Slow | Optimized (fast) |
| File size | Larger | Compressed |
| Memory efficiency | Standard | Better for large arrays |
| Scikit-learn support | Manual | Native integration |

**Code Example:**
```python
import joblib

# Saving (during training)
joblib.dump(model, 'data/models/sms_classifier.pkl')
joblib.dump(feature_extractor, 'data/models/feature_extractor.pkl')

# Loading (during prediction)
model = joblib.load('data/models/sms_classifier.pkl')
feature_extractor = joblib.load('data/models/feature_extractor.pkl')
```

---

#### 🔐 Model Cache (Singleton Pattern)

**Location:** `src/model_cache.py`

To avoid loading models from disk on every API request, we use a **Singleton Pattern**:

```python
class ModelCache:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ModelCache._initialized:
            return  # Already loaded, skip
        
        # Load from Local Storage (only once)
        self.model = joblib.load('data/models/sms_classifier.pkl')
        self.feature_extractor = joblib.load('data/models/feature_extractor.pkl')
        
        ModelCache._initialized = True

# Usage: Models loaded once when module is imported
model_cache = ModelCache()
```

**Benefits:**
- Models loaded only **once** at startup
- All API requests share the same model instance
- Reduces memory usage
- Prediction time: ~50ms (instead of 3-5 seconds if reloading)

---

## 🚀 How to Run

### Backend (Flask API)
```bash
cd phishing_detection
python -m src.api_fast
```
Server starts at: `http://localhost:5000`

### Frontend (React)
```bash
cd frontend
npm install
npm run dev
```
App opens at: `http://localhost:5173`

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 98.39% |
| **Model Startup Time** | ~5 seconds |
| **Prediction Time** | ~50ms per message |
| **Training Dataset** | 5,574 SMS messages |
| **Feature Vector Size** | 519 features |

---

## 🔮 Future Enhancements

1. **URL Detection Module** - Analyze suspicious URLs for phishing patterns
2. **Visual Forensics Module** - Screenshot analysis of phishing websites
3. **Email Detection** - Extend to email phishing detection
4. **Real-time Threat Intelligence** - Integration with threat databases
5. **Browser Extension** - Detect phishing in real-time while browsing

---

## 📚 Technologies Used

| Layer | Technology |
|-------|------------|
| Frontend | React, Vite, TailwindCSS, Axios |
| Backend | Flask, Flask-CORS |
| ML | scikit-learn, NLTK, NumPy, Pandas |
| Storage | Joblib (pickle), CSV |
| Visualization | Matplotlib, Seaborn |

---

*Last Updated: January 2026*
