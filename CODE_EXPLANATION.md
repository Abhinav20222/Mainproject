# PhishGuard AI - Complete Code Explanation (Viva Preparation)

> Line-by-line explanation of every important code section

---

## 📑 Table of Contents

1. [Random Forest Selection Code](#1-random-forest-selection-code)
2. [TF-IDF Vectorization Code](#2-tf-idf-vectorization-code)
3. [Frontend to Backend API Connection](#3-frontend-to-backend-api-connection)
4. [Feature Extraction Code](#4-feature-extraction-code)
5. [Threat Score Calculation & Display](#5-threat-score-calculation--display)
6. [Cache Problem Solution](#6-cache-problem-solution)

---

## 1. Random Forest Selection Code

**File:** `src/sms_detection/train_model.py`

### Code That Trains All Models:

```python
# Line 39-56: Initialize all 4 models to compare
self.models = {
    'Naive Bayes': MultinomialNB(),
    # MultinomialNB = Multinomial Naive Bayes classifier
    # Good for text classification, uses word frequencies
    
    'Logistic Regression': LogisticRegression(
        max_iter=1000,       # Maximum 1000 iterations to find best weights
        random_state=random_state,  # Fixed seed for reproducibility
        solver='liblinear'   # Algorithm to find optimal weights
    ),
    
    'Random Forest': RandomForestClassifier(
        n_estimators=100,    # Create 100 decision trees
        random_state=random_state,  # Fixed seed for reproducibility
        n_jobs=-1            # Use all CPU cores for faster training
    ),
    
    'SVM': SVC(
        kernel='linear',     # Use linear boundary (straight line)
        probability=True,    # Enable probability predictions
        random_state=random_state
    )
}
```

### Line-by-Line Explanation:

| Line | Code | Meaning |
|------|------|---------|
| `self.models = {...}` | Creates a dictionary | Stores all 4 models with their names as keys |
| `MultinomialNB()` | Naive Bayes | Probabilistic classifier for text data |
| `max_iter=1000` | Maximum iterations | Allows up to 1000 attempts to find best solution |
| `random_state=random_state` | Random seed | Ensures same results every time you run |
| `n_estimators=100` | 100 trees | Random Forest uses 100 decision trees |
| `n_jobs=-1` | Parallel processing | Uses all CPU cores for speed |
| `kernel='linear'` | Linear SVM | Uses straight line to separate classes |
| `probability=True` | Enable confidence | Returns probability, not just yes/no |

---

### Code That Selects Best Model:

```python
# Line 209-226: Select best model based on F1-Score
def select_best_model(self):
    """Select best model based on F1-score"""
    best_f1 = 0  # Start with 0 as baseline
    
    # Loop through all trained models
    for name, result in self.results.items():
        # If this model's F1-score is higher than current best
        if result['f1_score'] > best_f1:
            best_f1 = result['f1_score']           # Update best score
            self.best_model_name = name            # Save model name
            self.best_model = result['model']      # Save model object
    
    # Print the winner
    print(f"Model: {self.best_model_name}")
    print(f"F1-Score: {best_f1:.4f}")
    
    return self.best_model, self.best_model_name
```

### Line-by-Line Explanation:

| Line | What It Does |
|------|--------------|
| `best_f1 = 0` | Initialize best score to 0 |
| `for name, result in self.results.items()` | Loop through each trained model |
| `if result['f1_score'] > best_f1` | Check if this model is better |
| `best_f1 = result['f1_score']` | Update the best score |
| `self.best_model_name = name` | Remember name of best model |
| `self.best_model = result['model']` | Store the actual model object |

### Why F1-Score?

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Precision = How many predicted phishing were actually phishing?
Recall    = How many actual phishing did we catch?
F1-Score  = Balance between both (best single metric)
```

---

### F1 Score Calculation Code (Complete Explanation)

**File:** `src/sms_detection/train_model.py` (Lines 102-148)

```python
# Line 102-148: Train a single model and calculate all metrics
def train_model(self, name, model):
    """Train a single model and return metrics"""
    
    # Step 1: Train the model on training data
    model.fit(self.X_train, self.y_train)
    # fit() learns patterns from X_train (features) and y_train (labels)
    # X_train = feature vectors (519 features each)
    # y_train = labels (0 = ham, 1 = spam)
    
    # Step 2: Make predictions on test data
    y_pred = model.predict(self.X_test)
    # predict() uses learned patterns to classify test messages
    # y_pred = [0, 1, 1, 0, 1, ...] (predicted labels)
    
    # Step 3: Get probability scores (if available)
    y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    # predict_proba() returns probability for each class
    # [:, 1] gets probability of being spam/phishing
    
    # Step 4: Calculate ACCURACY
    accuracy = accuracy_score(self.y_test, y_pred)
    # accuracy = (correct predictions) / (total predictions)
    # Example: 980 correct out of 1000 = 0.98 (98%)
    
    # Step 5: Calculate PRECISION
    precision = precision_score(self.y_test, y_pred, zero_division=0)
    # precision = TP / (TP + FP)
    # TP = True Positives (correctly identified phishing)
    # FP = False Positives (safe messages wrongly marked as phishing)
    # "Of all messages we said were phishing, how many actually were?"
    
    # Step 6: Calculate RECALL
    recall = recall_score(self.y_test, y_pred, zero_division=0)
    # recall = TP / (TP + FN)
    # TP = True Positives (correctly identified phishing)
    # FN = False Negatives (phishing messages we missed)
    # "Of all actual phishing messages, how many did we catch?"
    
    # Step 7: Calculate F1-SCORE
    f1 = f1_score(self.y_test, y_pred, zero_division=0)
    # F1 = 2 × (precision × recall) / (precision + recall)
    # Harmonic mean of precision and recall
    # Best single metric that balances both
    
    # Step 8: Calculate ROC-AUC (if probabilities available)
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        # Area Under ROC Curve (0.5 = random, 1.0 = perfect)
    else:
        roc_auc = None
    
    # Step 9: Cross-validation for reliability
    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
    # cv=5 means 5-fold cross-validation
    # Trains 5 times on different data splits
    # Returns 5 F1 scores to check consistency
    
    # Print results
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"CV F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Return all metrics
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'cv_scores': cv_scores,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
```

### Visual Explanation with Confusion Matrix:

```
Confusion Matrix Example (1115 test messages):

                        PREDICTED
                    Ham (Safe)    Spam (Phishing)
                  ┌─────────────┬─────────────────┐
ACTUAL  Ham       │     965     │       3         │  ← 968 actual ham
        (Safe)    │     (TN)    │      (FP)       │
                  ├─────────────┼─────────────────┤
ACTUAL  Spam      │      15     │      132        │  ← 147 actual spam
        (Phishing)│     (FN)    │      (TP)       │
                  └─────────────┴─────────────────┘
                       ↓              ↓
                  980 predicted   135 predicted
                     ham            spam


TN (True Negative)  = 965  → Correctly identified safe messages
FP (False Positive) =   3  → Safe messages wrongly marked as phishing
FN (False Negative) =  15  → Phishing messages we missed (DANGEROUS!)
TP (True Positive)  = 132  → Correctly identified phishing messages
```

### Step-by-Step F1 Calculation:

```python
# From the confusion matrix above:
TP = 132    # True Positives
FP = 3      # False Positives  
FN = 15     # False Negatives
TN = 965    # True Negatives

# Step 1: Calculate Precision
precision = TP / (TP + FP)
precision = 132 / (132 + 3)
precision = 132 / 135
precision = 0.9778  # 97.78%
# "Of 135 messages we marked as phishing, 132 actually were"

# Step 2: Calculate Recall
recall = TP / (TP + FN)
recall = 132 / (132 + 15)
recall = 132 / 147
recall = 0.8980  # 89.80%
# "Of 147 actual phishing messages, we caught 132"

# Step 3: Calculate F1-Score
f1 = 2 * (precision * recall) / (precision + recall)
f1 = 2 * (0.9778 * 0.8980) / (0.9778 + 0.8980)
f1 = 2 * (0.8781) / (1.8758)
f1 = 1.7562 / 1.8758
f1 = 0.9363  # 93.63%

# F1-Score = 93.63% (balance between precision and recall)
```

### Why Use F1-Score Instead of Accuracy?

```
Problem with Accuracy alone:

Dataset: 4827 ham + 747 spam = 5574 total
Imbalanced: 86.6% ham, 13.4% spam

If model predicts EVERYTHING as "ham":
- Accuracy = 4827/5574 = 86.6% ✗ (looks good but useless!)
- Precision = 0/0 = 0% (no phishing detected)
- Recall = 0/747 = 0% (missed all phishing)
- F1 = 0% ← Reveals the model is terrible!

F1-Score penalizes models that ignore the minority class (phishing)
```

### Scikit-learn Functions Used:

```python
from sklearn.metrics import (
    accuracy_score,      # (TP + TN) / Total
    precision_score,     # TP / (TP + FP)
    recall_score,        # TP / (TP + FN)
    f1_score,           # 2 × (P × R) / (P + R)
    roc_auc_score,      # Area under ROC curve
    confusion_matrix    # 2x2 matrix of predictions
)

# Usage:
accuracy  = accuracy_score(y_true, y_pred)   # Compare actual vs predicted
precision = precision_score(y_true, y_pred)  
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)

# y_true = [0, 1, 1, 0, 1, ...] (actual labels)
# y_pred = [0, 1, 0, 0, 1, ...] (predicted labels)
```

### Quick Viva Answer for F1-Score:

> **Question:** "How do you calculate F1-Score?"
>
> **Answer:** "F1-Score is the harmonic mean of Precision and Recall:
> `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
>
> - **Precision** = Of all messages predicted as phishing, how many actually were?
> - **Recall** = Of all actual phishing messages, how many did we catch?
> - **F1** = Balances both metrics into a single score.
>
> We use F1 instead of accuracy because our dataset is imbalanced (86% safe, 14% phishing). A model that predicts everything as "safe" would have 86% accuracy but 0% F1-score."

---

## 2. TF-IDF Vectorization Code

**File:** `src/sms_detection/feature_extraction.py`

### What is TF-IDF?

```
TF-IDF = Term Frequency × Inverse Document Frequency

TF  = How often a word appears in THIS message
IDF = How rare is this word across ALL messages

Example:
- "urgent" appears in few messages → HIGH IDF → Important word
- "the" appears in every message → LOW IDF → Ignored
```

### TF-IDF Initialization Code:

```python
# Line 34-44: Configure TF-IDF Vectorizer
self.tfidf = TfidfVectorizer(
    max_features=500,          # Keep only top 500 most important words
    ngram_range=(1, 2),        # Use both single words AND word pairs
    min_df=2,                  # Ignore words appearing in < 2 messages
    max_df=0.95,               # Ignore words appearing in > 95% messages
    lowercase=True,            # Convert all text to lowercase
    strip_accents='unicode',   # Remove accents (é → e)
    analyzer='word',           # Analyze by words (not characters)
    token_pattern=r'\w{2,}',   # Only words with 2+ characters
    stop_words=None            # Don't remove stopwords (already done)
)
```

### Line-by-Line Explanation:

| Parameter | Value | Why? |
|-----------|-------|------|
| `max_features=500` | 500 | Limits vocabulary to 500 most important words to avoid overfitting |
| `ngram_range=(1, 2)` | (1, 2) | Captures "bank" AND "bank account" as separate features |
| `min_df=2` | 2 | Words must appear in at least 2 messages (removes typos) |
| `max_df=0.95` | 0.95 | Ignores words in >95% of messages (like "the", "is") |
| `lowercase=True` | True | "URGENT" and "urgent" become same word |
| `strip_accents='unicode'` | unicode | "résumé" becomes "resume" |
| `token_pattern=r'\w{2,}'` | regex | Only keeps words with 2+ letters |

### TF-IDF Transform Code:

```python
# Line 68-77: Convert text to TF-IDF vectors
print("Extracting TF-IDF features...")

# Convert all text to TF-IDF vectors
tfidf_features = self.tfidf.fit_transform(text_data)
# fit_transform does 2 things:
# 1. fit    = Learn vocabulary from training data
# 2. transform = Convert each message to 500-number vector

print(f"Shape: {tfidf_features.shape}")
# Shape = (5574, 500) means 5574 messages, each with 500 features

# Convert sparse matrix to DataFrame for easier handling
tfidf_df = pd.DataFrame(
    tfidf_features.toarray(),  # Convert to regular array
    columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    # Column names: tfidf_0, tfidf_1, ... tfidf_499
)
```

### Visual Example:

```
Original Message: "URGENT! Click here to verify your bank account"

After TF-IDF (500 numbers):
[0.0, 0.0, 0.45, 0.0, ..., 0.72, ..., 0.38, 0.0]
  ↑    ↑    ↑                ↑         ↑
  the  and  urgent           bank      verify
  (0)  (0)  (HIGH)           (HIGH)    (HIGH)
  
Words like "the" = 0 (too common)
Words like "urgent" = 0.72 (rare, important for phishing)
```

---

## 3. Frontend to Backend API Connection

### Backend: Flask API Endpoints

**File:** `src/api_fast.py`

```python
# Line 21-22: Create Flask app with CORS
app = Flask(__name__)    # Create Flask application
CORS(app)                # Enable Cross-Origin Resource Sharing
                         # (Allows frontend on port 5173 to call backend on port 5000)
```

```python
# Line 215-223: Health Check Endpoint
@app.route('/api/health', methods=['GET'])  # Define URL: /api/health
def health_check():
    """Health check - returns online when models are ready"""
    return jsonify({                         # Return JSON response
        'status': 'online' if models_ready else 'loading',
        'service': 'PhishGuard AI',
        'model': 'SMS Phishing Detector v2.0 (FAST)',
        'models_loaded': models_ready        # True/False
    })
```

```python
# Line 226-257: Analyze Message Endpoint
@app.route('/api/analyze', methods=['POST'])  # POST request to /api/analyze
def analyze_message():
    """Analyze a message for phishing"""
    start_time = time.time()                  # Start timer
    
    try:
        # Check if models are loaded
        if not models_ready:
            return jsonify({'error': 'Models still loading'}), 503
        
        # Get JSON data from request
        data = request.get_json()             # Parse JSON body
        
        # Validate input
        if not data or 'message' not in data:
            return jsonify({
                'error': 'Missing message field',
                'usage': 'POST {"message": "your text here"}'
            }), 400
        
        # Get and clean message
        message = data['message'].strip()     # Remove whitespace
        
        # Make prediction
        result = predict_fast(message)        # Call prediction function
        
        # Add processing time
        result['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)
        
        return jsonify(result)                # Return result as JSON
    
    except Exception as e:
        return jsonify({'error': 'Analysis failed'}), 500
```

### Frontend: React API Calls

**File:** `frontend/src/App.jsx`

```javascript
// Line 5: Define backend URL
const API_URL = "http://localhost:5000";

// Line 22-34: Health Check (runs every 1 second)
useEffect(() => {
  const checkHealth = async () => {
    try {
      // Call GET /api/health
      const res = await axios.get(`${API_URL}/api/health`, { timeout: 2000 });
      
      // Check if status is "online"
      setApiOnline(res.data.status === "online");
    } catch {
      setApiOnline(false);  // If error, mark as offline
    }
  };
  
  checkHealth();                              // Check immediately
  const interval = setInterval(checkHealth, 1000);  // Then every 1 second
  return () => clearInterval(interval);       // Cleanup on unmount
}, []);

// Line 36-51: Analyze Message
const analyzeMessage = async () => {
  if (!message.trim()) return;                // Don't send empty messages
  
  setLoading(true);                           // Show loading spinner
  setError(null);                             // Clear previous errors
  setResult(null);                            // Clear previous result
  
  try {
    // Call POST /api/analyze with message in body
    const res = await axios.post(`${API_URL}/api/analyze`, { message });
    
    setResult(res.data);                      // Store result
  } catch (err) {
    setError(err.response?.data?.error || "Failed to connect");
  } finally {
    setLoading(false);                        // Hide loading spinner
  }
};
```

### Visual Flow:

```
┌─────────────────────────────────────────────────────────────────────┐
│ FRONTEND (React - localhost:5173)                                    │
│                                                                      │
│  User types: "URGENT! Click here to verify your bank"               │
│                        │                                             │
│                        ▼                                             │
│  axios.post("/api/analyze", { message: "URGENT! Click..." })        │
└─────────────────────────────────────────────────────────────────────┘
                         │
                         │ HTTP POST Request
                         │ Content-Type: application/json
                         │ Body: {"message": "URGENT! Click..."}
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BACKEND (Flask - localhost:5000)                                     │
│                                                                      │
│  @app.route('/api/analyze', methods=['POST'])                       │
│  def analyze_message():                                              │
│      data = request.get_json()         # Get message                │
│      result = predict_fast(message)    # Run ML model               │
│      return jsonify(result)            # Return JSON                │
└─────────────────────────────────────────────────────────────────────┘
                         │
                         │ HTTP Response
                         │ Status: 200 OK
                         │ Body: {"is_phishing": true, "threat_score": 85, ...}
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FRONTEND (React)                                                     │
│                                                                      │
│  setResult(res.data)    →    Display threat gauge, features, etc.   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Feature Extraction Code

**File:** `src/api_fast.py` (Lines 107-140)

### Keyword Detection:

```python
# Line 38-46: Define keyword lists
URGENCY_KEYWORDS = ['urgent', 'immediately', 'now', 'asap', 'hurry', 'limited', 
    'expire', 'today', 'fast', 'quick', 'act now', 'limited time']
    
FINANCIAL_KEYWORDS = ['bank', 'account', 'credit', 'debit', 'card', 'money', 'cash', 
    'payment', 'transaction', 'dollar', 'prize', 'won', 'reward', 'refund', 'tax']
    
ACTION_KEYWORDS = ['click', 'call', 'reply', 'confirm', 'verify', 'update', 
    'claim', 'redeem', 'activate', 'download', 'install']
    
THREAT_KEYWORDS = ['suspend', 'block', 'locked', 'unauthorized', 'unusual activity',
    'security alert', 'compromised', 'fraud']
```

### URL, Phone, Email Detection (Regex):

```python
# Line 32-36: Pre-compiled regex patterns for speed
url_pattern = re.compile(r'http\S+|www\.\S+|https\S+|\S+\.com|\S+\.org|\S+\.net')
# Matches: http://..., https://..., www..., anything.com/.org/.net

email_pattern = re.compile(r'\S+@\S+')
# Matches: anything@anything (email addresses)

phone_pattern = re.compile(r'\d{10,}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}')
# Matches: 1234567890 or 123-456-7890 or 123.456.7890

number_pattern = re.compile(r'\d+')
# Matches: any sequence of digits
```

### Feature Extraction Function:

```python
# Line 107-140: Extract all features from a message
def extract_features_fast(text):
    """Fast feature extraction without DataFrame"""
    
    text_lower = text.lower()        # Convert to lowercase
    words = text.split()             # Split into words
    text_len = len(text) if len(text) > 0 else 1  # Avoid division by zero
    
    features = {
        # Basic statistics
        'message_length': len(text),                    # Total characters
        'word_count': len(words),                       # Number of words
        'char_count': len(text),                        # Same as length
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        
        # Special character counts
        'special_char_count': sum(1 for c in text if c in string.punctuation),
        # Count how many characters are !@#$%^&*() etc.
        
        'digit_count': sum(1 for c in text if c.isdigit()),
        # Count how many characters are 0-9
        
        'uppercase_count': sum(1 for c in text if c.isupper()),
        # Count how many letters are UPPERCASE
    }
    
    # Calculate ratios
    features['uppercase_ratio'] = features['uppercase_count'] / text_len
    # What percentage of message is UPPERCASE?
    
    features['digit_ratio'] = features['digit_count'] / text_len
    # What percentage of message is numbers?
    
    features['special_char_ratio'] = features['special_char_count'] / text_len
    # What percentage of message is punctuation?
    
    # Pattern detection (0 = not found, 1 = found)
    features['has_url'] = 1 if url_pattern.search(text_lower) else 0
    features['has_email'] = 1 if email_pattern.search(text_lower) else 0
    features['has_phone'] = 1 if phone_pattern.search(text) else 0
    features['has_currency'] = 1 if any(s in text for s in ['$', '£', '€']) else 0
    
    # Keyword counts
    features['urgency_count'] = sum(1 for k in URGENCY_KEYWORDS if k in text_lower)
    # How many urgency words like "urgent", "now", "asap"?
    
    features['financial_count'] = sum(1 for k in FINANCIAL_KEYWORDS if k in text_lower)
    # How many financial words like "bank", "account", "prize"?
    
    features['action_count'] = sum(1 for k in ACTION_KEYWORDS if k in text_lower)
    # How many action words like "click", "call", "verify"?
    
    features['threat_count'] = sum(1 for k in THREAT_KEYWORDS if k in text_lower)
    # How many threat words like "suspend", "locked", "fraud"?
    
    # Flag excessive patterns
    features['excessive_caps'] = 1 if features['uppercase_ratio'] > 0.3 else 0
    # Is more than 30% of message in CAPS?
    
    features['excessive_punctuation'] = 1 if features['special_char_ratio'] > 0.15 else 0
    # Is more than 15% of message punctuation?
    
    return features
```

### Example Output:

```
Message: "URGENT! Your bank account suspended. Call 1-800-555-1234 NOW!"

Features Extracted:
{
    'message_length': 62,
    'word_count': 8,
    'uppercase_count': 14,           # U,R,G,E,N,T,C,a,l,l,N,O,W,!
    'uppercase_ratio': 0.23,         # 14/62 = 23%
    'has_url': 0,                    # No URL found
    'has_phone': 1,                  # Found: 1-800-555-1234
    'has_currency': 0,               # No $ € £
    'urgency_count': 2,              # "urgent", "now"
    'financial_count': 2,            # "bank", "account"
    'action_count': 1,               # "call"
    'threat_count': 1,               # "suspended"
    'excessive_caps': 0              # 23% < 30% threshold
}
```

---

## 5. Threat Score Calculation & Display

### Backend: Calculate Threat Score

**File:** `src/api_fast.py` (Lines 160-212)

```python
def predict_fast(message):
    """Ultra-fast prediction"""
    
    # Step 1: Extract statistical features
    stat_features = extract_features_fast(message)
    
    # Step 2: Preprocess text for TF-IDF
    processed_text = preprocess_text_fast(message)
    
    # Step 3: Convert to TF-IDF vector
    tfidf_features = feature_extractor.tfidf.transform([processed_text]).toarray()[0]
    # Result: [0.0, 0.45, 0.0, ..., 0.72, ...] (500 numbers)
    
    # Step 4: Scale numerical features
    numerical_values = [stat_features.get(fname, 0) for fname in numerical_feature_names]
    numerical_scaled = feature_extractor.scaler.transform([numerical_values])[0]
    
    # Step 5: Combine all features
    all_features = np.concatenate([tfidf_features, numerical_scaled]).reshape(1, -1)
    # Result: (1, 519) - 500 TF-IDF + 19 statistical features
    
    all_features = np.abs(all_features)  # Make all positive for Naive Bayes
    
    # Step 6: Make prediction
    prediction = int(model.predict(all_features)[0])
    # Result: 0 = safe (ham), 1 = phishing (spam)
    
    # Step 7: Get confidence score
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(all_features)[0]
        # Result: [0.15, 0.85] = 15% safe, 85% phishing
        confidence = float(probabilities[prediction])
        # confidence = 0.85 (85%)
    else:
        confidence = 1.0
    
    # Step 8: Calculate threat score (0-100)
    threat_score = int(confidence * 100) if prediction == 1 else int((1 - confidence) * 100)
    # If phishing with 85% confidence → threat_score = 85
    # If safe with 85% confidence → threat_score = 15 (100 - 85)
    
    # Step 9: Determine threat level
    if threat_score < 30:
        threat_level = 'safe'
    elif threat_score < 60:
        threat_level = 'suspicious'
    elif threat_score < 85:
        threat_level = 'dangerous'
    else:
        threat_level = 'critical'
    
    # Step 10: Return result
    return {
        'message': message,
        'prediction': 'spam' if prediction == 1 else 'ham',
        'is_phishing': bool(prediction == 1),
        'confidence': confidence,
        'threat_score': threat_score,      # 0-100 scale
        'threat_level': threat_level,       # safe/suspicious/dangerous/critical
        'features': {
            'urgency_keywords': stat_features['urgency_count'],
            'financial_keywords': stat_features['financial_count'],
            'has_url': bool(stat_features['has_url']),
            'has_phone': bool(stat_features['has_phone']),
            ...
        }
    }
```

### Frontend: Display Threat Bar

**File:** `frontend/src/App.jsx` (Lines 194-208)

```jsx
{/* Threat Gauge */}
<div className="mb-8">
  {/* Labels above the bar */}
  <div className="flex justify-between text-sm text-gray-400 mb-2">
    <span>Safe</span>
    <span>Suspicious</span>
    <span>Dangerous</span>
    <span>Critical</span>
  </div>
  
  {/* The actual progress bar */}
  <div className="h-4 bg-gray-800 rounded-full overflow-hidden">
    {/* The colored fill */}
    <div
      className={`h-full bg-gradient-to-r ${getThreatColor(result.threat_score).bg} 
                  transition-all duration-1000 ease-out rounded-full 
                  ${getThreatColor(result.threat_score).glow} shadow-lg`}
      style={{ width: `${result.threat_score}%` }}
      {/* width = threat_score percentage (e.g., 85% fills 85% of bar) */}
    />
  </div>
</div>
```

### Color Selection Function:

```jsx
// Line 59-64: Get color based on threat score
const getThreatColor = (score) => {
  if (score < 30) return { 
    bg: "from-green-500 to-emerald-400",    // Green gradient
    text: "text-green-400", 
    glow: "shadow-green-500/50" 
  };
  if (score < 60) return { 
    bg: "from-yellow-500 to-amber-400",     // Yellow gradient
    text: "text-yellow-400", 
    glow: "shadow-yellow-500/50" 
  };
  if (score < 85) return { 
    bg: "from-orange-500 to-red-400",       // Orange gradient
    text: "text-orange-400", 
    glow: "shadow-orange-500/50" 
  };
  return { 
    bg: "from-red-600 to-rose-500",         // Red gradient
    text: "text-red-400", 
    glow: "shadow-red-500/50" 
  };
};
```

### Visual Flow:

```
Threat Score = 85

┌─────────────────────────────────────────────────────────────────────────┐
│  Safe        Suspicious       Dangerous        Critical                 │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │████████████████████████████████████████████████████████████░░░░░░░░││
│  │◄──────────────────── 85% FILLED (RED) ────────────────────►│░░░░░░││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                         │
│  Score: 85   Level: CRITICAL   Color: Red                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Cache Problem Solution

### What Was the Problem?

```
BEFORE (Slow - 3-5 seconds per request):
┌─────────────────────────────────────────────────────────────────────────┐
│ Request 1: Load model from disk → Predict → Return (3000ms)             │
│ Request 2: Load model from disk → Predict → Return (3000ms)             │
│ Request 3: Load model from disk → Predict → Return (3000ms)             │
└─────────────────────────────────────────────────────────────────────────┘
Problem: Loading 2.4MB model file from disk on EVERY request!
```

### The Solution: Singleton Pattern

**File:** `src/model_cache.py`

```python
class ModelCache:
    """Singleton model cache - loads models ONCE and keeps in memory"""
    
    _instance = None       # Single instance (only one exists)
    _initialized = False   # Has it been loaded already?
    
    def __new__(cls):
        # Called when creating new instance
        if cls._instance is None:
            # First time: create new instance
            cls._instance = super().__new__(cls)
        # Return the SAME instance every time
        return cls._instance
    
    def __init__(self):
        # Only run initialization ONCE
        if ModelCache._initialized:
            return  # Already loaded, skip!
        
        print("Loading models...")
        
        # Load model from disk (only once!)
        self.model = joblib.load('data/models/sms_classifier.pkl')
        self.feature_extractor = joblib.load('data/models/feature_extractor.pkl')
        
        # Initialize NLP components
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Mark as initialized
        ModelCache._initialized = True
        self.is_ready = True
        
        print("Models loaded!")

# Create singleton instance when module is imported
model_cache = ModelCache()
```

### How It Works:

```
AFTER (Fast - 50ms per request):
┌─────────────────────────────────────────────────────────────────────────┐
│ Server Start: Load model ONCE → Store in memory (5000ms)                │
│                                                                         │
│ Request 1: Use cached model → Predict → Return (50ms)                   │
│ Request 2: Use cached model → Predict → Return (50ms)                   │
│ Request 3: Use cached model → Predict → Return (50ms)                   │
└─────────────────────────────────────────────────────────────────────────┘
Solution: Load ONCE at startup, reuse for ALL requests!
```

### Singleton Pattern Explained:

```python
# First import/call:
cache = ModelCache()      # Creates NEW instance, loads models (5 sec)

# Second import/call:
cache = ModelCache()      # Returns SAME instance, no loading (0 sec)

# Third import/call:
cache = ModelCache()      # Returns SAME instance, no loading (0 sec)

# All three variables point to the SAME object in memory!
```

### Performance Improvement:

| Metric | Before Cache | After Cache | Improvement |
|--------|--------------|-------------|-------------|
| First request | 3000ms | 5000ms (startup) | N/A |
| Subsequent requests | 3000ms | 50ms | **60x faster** |
| Memory usage | Re-load each time | One copy in RAM | **Efficient** |
| Model file access | Every request | Only once | **Less I/O** |

### Simple Explanation for Viva:

> **Question:** "How did you solve the cache problem?"
>
> **Answer:** "We implemented a **Singleton Pattern** for model caching. The ML model (2.4MB) is loaded from disk **only once** when the server starts. Then it stays in memory (RAM) and is reused for all API requests. This changed response time from 3-5 seconds to just 50 milliseconds - a **60x improvement**."

---

## 📝 Quick Reference for Viva

| Topic | Key Point |
|-------|-----------|
| **Random Forest Selection** | Compares 4 models, picks highest F1-score |
| **TF-IDF** | Converts text to 500 numbers based on word importance |
| **API Connection** | React calls Flask via axios, Flask returns JSON |
| **Feature Extraction** | Regex for URL/phone, keyword counting, ratios |
| **Threat Score** | `confidence × 100` for phishing, `(1-confidence) × 100` for safe |
| **Cache Solution** | Singleton pattern loads model once, reuses for all requests |

---

*Created for Viva Preparation - January 2026*
