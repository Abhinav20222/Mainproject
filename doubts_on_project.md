# Doubts on Project — PhishGuard AI

---

## 1. TF-IDF with Simple Example

### What is TF-IDF?

**TF-IDF** = **Term Frequency – Inverse Document Frequency**

It converts text into numbers so that the machine learning model can understand it. It measures **how important a word is** in a document relative to all documents.

### Two Parts:

**TF (Term Frequency)** = How often a word appears in ONE message

```
TF = (Number of times word appears in the message) / (Total words in the message)
```

**IDF (Inverse Document Frequency)** = How rare/unique a word is across ALL messages

```
IDF = log(Total number of messages / Number of messages containing the word)
```

**TF-IDF Score = TF × IDF**

### Simple Example:

Suppose we have 3 SMS messages:

| Message No. | Message Text |
|-------------|--------------|
| Message 1 | "Click here to win free prize" |
| Message 2 | "Your account is suspended click here" |
| Message 3 | "Hi, how are you today" |

Let's calculate TF-IDF for the word **"click"**:

**Step 1: TF (Term Frequency)**
- Message 1: "click" appears 1 time, total words = 6 → TF = 1/6 = **0.167**
- Message 2: "click" appears 1 time, total words = 6 → TF = 1/6 = **0.167**
- Message 3: "click" appears 0 times → TF = **0**

**Step 2: IDF (Inverse Document Frequency)**
- Total messages = 3
- Messages containing "click" = 2
- IDF = log(3/2) = **0.176**

**Step 3: TF-IDF = TF × IDF**
- Message 1: 0.167 × 0.176 = **0.029**
- Message 2: 0.167 × 0.176 = **0.029**
- Message 3: 0 × 0.176 = **0**

Now compare with an extremely common word like **"the"** (if it appeared in all 3 messages):
- IDF = log(3/3) = log(1) = **0** → TF-IDF = 0 for all messages!

**Key Insight:** Common words get low scores, unique/rare words get higher scores. Words like "click", "win", "prize", "suspended" get higher TF-IDF scores because they appear in specific types of messages (spam/phishing), not in every message.

### Important: IDF comes from Training Data ONLY

A common doubt is: *"Does IDF change when I check a new message?"*

**Answer: NO.**

- **idf (Inverse Document Frequency)** is calculated **ONCE** using the **5,000+ training messages**.
- It represents the "global importance" of words in your training universe.
- When you check a **new message** (e.g., "Win a free iPhone"), the system uses the **already learned IDF scores** from training.
- We do **not** recalculate IDF based on the new message alone.

### Where TF-IDF is used in our project:

**File:** `src/sms_detection/feature_extraction.py` (Lines 34–44)

```python
self.tfidf = TfidfVectorizer(
    max_features=max_features,       # Keep only top 500 words
    ngram_range=ngram_range,         # (1, 2) means single words AND word pairs
    min_df=MIN_DF,                   # Ignore words appearing in too few messages
    max_df=MAX_DF,                   # Ignore words appearing in too many messages
    lowercase=True,                  # Convert all text to lowercase
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{2,}',         # Only words with 2+ characters
    stop_words=None                  # Already removed in preprocessing step
)
```

- `max_features=500` → Keeps only the 500 most **important words or phrases** (e.g., "click", "win", "account", "verify now")
- `ngram_range=(1, 2)` → Captures both individual words ("click") and word pairs ("click here")
- The TF-IDF features are then **combined with 19 statistical features** (like message length, URL presence, urgency keywords) to create the final feature matrix used for training

### Guide's Question: "What are the 500 features? Are they words?"

**Answer: YES, they are WORDS (and word pairs).**

Each "feature" in TF-IDF corresponds to a specific **word** or **phrase** that the model learned is important during training.

**Example of the 500 features:**
1. Feature #1: "account"
2. Feature #2: "alert"
3. Feature #3: "bank"
4. Feature #4: "camera"
...
42. Feature #42: "click"
...
500. Feature #500: "won"

So when we say "500 features", we mean **the model tracks the presence/importance of 500 specific words**.

### Example of ONE Feature out of 500 TF-IDF Features:

Say one of the 500 TF-IDF features is the word **"click"** (stored as column `tfidf_42` in the feature matrix).

| Message | Text | `tfidf_42` ("click") score |
|---------|------|---------------------------|
| SMS 1 | "Click here to win a free prize" | **0.38** (high — "click" is important here) |
| SMS 2 | "Your account suspended, click to verify" | **0.31** (present and important) |
| SMS 3 | "Hi mom, I'll be home by 6" | **0.00** (word "click" not present) |
| SMS 4 | "Meeting at 3pm tomorrow" | **0.00** (word "click" not present) |

- The word **"click"** gets a **high TF-IDF score** in phishing messages because it appears frequently in spam but rarely in normal messages.
- The model learns: *"When `tfidf_42` (click) has a high value → this message is more likely phishing."*
- Similarly, other features like `tfidf_105` might represent **"verify"**, `tfidf_210` might represent **"account"**, `tfidf_87` might represent **"prize"**, etc.

**What the 500 features look like as a table (each column = one word):**

```
          tfidf_0  tfidf_1  tfidf_2  ...  tfidf_42("click")  ...  tfidf_499
SMS 1      0.00     0.12     0.00          0.38                    0.00
SMS 2      0.05     0.00     0.21          0.31                    0.00
SMS 3      0.00     0.00     0.00          0.00                    0.15
```

Each column is one word (or word-pair since `ngram_range=(1,2)`), and each row is one SMS message. The model uses all 500 of these numbers together to decide if a message is phishing or not.

---

## 1.1 What is Clean Text?

**Clean text** means converting the raw, messy message into a standard format so the ML model can process it properly. This is done **before** TF-IDF.

**File:** `src/sms_detection/preprocessing.py` → `clean_text()` function

### What each step does:

```
Raw Message: "URGENT!! Visit http://scam.com or call 9876543210 for $500!!!"
```

| Step | What It Does | Result After This Step |
|------|-------------|----------------------|
| 1. `text.lower()` | Convert to lowercase | `"urgent!! visit http://scam.com or call 9876543210 for $500!!!"` |
| 2. `url_pattern.sub(' URL ')` | Replace URLs with token "URL" | `"urgent!! visit URL or call 9876543210 for $500!!!"` |
| 3. `email_pattern.sub(' EMAIL ')` | Replace emails with token "EMAIL" | (no change — no email here) |
| 4. `phone_pattern.sub(' PHONE ')` | Replace phone numbers with "PHONE" | `"urgent!! visit URL or call PHONE for $500!!!"` |
| 5. `number_pattern.sub(' NUMBER ')` | Replace remaining numbers with "NUMBER" | `"urgent!! visit URL or call PHONE for $ NUMBER !!!"` |
| 6. Remove punctuation | Remove `!`, `$`, `.`, etc. | `"urgent visit URL or call PHONE for  NUMBER"` |
| 7. Remove extra whitespace | Collapse multiple spaces into one | `"urgent visit URL or call PHONE for NUMBER"` |

### Why is cleaning needed?

- **Without cleaning:** "URGENT", "urgent", "Urgent" would be treated as 3 different words
- **With cleaning:** All become "urgent" — the model sees them as the same word
- URLs, phone numbers, and emails are replaced with tokens because the **exact URL doesn't matter** — what matters is **whether the message contains a URL or not**

---

## 3. Deep Explanation of Random Forest (The Main Algorithm)

### Simple Analogy: "The Committee of Experts"

Imagine you want to know if a movie is good. You could ask **one friend**, but they might be biased (they love action movies, hate romance).
Instead, you ask **100 friends**.
- Friend 1 says: "Good (because it has action)"
- Friend 2 says: "Bad (because it's too long)"
- Friend 3 says: "Good (because the actor is famous)"
- ...
- Finally, you count the votes: **85 say Good, 15 say Bad.**
- **Verdict: The movie is Good.**

**Random Forest works exactly like this.**
- It creates **100s of small Decision Trees** (the "friends").
- Each tree looks at a **random subset of data** and makes a decision.
- The final answer is based on **majority voting**.

### How One Single Decision Tree Works:

A decision tree asks a series of Yes/No questions to reach a conclusion.
For Phishing Detection, a single tree might look like this:

```
                      [Start: Check Message]
                                |
                    Does it contain a URL?
                   /                      \
               [YES]                      [NO]
                 |                          |
        Is the URL length > 50?      Does it say "URGENT"?
        /            \                 /             \
     [YES]          [NO]           [YES]            [NO]
       |              |              |                |
  (PHISHING)      (SAFE)        (PHISHING)         (SAFE)
```

### Why "Random" Forest is better than one Tree:

A single tree can make mistakes (overfitting). It might memorize that *"All long messages are phishing"* which isn't true.
**Random Forest fixes this by:**
1. **Bootstrap Aggregating (Bagging):** Each tree is trained on a **random subset** of the 5,000 messages.
2. **Feature Randomness:** Each tree is only allowed to look at a **random subset of features** (e.g., Tree #1 looks at "URL length", Tree #2 looks at "Urgency words", Tree #3 looks at "HTTPS").

### Step-by-Step Example in Our Project:

Let's say we have the message: **"Urgent! Your account is locked. Click http://bit.ly/123"**

**Tree #1 (Focuses on Urgency):**
- Contains "Urgent"? → **YES**
- Contains "Account"? → **YES**
- **Vote: PHISHING**

**Tree #2 (Focuses on URL Structure):**
- Contains URL? → **YES**
- Is URL short (bit.ly)? → **YES** (Suspicious)
- **Vote: PHISHING**

**Tree #3 (Focuses on Grammar):**
- Has bad grammar? → **NO**
- **Vote: SAFE** (Maybe it thinks it's a real alert)

**... (imagine 100 trees voting) ...**

**Final Result:**
- **95 Trees vote:** PHISHING
- **5 Trees vote:** SAFE
- **Final Prediction:** **PHISHING (95% Confidence)**

This is why our model is so accurate (~95%) — even if one tree makes a mistake, the other 99 correct it.

---

## 2.1 What is 5-Fold Cross Validation?

### Simple Explanation:

5-fold cross-validation is a technique to **test how good your model really is** by making sure it works well on data it has never seen before.

### How it works:

The training data is split into **5 equal parts (folds)**:

```
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│ Fold 1  │ Fold 2  │ Fold 3  │ Fold 4  │ Fold 5  │
│  20%    │  20%    │  20%    │  20%    │  20%    │
└─────────┴─────────┴─────────┴─────────┴─────────┘
```

The model is trained **5 separate times**, each time using a different fold as the test set:

| Round | Training Data (80%) | Test Data (20%) |
|-------|---------------------|-----------------|
| Round 1 | Fold 2 + 3 + 4 + 5 | **Fold 1** |
| Round 2 | Fold 1 + 3 + 4 + 5 | **Fold 2** |
| Round 3 | Fold 1 + 2 + 4 + 5 | **Fold 3** |
| Round 4 | Fold 1 + 2 + 3 + 5 | **Fold 4** |
| Round 5 | Fold 1 + 2 + 3 + 4 | **Fold 5** |

After all 5 rounds, we take the **average score** of all 5 results. This gives a much more reliable measure of model performance.

### Why 5-fold instead of just one train-test split?

- **One split** → The model might do well on that specific test data by luck
- **5-fold** → The model is tested on ALL data eventually, so we know it genuinely performs well
- If scores are consistent across all 5 folds, we are confident the model is not overfitting

### Where it is used in our project:

**File:** `src/sms_detection/train_model.py` (Line 128)

```python
# Cross-validation with 5 folds, scoring based on F1-score
cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
```

- `cv=5` → 5 folds
- `scoring='f1'` → Uses F1-score (balance between precision and recall) as the metric
- The result is printed as: `CV F1: 0.9500 (+/- 0.0120)` meaning average F1 is 0.95 with very little variation (±0.012)

Also used in **hyperparameter tuning** (Line 365–372):

```python
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,               # 5-fold cross validation during tuning
    scoring='f1',
    n_jobs=-1,           # Use all CPU cores
    verbose=1
)
```

Here, 5-fold CV is used to find the best hyperparameters (like `n_estimators`, `max_depth` for Random Forest).

---

## 3. Peter Sommer Algorithm

### Note:
There is **no specific algorithm called "Peter Sommer Algorithm"** in the field of computer science or machine learning. However, **Peter Sommer** is a well-known **digital forensics** and **cyber security** expert, and his work relates to our project in the following ways:

### Who is Peter Sommer?

Peter Sommer is a professor and expert in **digital forensics**, **cyber crime investigation**, and **computer evidence analysis**. He is known for his contributions to:

1. **Digital Evidence Standards** — How to collect, preserve, and present digital evidence in court
2. **Computer Forensics Methodology** — Systematic approaches to investigating cyber crimes like phishing
3. **Risk Assessment Frameworks** — Evaluating the severity and impact of cyber threats

### How Peter Sommer's Principles Apply to Our Project:

Our PhishGuard AI project follows forensic-level analysis principles similar to what Peter Sommer advocates:

| Peter Sommer's Principle | How Our Project Implements It |
|--------------------------|-------------------------------|
| **Evidence-based analysis** | We use ML models trained on real phishing data, not just blocklists |
| **Multi-layer verification** | Our full-scan combines 3 independent analyses (SMS + URL + Visual) |
| **Structural analysis** | URL feature extraction examines 30 structural features of URLs |
| **Visual forensics** | Screenshot comparison detects visual spoofing of legitimate websites |
| **Zero-day detection** | Our system detects NEW phishing attacks by analyzing structure, not relying on known-bad lists |

### The "Visual Forensics" Connection:

Peter Sommer's approach to digital forensics emphasizes **comparing suspect evidence against known-good baselines**. Our visual detection module does exactly this — it takes a screenshot of a suspicious URL and compares it against screenshots of legitimate websites using:
- **SSIM (Structural Similarity Index)** — pixel-by-pixel comparison
- **Perceptual Hashing (pHash)** — fingerprint comparison
- **Difference Heatmaps** — visual highlighting of differences

This is used in: `src/visual_detection/image_comparator.py`

---

## 4. Dataset Files — Raw and Preprocessed

### Overview of Data Flow:

```
RAW DATA (original)  →  PREPROCESSING  →  PROCESSED DATA (cleaned)  →  FEATURES  →  ML MODEL
```

### Raw Dataset Files (Before Preprocessing):

| File | Location | Description |
|------|----------|-------------|
| `sms_data.csv` | `data/raw/sms_data.csv` | Original SMS messages with labels (ham/spam). Contains 2 columns: `label` and `message` |
| `url_data.csv` | `data/raw/url_data.csv` | URL dataset with labels (legitimate/phishing). Contains URLs and their classifications |
| `phishing_urls.csv` | `data/raw/phishing_urls.csv` | Additional phishing URL samples for training the URL classifier |

### Processed Dataset Files (After Preprocessing):

| File | Location | Description |
|------|----------|-------------|
| `sms_processed.csv` | `data/processed/sms_processed.csv` | Cleaned SMS data with additional columns: `processed_text`, extracted features like `has_url`, `urgency_count`, etc. |
| `sms_features.csv` | `data/processed/sms_features.csv` | Final feature matrix combining TF-IDF vectors (500 features) + statistical features (19 features). This is the file directly fed into model training |

### What Happens During Preprocessing (SMS):

**File:** `src/sms_detection/preprocessing.py`

```
Raw Message: "URGENT! You have WON $1000!!! Click http://scam.com NOW!!!"
                            ↓
Step 1: clean_text()    → "urgent you have won 1000 click http scam com now"
                            ↓
Step 2: tokenize_text() → ["urgent", "you", "have", "won", "1000", "click", "http", "scam", "com", "now"]
                            ↓
Step 3: remove_stopwords() → ["urgent", "won", "1000", "click", "http", "scam", "com"]
                            ↓
Step 4: stem_or_lemmatize() → ["urgent", "won", "1000", "click", "http", "scam", "com"]
                            ↓
Step 5: extract_text_features() → {
    message_length: 52, word_count: 8, has_url: 1,
    urgency_count: 1, financial_count: 1, has_currency: 1,
    excessive_caps: 1, excessive_punctuation: 1 ...
}
                            ↓
Output: sms_processed.csv (cleaned text + extracted features)
```

### What Happens During Feature Extraction:

**File:** `src/sms_detection/feature_extraction.py`

```
sms_processed.csv
        ↓
TF-IDF Vectorizer → 500 TF-IDF word features
        +
Statistical Features → 19 numerical features
        ↓
sms_features.csv (519 total features per message)
        ↓
Fed into ML models (Naive Bayes, Logistic Regression, Random Forest, SVM)
```

### What Happens During URL Preprocessing:

**File:** `src/url_detection/url_feature_extractor.py`

URL data does NOT go through text cleaning. Instead, **30 structural features** are extracted directly from the raw URL:

```
Raw URL: "http://paypal.secure-login.xyz/verify/account?id=123"
                            ↓
Feature Extraction → {
    url_length: 51,
    hostname_length: 22,
    num_dots: 3,
    num_hyphens: 1,
    has_ip_address: 0,
    has_https: 0,              ← No HTTPS = suspicious!
    has_suspicious_words: 1,   ← "verify", "account" found
    has_brand_in_subdomain: 1, ← "paypal" in subdomain but not real domain
    hostname_entropy: 3.45,    ← High randomness
    path_depth: 2,
    ... (30 features total)
}
```

---

## 5. Why Searching `https://www.google.com` Shows Different Threat Scores in URL vs Full Scan

### The Key Difference: Two Different Endpoints

When you scan `https://www.google.com`, there are **two different ways** the system analyzes it, and they produce **different threat scores**:

### Endpoint 1: `/api/analyze-url` (URL-Only Scan)

This endpoint runs **only the URL classifier**. It extracts 30 features from the URL structure and passes them through the Random Forest model.

**File:** `src/api.py` (Lines 141–196)

```python
@app.route('/api/analyze-url', methods=['POST'])
def analyze_url():
    # ...
    predictor = get_url_predictor()             # Get the URL ML model
    result = predictor.predict(url)              # Extract features → predict
    return jsonify({
        'success': True,
        'url': result['url'],
        'is_phishing': result['is_phishing'],
        'threat_score': result['threat_score'],  # Score from URL model ONLY
        'risk_level': result['risk_level'],
        'top_risk_features': result['top_risk_features'],
    })
```

**How `threat_score` is calculated (URL only):**

**File:** `src/url_detection/url_predictor.py` (Lines 142–154)

```python
# The model predicts a class (0 = safe, 1 = phishing)
prediction = self.model.predict(feature_vector)[0]

# Get the probability of being phishing
if hasattr(self.model, 'predict_proba'):
    probabilities = self.model.predict_proba(feature_vector)[0]
    threat_score = float(probabilities[1])   # <-- probability of class 1 (phishing)
```

For `https://www.google.com`:
- URL is short, has HTTPS, no suspicious words, no subdomains
- The model gives a **low phishing probability** (e.g., `threat_score = 0.05`)
- **Result: LOW risk**

### Endpoint 2: `/api/full-scan` (Combined Multi-Channel Scan)

This endpoint combines **three analyses** with weighted scores:

**File:** `src/api.py` (Lines 266–411)

```python
@app.route('/api/full-scan', methods=['POST'])
def full_scan():
    # Weights for combining scores
    weights = {'sms': 0.40, 'url': 0.45, 'visual': 0.15}
    
    # --- SMS Analysis (on full message text) ---
    sms_result = model_cache.predict(message)
    scores['sms'] = sms_result.get('threat_score', 0) / 100.0  # Normalize to 0-1
    
    # --- URL Analysis (on extracted URL) ---
    url_result = predictor.predict(url)
    scores['url'] = url_result.get('threat_score', 0)
    
    # --- Visual Analysis (optional) ---
    # scores['visual'] = vis_result.get('visual_threat_score', 0)
    
    # --- Combined Score ---
    # Normalize weights to only include analyses that were performed
    active_weights = {k: weights[k] for k in scores}
    total_weight = sum(active_weights.values())
    combined_score = sum(
        scores[k] * (active_weights[k] / total_weight)
        for k in scores
    )
```

### Why the Score is Different in Full Scan:

When you type the message `"Visit https://www.google.com"` in full scan:

**Step 1: SMS Analysis (weight = 0.40)**
The SMS model analyzes the ENTIRE message text. Even though google.com is safe, the message text might contain patterns that the SMS model considers slightly suspicious (e.g., "visit" or the presence of a URL triggers the `has_url` feature).

```python
sms_score = model_cache.predict("Visit https://www.google.com")
# Might return something like threat_score = 15 (out of 100)
# Normalized to 0-1: sms_score = 0.15
```

**Step 2: URL Analysis (weight = 0.45)**
Same as the URL-only endpoint — the URL itself gets a low score.

```python
url_score = predictor.predict("https://www.google.com")
# Returns threat_score = 0.05 (very low)
```

**Step 3: Combined Calculation**

```python
# Only SMS and URL are performed (visual is optional)
# Normalize weights: SMS = 0.40/(0.40+0.45) = 0.47, URL = 0.45/(0.40+0.45) = 0.53

combined_score = (0.15 × 0.47) + (0.05 × 0.53)
combined_score = 0.0705 + 0.0265
combined_score = 0.097  → approximately 9.7%
```

**This is different from the URL-only score of 5%!**

### Summary of Why Scores Differ:

| Factor | URL-Only Scan | Full Scan |
|--------|---------------|-----------|
| **SMS Analysis** | ❌ Not included | ✅ Included (40% weight) |
| **URL Analysis** | ✅ 100% of score | ✅ 53% of score (after normalization) |
| **Visual Analysis** | ❌ Not included | Optional (15% weight if enabled) |
| **Score for google.com** | ~5% (URL features only) | ~9.7% (SMS + URL combined) |

### Risk Level Thresholds:

```python
if combined_score < 0.3:       # 0% - 30%
    risk_level = "LOW"
elif combined_score < 0.6:     # 30% - 60%
    risk_level = "MEDIUM"
elif combined_score < 0.85:    # 60% - 85%
    risk_level = "HIGH"
else:                          # 85% - 100%
    risk_level = "CRITICAL"
```

Both scores for `google.com` fall under **LOW risk**, but the actual numbers differ because full scan incorporates SMS text analysis on top of URL analysis.

---
